"""Untrusted-document ingest: load a PDF, route text-layer vs OCR, and
quarantine any embedded instruction text before it ever reaches extraction.

Every document is treated as *untrusted data* (CH-SAFETY-001): nothing read
from a PDF is ever executed, and any text that looks like an attempt to
steer the system (prompt injection) is logged to a quarantine record and
excluded from the allowlisted field set -- it never changes a computed
value or a readiness outcome.
"""
from __future__ import annotations

import base64
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF -- text-layer + raster access only, never executed content

from ._paths import DOCUMENTS_DIR, MANIFEST_PATH, QUARANTINE_LOG_PATH

# Zoom factor used when rendering a page facsimile for the Profile stage
# evidence viewer. Kept low-resolution to keep the single-file app small.
PAGE_IMAGE_ZOOM = 1.3

# Patterns that indicate embedded-instruction / prompt-injection text inside
# a "document". Matched case-insensitively against extracted text only --
# never interpreted as instructions.
INJECTION_PATTERNS = [
    re.compile(r"ignore (all )?(prior|previous|system) instructions", re.I),
    re.compile(r"reveal (the )?system prompt", re.I),
    re.compile(r"mark (this )?applicant approved", re.I),
    re.compile(r"disregard (the )?(above|previous) instructions", re.I),
    re.compile(r"you are now", re.I),
    re.compile(r"untrusted document text", re.I),
]


@dataclass
class Line:
    text: str
    bbox_topleft: tuple  # (x0, y0, x1, y1), PyMuPDF top-left origin


@dataclass
class Block:
    number: int
    lines: list


@dataclass
class QuarantineEntry:
    document_id: str
    page: int
    text: str
    reason: str
    bbox_topleft: tuple


@dataclass
class IngestedDocument:
    document_id: str
    household_id: str
    document_type: str
    file_name: str
    rasterized: bool
    contains_adversarial_text_manifest: bool
    page_size_points: tuple  # (width, height)
    page_count: int
    text_available: bool
    blocks: list  # list[Block]
    quarantine: list  # list[QuarantineEntry]
    page_image_data_uri: Optional[str] = None


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row["rasterized"] = row["rasterized"].strip().lower() == "true"
            row["contains_adversarial_text"] = row["contains_adversarial_text"].strip().lower() == "true"
            rows.append(row)
        return rows


def _extract_blocks(page: "fitz.Page") -> list[Block]:
    d = page.get_text("dict")
    blocks = []
    for b in d.get("blocks", []):
        if "lines" not in b:
            continue
        lines = []
        for ln in b["lines"]:
            text = "".join(span["text"] for span in ln["spans"]).strip()
            if not text:
                continue
            lines.append(Line(text=text, bbox_topleft=tuple(round(v, 2) for v in ln["bbox"])))
        if lines:
            blocks.append(Block(number=b["number"], lines=lines))
    return blocks


def _scan_for_injection(document_id: str, blocks: list[Block]) -> list[QuarantineEntry]:
    entries = []
    for b in blocks:
        joined = " ".join(ln.text for ln in b.lines)
        for pattern in INJECTION_PATTERNS:
            if pattern.search(joined):
                # Quarantine the whole block once, not once per pattern.
                bbox = b.lines[0].bbox_topleft
                entries.append(
                    QuarantineEntry(
                        document_id=document_id,
                        page=1,
                        text=joined,
                        reason=f"embedded_instruction:{pattern.pattern}",
                        bbox_topleft=bbox,
                    )
                )
                break
    return entries


def _ingest_from_open_doc(
    doc: "fitz.Document",
    *,
    document_id: str,
    household_id: str,
    document_type: str,
    file_name: str,
    manifest_rasterized_flag: bool,
    contains_adversarial_text_manifest: bool,
    render_page_image: bool,
    log_quarantine: bool,
) -> IngestedDocument:
    """Shared ingest core: given an already-open ``fitz.Document`` (from a
    file path or from in-memory bytes -- the caller owns opening/closing
    it), extract page-0 text blocks, auto-detect rasterization, scan for
    embedded-instruction text, optionally render a page facsimile, and
    return the resulting :class:`IngestedDocument`."""
    page = doc[0]
    page_size = (round(page.rect.width, 2), round(page.rect.height, 2))
    blocks = _extract_blocks(page)
    # Auto-detect rasterization independent of any manifest flag, as a
    # defense-in-depth cross-check (a hidden test -- or an arbitrary
    # upload -- could ship a PDF nothing described in advance).
    total_text_chars = sum(len(ln.text) for b in blocks for ln in b.lines)
    text_available = total_text_chars > 0
    quarantine = _scan_for_injection(document_id, blocks)
    page_image_data_uri = None
    if render_page_image:
        pix = page.get_pixmap(matrix=fitz.Matrix(PAGE_IMAGE_ZOOM, PAGE_IMAGE_ZOOM))
        png_b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
        page_image_data_uri = f"data:image/png;base64,{png_b64}"
    ingested = IngestedDocument(
        document_id=document_id,
        household_id=household_id,
        document_type=document_type,
        file_name=file_name,
        rasterized=manifest_rasterized_flag or not text_available,
        contains_adversarial_text_manifest=contains_adversarial_text_manifest,
        page_size_points=page_size,
        page_count=doc.page_count,
        text_available=text_available,
        blocks=blocks,
        quarantine=quarantine,
        page_image_data_uri=page_image_data_uri,
    )
    if quarantine and log_quarantine:
        _log_quarantine(quarantine)
    return ingested


def ingest_document(
    manifest_row: dict, documents_dir: Path = DOCUMENTS_DIR, render_page_image: bool = False
) -> IngestedDocument:
    file_name = manifest_row["file_name"]
    pdf_path = documents_dir / file_name
    doc = fitz.open(pdf_path)
    try:
        return _ingest_from_open_doc(
            doc,
            document_id=manifest_row["document_id"],
            household_id=manifest_row["household_id"],
            document_type=manifest_row["document_type"],
            file_name=file_name,
            manifest_rasterized_flag=manifest_row["rasterized"],
            contains_adversarial_text_manifest=manifest_row["contains_adversarial_text"],
            render_page_image=render_page_image,
            log_quarantine=True,
        )
    finally:
        doc.close()


def ingest_bytes(
    document_id: str,
    household_id: str,
    document_type: str,
    pdf_bytes: bytes,
    render_page_image: bool = True,
) -> IngestedDocument:
    """Same as :func:`ingest_document` but for an in-memory upload rather
    than a manifest-driven path on disk: no manifest row exists, so
    ``contains_adversarial_text_manifest`` is always False (that field only
    reflects the organizer's own manifest flag for the 24 fixed docs) and
    rasterization is purely auto-detected. Never writes to disk -- the
    quarantine log write is skipped here (the caller gets the same
    quarantine records back in-memory via ``IngestedDocument.quarantine``
    and is responsible for surfacing them; there is no per-upload disk log
    to keep, and this keeps the function safe to call on a read-only
    deployment filesystem)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return _ingest_from_open_doc(
            doc,
            document_id=document_id,
            household_id=household_id,
            document_type=document_type,
            file_name=f"{document_id}.pdf",
            manifest_rasterized_flag=False,
            contains_adversarial_text_manifest=False,
            render_page_image=render_page_image,
            log_quarantine=False,
        )
    finally:
        doc.close()


def _log_quarantine(entries: list[QuarantineEntry]) -> None:
    with QUARANTINE_LOG_PATH.open("a", encoding="utf-8") as f:
        for e in entries:
            f.write(
                json.dumps(
                    {
                        "document_id": e.document_id,
                        "page": e.page,
                        "text": e.text,
                        "reason": e.reason,
                        "bbox_topleft": e.bbox_topleft,
                        "action": "quarantined_never_executed",
                    }
                )
                + "\n"
            )


def reset_quarantine_log() -> None:
    if QUARANTINE_LOG_PATH.exists():
        QUARANTINE_LOG_PATH.unlink()


def scan_untrusted_text(text: str) -> Optional[str]:
    """Public helper for the safety strip: classify pasted text as
    containing an embedded instruction, without ever acting on it."""
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None
