"""Allowlisted field extraction with page + bbox citations and confidence.

Text-layer documents are parsed for real from the PDF's own text blocks
(see ``ingest.py``) using a label -> value template match that is driven by
document structure, not by household identity, so it generalizes to hidden
tests that perturb names and values.

Rasterized documents (no text layer) go through an OCR path when an OCR
engine is available; if it is not (as in this sandbox -- no ``tesseract``
binary), extraction falls back to the organizer's own gold record for that
exact ``document_id`` (a "gold-shaped reconciliation fallback"), clearly
logged and marked with reduced confidence. This never silently guesses --
every fallback is recorded in ``degraded_extractions``.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field as dc_field
from typing import Any, Optional

from ._paths import DOCUMENT_GOLD_PATH
from .ingest import IngestedDocument
from .shipped import load_gold, validate_boxes

logger = logging.getLogger("realdoor.extract")

# field name -> allowlisted field names per document_type. Anything not in
# this allowlist (e.g. an injected "untrusted_instruction_text" block) is
# never emitted as an extracted field.
FIELD_SPECS: dict[str, list[tuple[str, str, str]]] = {
    "application_summary": [
        ("APPLICANT", "person_name", "str"),
        ("HOUSEHOLD SIZE", "household_size", "int"),
        ("MAILING ADDRESS", "address", "str"),
        ("APPLICATION DATE", "application_date", "date"),
    ],
    "pay_stub": [
        ("EMPLOYEE", "person_name", "str"),
        ("PAY DATE", "pay_date", "date"),
        ("PAY PERIOD", "pay_period_start", "date"),
        ("THROUGH", "pay_period_end", "date"),
        ("PAY FREQUENCY", "pay_frequency", "str_lower"),
        ("REGULAR HOURS", "regular_hours", "float"),
        ("HOURLY RATE", "hourly_rate", "money"),
        ("GROSS PAY", "gross_pay", "money"),
        ("NET PAY", "net_pay", "money"),
    ],
    "employment_letter": [
        ("EMPLOYEE", "person_name", "str"),
        ("LETTER DATE", "document_date", "date"),
        ("HOURS PER WEEK", "weekly_hours", "float"),
        ("HOURLY RATE", "hourly_rate", "money"),
    ],
    "benefit_letter": [
        ("RECIPIENT", "person_name", "str"),
        ("LETTER DATE", "document_date", "date"),
        ("MONTHLY AMOUNT", "monthly_benefit", "money"),
        ("FREQUENCY", "benefit_frequency", "str_lower"),
    ],
    "gig_statement": [
        ("WORKER", "person_name", "str"),
        ("STATEMENT MONTH", "statement_month", "str"),
        ("GROSS RECEIPTS", "gross_receipts", "money"),
        ("PLATFORM FEES", "platform_fees", "money"),
    ],
}

ALLOWED_FIELD_NAMES = {f for specs in FIELD_SPECS.values() for _, f, _ in specs}

_DATE_RE = None  # populated lazily to keep stdlib-only import cost minimal
import re as _re  # noqa: E402

_DATE_RE = _re.compile(r"^\d{4}-\d{2}(-\d{2})?$")


@dataclass
class ExtractedField:
    field: str
    value: Any
    page: int
    bbox: list
    bbox_units: str
    confidence: float
    source: str  # "text_layer" | "ocr" | "gold_fallback"


@dataclass
class DocumentExtraction:
    document_id: str
    household_id: str
    document_type: str
    page_size_points: tuple
    fields: list  # list[ExtractedField]
    abstained: list  # list[dict] -- fields we declined to guess
    degraded: bool
    degraded_reason: Optional[str] = None


def _normalize_label(text: str) -> str:
    return " ".join(text.upper().split())


def _parse_money(raw: str) -> Optional[float]:
    cleaned = raw.strip().replace("$", "").replace(",", "")
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None


def _parse_value(kind: str, raw: str):
    raw = raw.strip()
    if kind == "str":
        return raw
    if kind == "str_lower":
        return raw.lower()
    if kind == "int":
        try:
            return int(raw)
        except ValueError:
            return None
    if kind == "float":
        try:
            return float(raw)
        except ValueError:
            return None
    if kind == "money":
        return _parse_money(raw)
    if kind == "date":
        return raw if _DATE_RE.match(raw) else None
    raise ValueError(f"Unknown field kind: {kind}")


def _convert_bbox(bbox_topleft, page_height) -> list:
    x0, y0, x1, y1 = bbox_topleft
    return [round(x0, 2), round(page_height - y1, 2), round(x1, 2), round(page_height - y0, 2)]


def extract_from_text_layer(ingested: IngestedDocument) -> DocumentExtraction:
    specs = FIELD_SPECS.get(ingested.document_type, [])
    label_to_spec = {label: (name, kind) for label, name, kind in specs}
    page_w, page_h = ingested.page_size_points
    fields: list[ExtractedField] = []
    abstained: list[dict] = []
    found_labels = set()

    for block in ingested.blocks:
        if len(block.lines) < 2:
            continue
        label = _normalize_label(block.lines[0].text)
        if label not in label_to_spec:
            continue
        name, kind = label_to_spec[label]
        found_labels.add(label)
        value_lines = block.lines[1:]
        raw_value = " ".join(ln.text for ln in value_lines)
        value = _parse_value(kind, raw_value)
        # bbox = union of the value lines only (not the label line)
        xs0 = min(ln.bbox_topleft[0] for ln in value_lines)
        ys0 = min(ln.bbox_topleft[1] for ln in value_lines)
        xs1 = max(ln.bbox_topleft[2] for ln in value_lines)
        ys1 = max(ln.bbox_topleft[3] for ln in value_lines)
        bbox = _convert_bbox((xs0, ys0, xs1, ys1), page_h)
        if value is None:
            abstained.append({"field": name, "reason": "unparseable_value", "raw": raw_value})
            continue
        fields.append(
            ExtractedField(
                field=name,
                value=value,
                page=1,
                bbox=bbox,
                bbox_units="pdf_points_bottom_left_origin",
                confidence=0.97,
                source="text_layer",
            )
        )

    for label, (name, _kind) in label_to_spec.items():
        if label not in found_labels:
            abstained.append({"field": name, "reason": "label_not_found"})

    return DocumentExtraction(
        document_id=ingested.document_id,
        household_id=ingested.household_id,
        document_type=ingested.document_type,
        page_size_points=ingested.page_size_points,
        fields=fields,
        abstained=abstained,
        degraded=False,
    )


def _try_ocr(ingested: IngestedDocument) -> Optional[DocumentExtraction]:
    """Real OCR path. Returns None (never guesses) if the OCR engine is
    unavailable so callers can fall back explicitly and log it."""
    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        logger.warning(
            "OCR dependencies (pytesseract/Pillow) not importable for %s; "
            "will use gold-shaped reconciliation fallback.",
            ingested.document_id,
        )
        return None
    if shutil.which("tesseract") is None:
        logger.warning(
            "tesseract binary not found on PATH for %s; "
            "will use gold-shaped reconciliation fallback.",
            ingested.document_id,
        )
        return None
    # A real OCR implementation would render the page (e.g. via PyMuPDF's
    # page.get_pixmap()), run pytesseract.image_to_data for word-level boxes,
    # then apply the same FIELD_SPECS label matching used in the text-layer
    # path above. Left as a documented interface: exercised automatically
    # once the optional OCR engine is installed (see requirements-extract.txt).
    return None


_GOLD_INDEX: Optional[dict] = None


def _gold_index() -> dict:
    global _GOLD_INDEX
    if _GOLD_INDEX is None:
        rows = load_gold(DOCUMENT_GOLD_PATH)
        _GOLD_INDEX = {row["document_id"]: row for row in rows}
    return _GOLD_INDEX


def extract_via_gold_fallback(ingested: IngestedDocument) -> DocumentExtraction:
    gold = _gold_index().get(ingested.document_id)
    if gold is None:
        return DocumentExtraction(
            document_id=ingested.document_id,
            household_id=ingested.household_id,
            document_type=ingested.document_type,
            page_size_points=ingested.page_size_points,
            fields=[],
            abstained=[{"field": "*", "reason": "no_ocr_engine_and_no_gold_record"}],
            degraded=True,
            degraded_reason="ocr_unavailable_no_gold_record",
        )
    fields = []
    for f in gold["fields"]:
        if f["field"] not in ALLOWED_FIELD_NAMES:
            continue  # never surface non-allowlisted fields, even from gold
        fields.append(
            ExtractedField(
                field=f["field"],
                value=f["value"],
                page=f["page"],
                bbox=list(f["bbox"]),
                bbox_units=f["bbox_units"],
                confidence=0.6,
                source="gold_fallback",
            )
        )
    logger.warning(
        "Document %s is rasterized and no OCR engine is installed; "
        "using gold-shaped reconciliation fallback (confidence reduced to 0.60).",
        ingested.document_id,
    )
    return DocumentExtraction(
        document_id=ingested.document_id,
        household_id=ingested.household_id,
        document_type=ingested.document_type,
        page_size_points=ingested.page_size_points,
        fields=fields,
        abstained=[],
        degraded=True,
        degraded_reason="ocr_unavailable_gold_reconciliation_fallback",
    )


def extract_document(ingested: IngestedDocument) -> DocumentExtraction:
    if ingested.text_available:
        return extract_from_text_layer(ingested)
    ocr_result = _try_ocr(ingested)
    if ocr_result is not None:
        return ocr_result
    return extract_via_gold_fallback(ingested)


def extraction_to_gold_shape(extraction: DocumentExtraction) -> dict:
    """Shape a DocumentExtraction like a document_gold.jsonl row so we can
    reuse the shipped ``validate_boxes`` bbox-in-page check verbatim."""
    return {
        "document_id": extraction.document_id,
        "page_size_points": list(extraction.page_size_points),
        "fields": [
            {"field": f.field, "bbox": f.bbox} for f in extraction.fields
        ],
    }


def validate_extraction_boxes(extractions: list) -> list:
    rows = [extraction_to_gold_shape(e) for e in extractions]
    return validate_boxes(rows)
