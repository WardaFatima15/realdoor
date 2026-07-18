"""Orchestrates a household's documents -> a schema-valid submission JSON +
a richer profile.json for the UI.

Pipeline stages, per household:
  1. ingest.py  -- load each PDF as untrusted data, quarantine embedded instructions
  2. extract.py -- allowlisted fields with page + bbox + confidence, abstain don't guess
  3. readiness.py -- annualize (shipped calculate.py), threshold lookup, reasons
  4. assemble citations (rule citations + per-field document citations)
  5. validate against submission.schema.json before writing anything out
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ._paths import (
    CHECKLISTS_PATH,
    PROFILES_DIR,
    SUBMISSIONS_DIR,
    SUBMISSION_SCHEMA_PATH,
)
from .extract import extract_document
from .ingest import ingest_document, load_manifest, reset_quarantine_log
from .readiness import evaluate_household
from .rules_qa import cite

try:
    import jsonschema

    _HAVE_JSONSCHEMA = True
except ImportError:  # keep the deterministic core runnable stdlib-only
    _HAVE_JSONSCHEMA = False


def _load_checklists() -> list[dict]:
    with CHECKLISTS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _household_size_from_checklist(checklist_row: dict) -> int:
    return checklist_row["household_size"]


def build_household(household_id: str, household_size: int, manifest_rows: list, render_images: bool = False) -> dict:
    """Runs ingest -> extract -> readiness for one household and returns a
    dict with both the schema-valid `submission` and the richer `profile`."""
    hh_rows = [r for r in manifest_rows if r["household_id"] == household_id]
    extractions = []
    ingested_docs = []
    quarantine_events = []
    degraded_docs = []

    for row in hh_rows:
        ingested = ingest_document(row, render_page_image=render_images)
        ingested_docs.append(ingested)
        if ingested.quarantine:
            for q in ingested.quarantine:
                quarantine_events.append(
                    {
                        "document_id": q.document_id,
                        "reason": q.reason,
                        "text": q.text,
                        "action": "quarantined_never_executed",
                    }
                )
        extraction = extract_document(ingested)
        extractions.append(extraction)
        if extraction.degraded:
            degraded_docs.append({"document_id": extraction.document_id, "reason": extraction.degraded_reason})

    result = evaluate_household(household_id, household_size, extractions)

    # --- citations: rule citations + per-field document citations ---
    citations = [
        cite("CH-INCOME-001"),
        cite("CH-READINESS-001"),
    ]
    if result.threshold is not None:
        citations.append(cite("HUD-MTSP-002"))
    else:
        citations.append(cite("HUD-MTSP-001"))
    for extraction in extractions:
        for f in extraction.fields:
            if f.field in ("gross_pay", "monthly_benefit", "gross_receipts", "regular_hours", "hourly_rate"):
                citations.append(
                    {
                        "document_id": extraction.document_id,
                        "field": f.field,
                        "page": f.page,
                        "bbox": f.bbox,
                        "bbox_units": f.bbox_units,
                        "confidence": f.confidence,
                        "source": f.source,
                    }
                )

    submission = {
        "household_id": household_id,
        "annualized_income": result.annualized_income,
        "comparison": result.comparison,
        "readiness_status": result.readiness_status,
        "citations": citations,
    }

    images_by_doc_id = {d.document_id: d.page_image_data_uri for d in ingested_docs}

    profile = {
        "household_id": household_id,
        "household_size": household_size,
        "annualized_income": result.annualized_income,
        "threshold": result.threshold,
        "comparison": result.comparison,
        "readiness_status": result.readiness_status,
        "reasons": result.reasons,
        "notes": result.notes,
        "income_components": [asdict(c) for c in result.income_components],
        "documents": [
            {
                "document_id": e.document_id,
                "document_type": e.document_type,
                "page_size_points": list(e.page_size_points),
                "degraded": e.degraded,
                "degraded_reason": e.degraded_reason,
                "page_image_data_uri": images_by_doc_id.get(e.document_id),
                "fields": [
                    {
                        "field": f.field,
                        "value": f.value,
                        "page": f.page,
                        "bbox": f.bbox,
                        "bbox_units": f.bbox_units,
                        "confidence": f.confidence,
                        "source": f.source,
                    }
                    for f in e.fields
                ],
                "abstained": e.abstained,
            }
            for e in extractions
        ],
        "quarantine_events": quarantine_events,
        "degraded_docs": degraded_docs,
        "citations": citations,
    }
    return {"submission": submission, "profile": profile}


def validate_submission(submission: dict) -> list:
    """Validate against submission.schema.json. Returns a list of error
    strings (empty means valid). Falls back to a minimal manual check if
    the optional `jsonschema` package isn't installed."""
    schema = json.loads(SUBMISSION_SCHEMA_PATH.read_text(encoding="utf-8"))
    if _HAVE_JSONSCHEMA:
        validator = jsonschema.Draft202012Validator(schema)
        return [str(e.message) for e in validator.iter_errors(submission)]
    errors = []
    for req in schema.get("required", []):
        if req not in submission:
            errors.append(f"missing required field: {req}")
    if submission.get("comparison") not in {"below_or_equal", "above", "no_frozen_threshold"}:
        errors.append("invalid comparison enum")
    if submission.get("readiness_status") not in {"READY_TO_REVIEW", "NEEDS_REVIEW"}:
        errors.append("invalid readiness_status enum")
    return errors


def run_all(write_files: bool = True, render_images: bool = False) -> dict:
    reset_quarantine_log()
    manifest_rows = load_manifest()
    checklists = _load_checklists()
    results = {}
    for row in checklists:
        household_id = row["household_id"]
        household_size = row["household_size"]
        built = build_household(household_id, household_size, manifest_rows, render_images=render_images)
        errors = validate_submission(built["submission"])
        built["schema_errors"] = errors
        results[household_id] = built
        if write_files:
            (SUBMISSIONS_DIR / f"{household_id}.submission.json").write_text(
                json.dumps(built["submission"], indent=2), encoding="utf-8"
            )
            (PROFILES_DIR / f"{household_id}.profile.json").write_text(
                json.dumps(built["profile"], indent=2), encoding="utf-8"
            )
    if write_files:
        all_profiles = {hid: r["profile"] for hid, r in results.items()}
        (PROFILES_DIR / "all_profiles.json").write_text(json.dumps(all_profiles, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    out = run_all()
    for hid, r in out.items():
        s = r["submission"]
        print(f"{hid}: annualized={s['annualized_income']} comparison={s['comparison']} "
              f"readiness={s['readiness_status']} reasons={r['profile']['reasons']} "
              f"schema_errors={r['schema_errors']}")
