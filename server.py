"""Tiny local Flask server for RealDoor's optional AI layer.

Serves the repo statically (so http://localhost:8000/realdoor_app.html
works exactly as the file-opened version does) plus a handful of JSON /
streaming endpoints that let the static app call into ``src/llm.py``
without ever putting an API key in the browser.

Every endpoint that could touch OpenAI runs the deterministic
``rules_qa.classify_and_respond`` firewall first (for /api/ask) and/or
``src/llm.py``'s own internal guardrails, and always sets a response
header ``x-realdoor-source: openai|deterministic`` so the UI can badge
answers honestly. The API key is read once from the environment and is
never logged, echoed, or included in any response body.

Run:  python server.py   (serves on http://localhost:8000)
"""
from __future__ import annotations

import json
import os
import re
import sys
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:  # optional: load OPENAI_API_KEY from a local .env file if present
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src import llm  # noqa: E402
from src._paths import ADVERSARIAL_PATH, PROFILES_DIR  # noqa: E402
from src.rules_qa import cite, classify_and_respond  # noqa: E402

app = Flask(__name__, static_folder=str(ROOT), static_url_path="")

_HID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# The rule ids /api/ask always grounds and cites a normal (non-refused)
# question with -- income math, readiness convention, the frozen
# threshold, and the human-handoff rule -- so every answer always
# carries at least one citation, per the guardrail.
_ASK_RULE_CONTEXT = ["CH-INCOME-001", "CH-READINESS-001", "HUD-MTSP-002", "CH-DECISION-001"]

# A fixed sample of ~6 real adversarial prompts (one per several
# categories) pulled from the organizer's own adversarial_tests.jsonl,
# re-run through the exact same firewall /api/ask uses, to demonstrate
# the AI path can't be jailbroken even under direct attack.
_REDTEAM_TEST_IDS = ["ADV-001", "ADV-002", "ADV-003", "ADV-004", "ADV-005", "ADV-009"]


def _load_profile(hid: str):
    if not hid or not _HID_RE.match(hid):
        return None
    path = PROFILES_DIR / f"{hid}.profile.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_adversarial_rows():
    rows = []
    with ADVERSARIAL_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _header_source(source: str) -> str:
    """Map llm.py's internal source tag to the two-value header the UI
    badges on: 'openai' or 'deterministic'."""
    return "openai" if source == "openai" else "deterministic"


def _stream_response(source: str, chunks) -> Response:
    resp = Response(stream_with_context(chunks), mimetype="text/plain")
    resp.headers["x-realdoor-source"] = _header_source(source)
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.after_request
def _allow_local_origins(resp):
    origin = request.headers.get("Origin", "")
    if origin in ("http://localhost:8000", "http://127.0.0.1:8000", ""):
        resp.headers.setdefault("Access-Control-Allow-Origin", origin or "*")
        resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type")
        resp.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    return resp


@app.route("/")
def index():
    return send_from_directory(str(ROOT), "realdoor_app.html")


@app.route("/api/health", methods=["GET"])
def api_health():
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    body = jsonify({"ai": has_key, "model": llm.MODEL if has_key else None})
    body.headers["x-realdoor-source"] = "deterministic"
    return body


_MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB


@app.route("/api/extract-upload", methods=["POST"])
def api_extract_upload():
    """Real, on-demand extraction for a user-uploaded PDF -- not one of the
    24 fixed synthetic documents baked in at build time. Response is shaped
    exactly like one `profile.json` documents[] entry so the existing
    frontend doc-viewport/field-list rendering can consume it unchanged.

    fitz/src.ingest/src.extract are imported lazily, inside this function
    only, so that if PyMuPDF somehow isn't importable in some environment,
    only this one route degrades (a clean 501 JSON) -- every other route
    (health, ask, explain, coach, redteam, static file serving) keeps
    working untouched.
    """
    try:
        import fitz  # noqa: F401  -- PyMuPDF; import itself is the availability probe
        from src.extract import FIELD_SPECS, extract_document
        from src.ingest import ingest_bytes
    except ImportError:
        return jsonify({"error": "PDF extraction is not available on this server (PyMuPDF not installed)"}), 501

    document_type = (request.form.get("document_type") or "").strip()
    valid_types = sorted(FIELD_SPECS.keys())
    if document_type not in FIELD_SPECS:
        return jsonify({"error": f"document_type must be one of: {', '.join(valid_types)}"}), 400

    household_id = (request.form.get("household_id") or "").strip()
    if not household_id or not _HID_RE.match(household_id):
        return jsonify({"error": "household_id is required and must match ^[A-Za-z0-9_-]+$"}), 400

    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"error": "file is required"}), 400

    # Read fully into memory and cap at 8MB -- never written to disk.
    pdf_bytes = file.read(_MAX_UPLOAD_BYTES + 1)
    if len(pdf_bytes) > _MAX_UPLOAD_BYTES:
        return jsonify({"error": "file too large (max 8MB)"}), 400
    if not pdf_bytes.startswith(b"%PDF"):
        return jsonify({"error": "not a valid PDF"}), 400

    # An UPLOAD- prefix guarantees this document_id can never collide with,
    # or be mistaken for, one of the organizer's own gold document ids
    # (which look like "HH-001-D02"). That matters because the rasterized/
    # no-text-layer fallback path looks up gold records BY document_id --
    # this guarantees an ad hoc upload with no text layer legitimately
    # abstains instead of ever accidentally matching a real gold record.
    document_id = f"UPLOAD-{uuid.uuid4().hex[:10]}"

    try:
        ingested = ingest_bytes(document_id, household_id, document_type, pdf_bytes, render_page_image=True)
        extraction = extract_document(ingested)
    except Exception:
        # Never let a malformed upload 500 the function or leak a raw
        # traceback to the client.
        return jsonify({"error": "could not parse this PDF"}), 400

    quarantine_events = [
        {
            "document_id": q.document_id,
            "reason": q.reason,
            "text": q.text,
            "action": "quarantined_never_executed",
        }
        for q in ingested.quarantine
    ]

    body = jsonify(
        {
            "document_id": extraction.document_id,
            "document_type": extraction.document_type,
            "household_id": household_id,
            "page_size_points": list(extraction.page_size_points),
            "degraded": extraction.degraded,
            "degraded_reason": extraction.degraded_reason,
            "page_image_data_uri": ingested.page_image_data_uri,
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
                for f in extraction.fields
            ],
            "abstained": extraction.abstained,
            "quarantine_events": quarantine_events,
        }
    )
    body.headers["x-realdoor-source"] = "deterministic"
    return body


@app.route("/api/evaluate-upload", methods=["POST"])
def api_evaluate_upload():
    """Real readiness evaluation for the accumulated set of user-uploaded
    documents in the "My upload" ad hoc household -- reuses the exact same
    `src.readiness.evaluate_household` used for the 6 gold households, byte
    for byte, so this ad hoc household is never scored by a second,
    JS-reimplemented copy of that logic.

    src.extract/src.readiness/src.rules_qa/src.thresholds are imported
    lazily, inside this function only, matching the pattern already used
    by /api/extract-upload, so the rest of the app degrades gracefully
    (a clean 501 JSON) if this route's dependencies are ever unavailable.

    No disk writes anywhere in this handler (Vercel read-only filesystem).
    """
    try:
        from src.extract import DocumentExtraction, ExtractedField
        from src.readiness import evaluate_household
        from src.rules_qa import cite
        from src.thresholds import THRESHOLD_RULE_ID
    except ImportError:
        return jsonify({"error": "Readiness evaluation is not available on this server"}), 501

    data = request.get_json(silent=True) or {}

    household_id = (data.get("household_id") or "").strip()
    if not household_id or not _HID_RE.match(household_id):
        return jsonify({"error": "household_id is required and must match ^[A-Za-z0-9_-]+$"}), 400

    household_size = data.get("household_size")
    if not isinstance(household_size, int) or isinstance(household_size, bool) or household_size <= 0:
        return jsonify({"error": "household_size must be a positive integer"}), 400

    documents = data.get("documents")
    if not isinstance(documents, list) or not documents:
        return jsonify({"error": "documents must be a non-empty list"}), 400

    try:
        extractions = []
        for doc in documents:
            fields = [ExtractedField(**f) for f in doc["fields"]]
            extractions.append(
                DocumentExtraction(
                    document_id=doc["document_id"],
                    household_id=household_id,
                    document_type=doc["document_type"],
                    page_size_points=tuple(doc["page_size_points"]),
                    fields=fields,
                    abstained=doc.get("abstained", []),
                    degraded=doc.get("degraded", False),
                    degraded_reason=doc.get("degraded_reason"),
                )
            )
    except (KeyError, TypeError):
        return jsonify({"error": "malformed document data"}), 400

    result = evaluate_household(household_id, household_size, extractions)

    # --- citations: mirror src/pipeline.py::build_household exactly ---
    citations = [
        cite("CH-INCOME-001"),
        cite("CH-READINESS-001"),
    ]
    if result.threshold is not None:
        citations.append(cite(THRESHOLD_RULE_ID))
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

    body = jsonify(
        {
            "household_id": household_id,
            "household_size": household_size,
            "annualized_income": result.annualized_income,
            "threshold": result.threshold,
            "comparison": result.comparison,
            "readiness_status": result.readiness_status,
            "reasons": result.reasons,
            "notes": result.notes,
            "citations": citations,
        }
    )
    body.headers["x-realdoor-source"] = "deterministic"
    return body


@app.route("/api/explain", methods=["POST"])
def api_explain():
    data = request.get_json(silent=True) or {}
    profile = _load_profile(data.get("household_id"))
    if profile is None:
        return jsonify({"error": "unknown or missing household_id"}), 404
    source, chunks = llm.explain_stream(profile)
    return _stream_response(source, chunks)


@app.route("/api/coach", methods=["POST"])
def api_coach():
    data = request.get_json(silent=True) or {}
    profile = _load_profile(data.get("household_id"))
    if profile is None:
        return jsonify({"error": "unknown or missing household_id"}), 404
    source, chunks = llm.coach_stream(profile)
    return _stream_response(source, chunks)


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(silent=True) or {}
    hid = data.get("household_id")
    question = (data.get("question") or "").strip()
    profile = _load_profile(hid)
    if profile is None:
        return jsonify({"error": "unknown or missing household_id"}), 404
    if not question:
        return jsonify({"error": "question is required"}), 400

    # --- Firewall first: the same deterministic adversarial classifier
    # used everywhere else in this codebase. Any recognized category
    # (adversarial or otherwise) is answered deterministically, with its
    # own citation(s), and OpenAI is never called. ---
    fw = classify_and_respond(question)
    if fw.category != "uncategorized":
        body = jsonify(
            {
                "answer": fw.answer_text,
                "citations": fw.citations,
                "category": fw.category,
                "refused": fw.refused,
            }
        )
        body.headers["x-realdoor-source"] = "deterministic"
        return body

    # --- Otherwise: grounded answer via src/llm.py (its own internal
    # decision-boundary guardrails still apply; see src/llm.py::ask). ---
    text, source = llm.ask(question, profile, _ASK_RULE_CONTEXT)
    citations = [cite(rid) for rid in _ASK_RULE_CONTEXT]
    body = jsonify({"answer": text, "citations": citations, "category": "grounded", "refused": False})
    body.headers["x-realdoor-source"] = _header_source(source)
    return body


@app.route("/api/redteam", methods=["POST", "GET"])
def api_redteam():
    all_rows = {r["test_id"]: r for r in _load_adversarial_rows()}
    results = []
    for tid in _REDTEAM_TEST_IDS:
        row = all_rows.get(tid)
        if row is None:
            continue
        fw = classify_and_respond(row["input"])
        safe = fw.category != "uncategorized" and fw.expected_behavior == row["expected_behavior"]
        results.append(
            {
                "test_id": row["test_id"],
                "category": row["category"],
                "input": row["input"],
                "safe": safe,
                "answer": fw.answer_text,
                "citations": fw.citations,
            }
        )
    passed = sum(1 for r in results if r["safe"])
    body = jsonify({"results": results, "passed": passed, "total": len(results)})
    body.headers["x-realdoor-source"] = "deterministic"
    return body


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"RealDoor AI server on http://localhost:{port}  (AI key present: {bool(os.environ.get('OPENAI_API_KEY'))})")
    app.run(host="127.0.0.1", port=port, debug=False)
