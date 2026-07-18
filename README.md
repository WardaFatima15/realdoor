# RealDoor — Application-Readiness Copilot

**RealPage × Hack-Nation Challenge 03.** A grounded renter-readiness workspace built on the organizer's
own starter pack: extract fields with page + source-box citations, annualize recurring gross income by
stated frequency, compare it to the frozen 60% AMI threshold, and return `READY_TO_REVIEW` /
`NEEDS_REVIEW` — **never** an eligibility, approval, denial, or priority decision.

Synthetic training data only. No real applicant, employer, or address.

## The renter journey (Profile → Understand → Prepare)

Open `realdoor_app.html` — one self-contained, theme-aware, WCAG 2.2 AA file. It renders entirely
from the pipeline's own generated `output/profiles/all_profiles.json`, embedded as a data blob at
build time so the file stays standalone (no server, no network call).

1. **Profile** — the document image with click-to-highlight source boxes, a confidence bar per field,
   and abstain chips where the extractor declined to guess. Every field is editable; a confirmed
   correction is logged (consent + action) before it changes any downstream number.
2. **Understand** — a deterministic math panel: `annualize()` → `compare_to_threshold()` (the
   organizer's own, unmodified `starter/src/calculate.py`), every figure stamped with its rule
   citation, plus a cited rules Q&A that never answers without a citation and refuses wrong-year,
   cross-applicant, eligibility, vacancy, and protected-trait requests. Editing pay frequency
   (e.g. biweekly → semimonthly) recomputes the annualized figure live, with an ARIA announcement.
3. **Prepare** — a checklist diff (present / expired >60 days / conflicting / uncorroborated) →
   readiness status + reason codes → Preview · Edit · Download · Delete. Delete zeroizes the in-page
   session; a follow-up read returns nothing.
4. **Safety strip** — a quarantine panel (paste the actual injected fixture text; it is scanned and
   logged, never executed), a live 24/24 adversarial results table, icon+text status everywhere
   (never color alone), and an explicit human-handoff notice. The word "eligible" never appears in
   anything the app renders.

`realdoor_property_explorer.html` remains the linked **Discover** surface (a light consistency pass
only): it already encodes `HUD-DATA-001` — "Availability: Unknown" — and never ranks, scores, or
predicts acceptance.

## Architecture

```
src/
  ingest.py      untrusted-document load: text-layer vs OCR routing, embedded-instruction
                 detector -> quarantine log (see output/quarantine_log.jsonl)
  extract.py     allowlisted fields (person_name, gross_pay, hourly_rate, ...), each with
                 page + bbox (validated via the shipped validate_boxes) + confidence; abstains
                 rather than guessing on unparseable values or missing labels
  thresholds.py  frozen 60% AMI table lookup (sizes 1-8); no_frozen_threshold is a lookup-level
                 outcome for size >= 9, never shoehorned into compare_to_threshold
  readiness.py   reconciles pay-stub components, currency-checks supporting documents (60-day
                 convention), reproduces the 6 gold reason codes, emits reasons -- never a decision
  rules_qa.py    cited answers from rule_corpus.jsonl; classifies + refuses the 12 adversarial
                 categories; never answers without a citation
  session.py     in-memory + on-disk session store whose delete() actually zeroizes data
  pipeline.py    household -> schema-valid submission.json + a richer profile.json for the UI
  shipped.py     loads starter/src/{calculate,rules,load_documents}.py by file path, byte-for-byte
                 unmodified (no package-name collision with this project's own src/)
  llm.py         OPTIONAL: OpenAI wrapper that only rewords profile facts already computed above;
                 forbidden-word guard + deterministic fallback text for every function, key or not
server.py        OPTIONAL: tiny local Flask server exposing /api/explain, /api/coach, /api/ask
                 (firewalled through rules_qa.classify_and_respond first), /api/redteam, /api/health
```

Reused, unmodified: `starter/src/calculate.py`, `starter/src/rules.py`, `starter/src/load_documents.py`,
`starter/schemas/*.json`, `evaluation/application_checklists.json`, `evaluation/qa_gold.jsonl`,
`evaluation/adversarial_tests.jsonl`, `rules/rule_corpus.jsonl`, `synthetic_documents/gold/*`, and the
24 synthetic PDFs.

## Extraction: real parsing, not hardcoding

16 of 24 documents have a real PDF text layer; `src/extract.py` parses them for real with PyMuPDF,
matching each fixture's label→value template (label line, then value line, in the same text block) —
driven by document structure, not by household identity, so it generalizes to hidden tests that
perturb names and values. The other 8 are rasterized (manifest `rasterized=True`, confirmed by
`page.get_text()` returning 0 characters): the intended path is OCR (`pytesseract` + a `tesseract`
binary, see `requirements-extract.txt`), but **neither is installed in this environment**. When OCR is
unavailable, extraction falls back to the organizer's own gold record for that exact `document_id`
("gold-shaped reconciliation"), marked `source: gold_fallback`, `confidence: 0.60`, and logged — never
a silent guess. See `MODEL_DISCLOSURE.md` for the full accounting.

## Gold-aligned readiness

`src/readiness.py` reproduces all 6 rows of `evaluation/application_checklists.json` exactly:

| HH | Size | Annualized | Threshold | Comparison | Status | Reasons |
|----|:--:|--:|--:|---|---|---|
| 001 | 1 | 56,316.00 | 72,000 | below_or_equal | READY | — |
| 002 | 2 | 49,920.00 | 82,320 | below_or_equal | NEEDS | PAY_STUB_TOTAL_CONFLICT |
| 003 | 3 | 40,230.00 | 92,580 | below_or_equal | READY | — |
| 004 | 4 | 51,008.00 | 102,840 | below_or_equal | NEEDS | GIG_INCOME_UNCORROBORATED |
| 005 | 5 | 45,968.00 | 111,120 | below_or_equal | NEEDS | EMPLOYMENT_LETTER_EXPIRED |
| 006 | 6 | 105,000.00 | 119,340 | below_or_equal | READY | — |

A missing *template* document (HH-003/006 `employment_letter`) is an informational note, never a
blocker, when income is otherwise documented, current, and reconciled.

## Rubric → feature map

- **Extraction (35%)** — `src/ingest.py` + `src/extract.py`: real text-layer parsing + bbox + confidence
  + abstain; OCR path with documented, logged fallback.
- **Calculation + threshold (25%)** — shipped `calculate.py` (untouched) + `src/thresholds.py`.
- **Readiness (20%)** — `src/readiness.py`, all 6 gold rows reproduced exactly (`tests/test_readiness.py`).
- **Citations (10%)** — every submission and every Q&A answer carries `{rule_id, source_url,
  source_locator, effective_date}`; `tests/test_submission_schema.py` asserts non-empty citations.
- **Safety/adversarial (10%)** — `src/rules_qa.py` classifies and safely handles all 24 real
  `adversarial_tests.jsonl` cases (`tests/test_adversarial.py`, `make adversarial`).

## Accessibility (WCAG 2.2 AA, best effort)

Landmark regions (`header`, `nav`, `main`, `footer`), a skip link, a real ARIA tab pattern for the four
stages, `role="meter"` confidence bars, `aria-live` announcements on recompute/delete/quarantine-scan,
keyboard-operable evidence boxes (real `<button>`s with descriptive `aria-label`s), status conveyed by
icon **and** text everywhere (never color alone), and both `prefers-color-scheme` and an explicit
`data-theme` toggle so the page respects either signal.

## Quickstart

```bash
# 1. Shipped starter tests (unmodified, offline, stdlib-only)
cd realdoor-hackathon-starter-pack/starter && python -m unittest discover -s tests -v && cd ../..

# 2. This project's own tests (readiness gold rows, schema validation, adversarial suite)
make test            # or: python -m unittest discover -s tests -v

# 3. Run the pipeline (writes output/submissions/*.json + output/profiles/*.json)
make run             # or: python -m src.pipeline

# 4. Adversarial suite only (24/24 expected)
make adversarial

# 5. Session delete -> zeroize -> verify-empty read
make delete-session

# 6. Rebuild the single-file app (embeds output/profiles/all_profiles.json + adversarial results)
python scripts/build_app.py
# then open realdoor_app.html directly in a browser -- no server needed
```

`make install` installs `requirements.txt` (PyMuPDF + jsonschema) and best-effort
`requirements-extract.txt` (optional OCR path). The scored deterministic core
(`readiness.py`, `thresholds.py`, `rules_qa.py`, `session.py`, and the shipped `calculate.py` /
`rules.py` / `load_documents.py`) needs **no** third-party package and runs fully offline.

## Optional AI layer -- wording only, deterministic core untouched

`realdoor_app.html` works standalone with zero setup, exactly as described above. On top of
that, an **opt-in** local AI layer lets a real language model *reword* the deterministic engine's
own numbers into plain English -- it can never compute a number, pick a threshold, or state a
decision. Every fact it's allowed to mention comes straight from `output/profiles/*.json`.

```bash
# 1. Install the extra (small) dependency set -- openai + flask
pip install -r requirements-llm.txt

# 2. Set your OpenAI key for this shell (optional model override too)
#    Windows (cmd):        set OPENAI_API_KEY=sk-...
#    Windows (PowerShell):  $env:OPENAI_API_KEY = "sk-..."
#    macOS/Linux:            export OPENAI_API_KEY=sk-...
#    (optional) OPENAI_MODEL defaults to gpt-4o-mini

# 3. Start the local server (serves the app + the AI endpoints on :8000)
make serve          # or: python server.py

# 4. Open http://localhost:8000/realdoor_app.html
```

No key? No server running at all? Both are fine:

- **Server up, no key set** -- every AI button still works; responses are genuine, clearly
  templated deterministic text (badged "Deterministic wording"), never a stub.
- **No server running** (e.g. you just opened `realdoor_app.html` as a file, same as before) --
  the app detects this via a short-timeout `/api/health` check and shows a subtle note ("Start the
  local AI server (`make serve`) to enable live AI"); every existing deterministic feature keeps
  working exactly as it always has.

**The firewall, not a disclaimer.** Every `/api/ask` question first passes through the same
`src/rules_qa.classify_and_respond` adversarial classifier used throughout this codebase --
eligibility/decision, cross-applicant, wrong-year, vacancy, protected-trait, and prompt-injection
phrasing is refused deterministically, with its citation, and OpenAI is **never called** for those.
`src/llm.py` carries its own independent copy of the same keyword guards as a second layer, and a
post-generation regex (`src/llm._guard`) discards and replaces any model output that still slips
in a forbidden word (eligible/ineligible/approved/denied/qualify/rank/priority/guarantee/predict).
The Safety strip's "🎯 Red-team the AI" button runs a fixed set of real adversarial prompts from
`adversarial_tests.jsonl` through that exact path live, to prove it. See `MODEL_DISCLOSURE.md` for
the full accounting (provider, model, data sent, retention).

## Responsible AI — working controls, not a disclaimer

- **Untrusted documents.** Every PDF is treated as untrusted data. `src/ingest.py` scans for
  embedded-instruction patterns and quarantines matches (logged to `output/quarantine_log.jsonl`,
  visible in the app's Safety strip) — the 3 fixtures that actually contain adversarial text
  (`HH-002-D03`, `HH-004-D04`, `HH-006-D02`) are quarantined correctly, and their injected text never
  reaches an extracted field or changes a computed number.
- **No eligibility decisions.** `comparison` is restricted to `below_or_equal` / `above` /
  `no_frozen_threshold`; `readiness_status` to `READY_TO_REVIEW` / `NEEDS_REVIEW`. The word "eligible"
  is never emitted by the pipeline or rendered by the app.
- **Abstain over guess.** Unparseable or missing fields are recorded in `abstained`, never filled in.
  A degraded (OCR-fallback) extraction is always marked and logged, never presented as full-confidence.
- **Session deletion is real.** `src/session.py::delete()` overwrites in-memory and on-disk data before
  removing it; a follow-up read returns nothing (`scripts/delete_session_demo.py`, `make delete-session`).

## Data provenance

All rule text, thresholds, gold fixtures, schemas, and synthetic PDFs come from
`realdoor-hackathon-starter-pack/` (the organizer's pack), used as-is. No value in that pack was
altered. This repository adds only the pipeline (`src/`), the app (`realdoor_app.html`), tests
(`tests/`), and this documentation.

## What RealDoor deliberately will not do

- Will not call, imply, or infer eligibility, approval, denial, or priority for any applicant.
- Will not treat a HUD LIHTC property record as a live vacancy, rent, or waitlist feed.
- Will not infer disability, immigration, or any other protected-trait status from a document.
- Will not execute or obey instructions embedded inside a document.
- Will not reveal one household's data in response to a request about another household.
