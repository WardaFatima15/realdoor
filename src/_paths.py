"""Central path resolution for the RealDoor pipeline.

Keeps every module pointed at the same on-disk locations without
hardcoding relative path guesses in more than one place.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = ROOT / "realdoor-hackathon-starter-pack"
STARTER_ROOT = PACK_ROOT / "starter"
STARTER_SRC = STARTER_ROOT / "src"
RULES_PATH = PACK_ROOT / "rules" / "rule_corpus.jsonl"
QA_GOLD_PATH = PACK_ROOT / "evaluation" / "qa_gold.jsonl"
ADVERSARIAL_PATH = PACK_ROOT / "evaluation" / "adversarial_tests.jsonl"
CHECKLISTS_PATH = PACK_ROOT / "evaluation" / "application_checklists.json"
SUBMISSION_SCHEMA_PATH = STARTER_ROOT / "schemas" / "submission.schema.json"
DOCUMENT_GOLD_SCHEMA_PATH = STARTER_ROOT / "schemas" / "document_gold.schema.json"
DOCUMENT_GOLD_PATH = PACK_ROOT / "synthetic_documents" / "gold" / "document_gold.jsonl"
FIELD_SCHEMA_PATH = PACK_ROOT / "synthetic_documents" / "gold" / "field_schema.json"
MANIFEST_PATH = PACK_ROOT / "synthetic_documents" / "gold" / "document_manifest.csv"
DOCUMENTS_DIR = PACK_ROOT / "synthetic_documents" / "documents"
MTSP_THRESHOLDS_CSV_PATH = PACK_ROOT / "data" / "mtsp_2026_boston_cambridge_quincy.csv"

OUTPUT_DIR = ROOT / "output"
PROFILES_DIR = OUTPUT_DIR / "profiles"
SUBMISSIONS_DIR = OUTPUT_DIR / "submissions"
SESSIONS_DIR = OUTPUT_DIR / "sessions"
QUARANTINE_LOG_PATH = OUTPUT_DIR / "quarantine_log.jsonl"

for d in (OUTPUT_DIR, PROFILES_DIR, SUBMISSIONS_DIR, SESSIONS_DIR):
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Read-only deployment filesystem (e.g. Vercel serverless): the
        # pre-built output/ directories are read from, never written to,
        # in that environment, so a failed mkdir here is not fatal.
        pass
