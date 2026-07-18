PYTHON ?= python

.PHONY: install run adversarial delete-session test starter-test all serve

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -r requirements-extract.txt || true

# Runs the full pipeline (ingest -> extract -> readiness) for all 6
# households and writes output/submissions/*.json + output/profiles/*.json.
run:
	$(PYTHON) -m src.pipeline

# Root test suite: readiness gold rows, submission schema validation, and
# the real 24-case adversarial suite.
test:
	$(PYTHON) -m unittest discover -s tests -v

# Shipped starter tests (calculate.py, pack integrity) -- unmodified.
starter-test:
	cd realdoor-hackathon-starter-pack/starter && $(PYTHON) -m unittest discover -s tests -v

# Runs just the real adversarial_tests.jsonl suite (24/24 expected green).
adversarial:
	$(PYTHON) -m unittest tests.test_adversarial -v

# Demonstrates session delete() -> zeroize -> verify-empty read.
delete-session:
	$(PYTHON) scripts/delete_session_demo.py

# Optional AI layer: runs the tiny local Flask server (server.py) on
# http://localhost:8000, serving the static app plus /api/explain,
# /api/coach, /api/ask, /api/redteam, /api/health. Requires
# `pip install -r requirements-llm.txt` once. Works with no
# OPENAI_API_KEY set (every AI feature falls back to genuine, templated
# deterministic text); set OPENAI_API_KEY (and optionally OPENAI_MODEL,
# default gpt-4o-mini) beforehand for live AI-worded explanations.
serve:
	$(PYTHON) server.py

all: starter-test test adversarial
