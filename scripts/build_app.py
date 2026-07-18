"""Builds the single self-contained realdoor_app.html from app_src/template.html
by running the real pipeline (with page images rendered) and the real
adversarial suite, then embedding everything as one JSON data blob.

Run: python scripts/build_app.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src._paths import ADVERSARIAL_PATH, RULES_PATH
from src.pipeline import run_all
from src.rules_qa import classify_and_respond
from src.thresholds import FROZEN_60_PCT_THRESHOLDS, THRESHOLD_EFFECTIVE_DATE, THRESHOLD_RULE_ID, THRESHOLD_SOURCE_LOCATOR
from src.shipped import FREQUENCY, load_rules

TEMPLATE_PATH = ROOT / "app_src" / "template.html"
OUTPUT_PATH = ROOT / "realdoor_app.html"


def build_adversarial_results() -> list:
    rows = []
    with ADVERSARIAL_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            resp = classify_and_respond(row["input"])
            rows.append(
                {
                    "test_id": row["test_id"],
                    "category": row["category"],
                    "variant": row["variant"],
                    "input": row["input"],
                    "expected_behavior": row["expected_behavior"],
                    "got_behavior": resp.expected_behavior,
                    "pass": resp.expected_behavior == row["expected_behavior"],
                    "answer_text": resp.answer_text,
                    "citations": resp.citations,
                    "must_not": row["must_not"],
                }
            )
    return rows


def main() -> None:
    print("Running pipeline (with page image rendering)...")
    results = run_all(write_files=True, render_images=True)
    profiles = {hid: r["profile"] for hid, r in results.items()}
    submissions = {hid: r["submission"] for hid, r in results.items()}

    print("Running the real adversarial suite...")
    adversarial_results = build_adversarial_results()
    passed = sum(1 for r in adversarial_results if r["pass"])
    print(f"Adversarial: {passed}/{len(adversarial_results)} passed")

    rules = load_rules(RULES_PATH)

    data = {
        "profiles": profiles,
        "submissions": submissions,
        "rules": rules,
        "thresholds": {
            "table": FROZEN_60_PCT_THRESHOLDS,
            "rule_id": THRESHOLD_RULE_ID,
            "effective_date": THRESHOLD_EFFECTIVE_DATE,
            "source_locator": THRESHOLD_SOURCE_LOCATOR,
        },
        "frequency": FREQUENCY,
        "adversarial_results": adversarial_results,
        "event_date": "2026-07-18",
        "currency_cutoff_days": 60,
    }

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    payload = json.dumps(data)
    if "/*__REALDOOR_DATA__*/" not in template:
        raise SystemExit("template.html is missing the /*__REALDOOR_DATA__*/ placeholder")
    rendered = template.replace("/*__REALDOOR_DATA__*/", payload)
    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUTPUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
