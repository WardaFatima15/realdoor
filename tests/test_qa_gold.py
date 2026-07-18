"""Validates all 36 gold Q&A pairs from the organizer's own qa_gold.jsonl
-- the 30 household-scoped questions against the real pipeline's own
computed snapshot (never a hardcoded expectation), and the 6 general
factual questions against the rule corpus. Every answer must carry
exactly the gold-listed citation rule_ids; where the pipeline computes a
deterministic value (threshold/annualized income/comparison/readiness),
the produced text must match the gold answer exactly."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src._paths import CHECKLISTS_PATH, QA_GOLD_PATH
from src.ingest import load_manifest
from src.pipeline import build_household
from src.rules_qa import answer_factual_question, answer_household_question, classify_and_respond


def _load_qa_gold():
    rows = []
    with QA_GOLD_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class QAGoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qa_gold = _load_qa_gold()
        checklists = json.loads(CHECKLISTS_PATH.read_text(encoding="utf-8"))
        manifest_rows = load_manifest()
        cls.snapshots = {}
        for row in checklists:
            hid = row["household_id"]
            built = build_household(hid, row["household_size"], manifest_rows)
            profile = built["profile"]
            cls.snapshots[hid] = {
                "household_id": hid,
                "household_size": profile["household_size"],
                "annualized_income": profile["annualized_income"],
                "threshold": profile["threshold"],
                "comparison": profile["comparison"],
                "readiness_status": profile["readiness_status"],
            }

    def test_qa_gold_has_36_rows(self):
        self.assertEqual(len(self.qa_gold), 36)

    def test_household_scoped_questions_match_exactly(self):
        household_rows = [r for r in self.qa_gold if r["household_id"]]
        self.assertEqual(len(household_rows), 30)
        for row in household_rows:
            with self.subTest(qa_id=row["qa_id"]):
                snapshot = self.snapshots[row["household_id"]]
                resp = answer_household_question(row["question"], snapshot)
                self.assertEqual(
                    resp.answer_text, row["answer"],
                    f"{row['qa_id']}: expected {row['answer']!r}, got {resp.answer_text!r}",
                )
                got_rule_ids = {c["rule_id"] for c in resp.citations}
                self.assertEqual(
                    got_rule_ids, set(row["rule_ids"]),
                    f"{row['qa_id']}: citation rule_ids {got_rule_ids} != gold {set(row['rule_ids'])}",
                )
                self.assertTrue(resp.citations, f"{row['qa_id']}: answer has no citation")

    def test_general_factual_questions_match_exactly(self):
        general_rows = [r for r in self.qa_gold if not r["household_id"]]
        self.assertEqual(len(general_rows), 6)
        for row in general_rows:
            with self.subTest(qa_id=row["qa_id"]):
                resp = answer_factual_question(row["question"])
                if resp is None:
                    # Falls through to the adversarial/general classifier
                    # (e.g. the vacancy-hallucination category already
                    # covers "does a property record prove a unit is
                    # vacant?" -- no separate factual entry needed for it).
                    resp = classify_and_respond(row["question"])
                self.assertIsNotNone(resp, f"{row['qa_id']}: no answer produced at all")
                got_rule_ids = {c["rule_id"] for c in resp.citations}
                self.assertEqual(
                    got_rule_ids, set(row["rule_ids"]),
                    f"{row['qa_id']}: citation rule_ids {got_rule_ids} != gold {set(row['rule_ids'])}",
                )
                self.assertTrue(resp.citations, f"{row['qa_id']}: answer has no citation")

    def test_no_answer_ever_missing_a_citation(self):
        for row in self.qa_gold:
            snapshot = self.snapshots.get(row["household_id"]) if row["household_id"] else None
            resp = (
                answer_household_question(row["question"], snapshot)
                if snapshot
                else (answer_factual_question(row["question"]) or classify_and_respond(row["question"]))
            )
            self.assertTrue(resp.citations, f"{row['qa_id']}: no citation")


if __name__ == "__main__":
    unittest.main()
