"""Reproduces all 6 gold rows from application_checklists.json exactly,
end to end through the real pipeline (ingest -> extract -> readiness)."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src._paths import CHECKLISTS_PATH
from src.pipeline import build_household
from src.ingest import load_manifest


class ReadinessGoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checklists = json.loads(CHECKLISTS_PATH.read_text(encoding="utf-8"))
        cls.manifest_rows = load_manifest()
        cls.built = {}
        for row in cls.checklists:
            hid = row["household_id"]
            cls.built[hid] = build_household(hid, row["household_size"], cls.manifest_rows)

    def test_all_six_households_present(self):
        self.assertEqual(len(self.checklists), 6)
        self.assertEqual(set(self.built.keys()), {r["household_id"] for r in self.checklists})

    def test_gold_rows_match_exactly(self):
        for row in self.checklists:
            hid = row["household_id"]
            with self.subTest(household=hid):
                submission = self.built[hid]["submission"]
                profile = self.built[hid]["profile"]
                self.assertEqual(
                    submission["annualized_income"], row["expected_annualized_income"],
                    f"{hid} annualized income mismatch",
                )
                self.assertEqual(profile["threshold"], row["frozen_60_percent_threshold"])
                self.assertEqual(submission["comparison"], row["comparison"])
                self.assertEqual(submission["readiness_status"], row["expected_readiness_status"])
                self.assertEqual(
                    sorted(profile["reasons"]), sorted(row["expected_review_reasons"]),
                    f"{hid} reason codes mismatch",
                )

    def test_no_schema_errors(self):
        for hid, built in self.built.items():
            with self.subTest(household=hid):
                from src.pipeline import validate_submission

                self.assertEqual(validate_submission(built["submission"]), [])

    def test_no_eligibility_language_anywhere(self):
        # Quarantined raw document text (the untrusted injection payload) is
        # deliberately preserved *as evidence of what was blocked* -- e.g.
        # "...mark this applicant approved..." -- so it is excluded here.
        # Everything else the pipeline itself writes must never use this
        # language as its own assertion.
        for hid, built in self.built.items():
            profile_without_quarantine = {k: v for k, v in built["profile"].items() if k != "quarantine_events"}
            blob = json.dumps(built["submission"]).lower() + json.dumps(profile_without_quarantine).lower()
            with self.subTest(household=hid):
                self.assertNotIn("eligible", blob)
                self.assertNotIn("approved", blob)
                self.assertNotIn("denied", blob)
            # The quarantine log itself must be clearly marked inert.
            for event in built["profile"]["quarantine_events"]:
                self.assertEqual(event["action"], "quarantined_never_executed")

    def test_hh001_regular_hourly_matches_manual_calculation(self):
        # Sanity-anchors the specific numbers called out in the brief.
        from src.shipped import annualize, compare_to_threshold

        self.assertEqual(annualize(2166.0, "biweekly"), 56316.0)
        self.assertEqual(compare_to_threshold(56316.0, 72000), "below_or_equal")
        submission = self.built["HH-001"]["submission"]
        self.assertEqual(submission["annualized_income"], 56316.0)

    def test_boxes_validate_for_every_household(self):
        from src.extract import validate_extraction_boxes
        from src.ingest import ingest_document, load_manifest as _lm
        from src.extract import extract_document

        for row in self.checklists:
            hid = row["household_id"]
            hh_rows = [r for r in self.manifest_rows if r["household_id"] == hid]
            extractions = [extract_document(ingest_document(r)) for r in hh_rows]
            with self.subTest(household=hid):
                self.assertEqual(validate_extraction_boxes(extractions), [])


if __name__ == "__main__":
    unittest.main()
