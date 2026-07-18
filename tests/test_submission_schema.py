"""Every pipeline submission must validate against submission.schema.json,
and no output may ever express an eligibility/approval/denial/ranking
decision (only readiness + a numerical comparison)."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src._paths import CHECKLISTS_PATH, SUBMISSION_SCHEMA_PATH
from src.ingest import load_manifest
from src.pipeline import build_household, validate_submission

try:
    import jsonschema

    HAVE_JSONSCHEMA = True
except ImportError:
    HAVE_JSONSCHEMA = False


class SubmissionSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checklists = json.loads(CHECKLISTS_PATH.read_text(encoding="utf-8"))
        cls.manifest_rows = load_manifest()
        cls.schema = json.loads(SUBMISSION_SCHEMA_PATH.read_text(encoding="utf-8"))
        cls.submissions = {}
        for row in cls.checklists:
            hid = row["household_id"]
            built = build_household(hid, row["household_size"], cls.manifest_rows)
            cls.submissions[hid] = built["submission"]

    def test_required_keys_present(self):
        required = set(self.schema["required"])
        for hid, sub in self.submissions.items():
            with self.subTest(household=hid):
                self.assertTrue(required.issubset(sub.keys()))

    @unittest.skipUnless(HAVE_JSONSCHEMA, "jsonschema not installed")
    def test_validates_against_real_schema(self):
        validator = jsonschema.Draft202012Validator(self.schema)
        for hid, sub in self.submissions.items():
            with self.subTest(household=hid):
                errors = list(validator.iter_errors(sub))
                self.assertEqual(errors, [], f"{hid}: {[e.message for e in errors]}")

    def test_validate_submission_helper_agrees(self):
        for hid, sub in self.submissions.items():
            with self.subTest(household=hid):
                self.assertEqual(validate_submission(sub), [])

    def test_comparison_enum_restricted(self):
        allowed = {"below_or_equal", "above", "no_frozen_threshold"}
        for hid, sub in self.submissions.items():
            with self.subTest(household=hid):
                self.assertIn(sub["comparison"], allowed)

    def test_readiness_enum_restricted(self):
        allowed = {"READY_TO_REVIEW", "NEEDS_REVIEW"}
        for hid, sub in self.submissions.items():
            with self.subTest(household=hid):
                self.assertIn(sub["readiness_status"], allowed)

    def test_citations_present_and_nonempty(self):
        for hid, sub in self.submissions.items():
            with self.subTest(household=hid):
                self.assertIsInstance(sub["citations"], list)
                self.assertGreater(len(sub["citations"]), 0)

    def test_no_decision_language(self):
        forbidden = ["eligible", "ineligible", "approved", "denied", "prioritized", "ranked"]
        for hid, sub in self.submissions.items():
            blob = json.dumps(sub).lower()
            with self.subTest(household=hid):
                for word in forbidden:
                    self.assertNotIn(word, blob)


class MalformedBboxSchemaTests(unittest.TestCase):
    """A malformed bbox (outside the page) must fail validation -- this
    backstops the malformed_bbox adversarial category at the schema layer."""

    def test_out_of_bounds_bbox_detected(self):
        from src.shipped import validate_boxes

        rows = [
            {
                "document_id": "ADV-TEST",
                "page_size_points": [612, 792],
                "fields": [{"field": "gross_pay", "bbox": [340, 528, 900, 544]}],
            }
        ]
        errors = validate_boxes(rows)
        self.assertEqual(len(errors), 1)


if __name__ == "__main__":
    unittest.main()
