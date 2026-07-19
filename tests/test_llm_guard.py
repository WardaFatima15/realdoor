"""Bilingual (English + Spanish) safety-guard tests for src/llm.py.

Stdlib-only: no network call, no OPENAI_API_KEY required. Covers three
things:

1. FORBIDDEN_RE (the post-generation guard) catches a representative set
   of Spanish decision/ranking/guarantee/prediction phrases, still catches
   the pre-existing English set, and does not false-positive on safe
   Spanish or safe English text.
2. ask()'s internal pre-model decision-boundary guard refuses a Spanish
   decision-seeking question ("¿soy elegible?") deterministically, with no
   API key present and no model ever consulted.
3. The hand-written Spanish deterministic fallbacks (explain/coach/ask)
   contain zero forbidden words in either language.
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

# Make sure no stray OPENAI_API_KEY from the environment leaks into these
# tests -- every assertion here must hold with no key present at all.
os.environ.pop("OPENAI_API_KEY", None)

from src import llm  # noqa: E402


class BilingualForbiddenRegexTests(unittest.TestCase):
    """FORBIDDEN_RE must catch English AND Spanish decision/ranking/
    guarantee/prediction vocabulary, and must not false-positive on safe
    text in either language."""

    SPANISH_DECISION_PHRASES = [
        "usted es elegible para este programa",
        "su solicitud fue aprobada",
        "este caso será denegado",
        "califica para el programa",
        "le garantizamos prioridad en la lista",
        "aquí está la predicción de aprobación",
        # unaccented misspellings, as models sometimes drop accents
        "le damos garantia de aprobacion",
        "esta es la prediccion de aprobacion",
        "usted no es elegible",
        "el hogar fue rechazado",
        "vamos a clasificar su expediente",
        "este hogar tiene prioridad",
        "el resultado es prioritario",
        "el sistema pronostica su aprobación",
    ]

    ENGLISH_DECISION_PHRASES = [
        "you are eligible for this program",
        "your application was approved",
        "this case will be denied",
        "you qualify for the program",
        "we guarantee you priority placement",
        "here is our prediction of approval",
        "you are ineligible",
        "this household is ranked first",
        "this file is prioritized",
    ]

    SAFE_SPANISH_PHRASES = [
        "su ingreso anualizado es",
        "estado de preparación",
        "revisión humana",
        "documentos",
        "comparación con el umbral",
    ]

    SAFE_ENGLISH_PHRASES = [
        "readiness status",
        "comparison",
        "human reviewer",
        "the annualized income is",
        "a document is on file",
    ]

    def test_catches_spanish_decision_phrases(self):
        self.assertGreaterEqual(len(self.SPANISH_DECISION_PHRASES), 12)
        for phrase in self.SPANISH_DECISION_PHRASES:
            with self.subTest(phrase=phrase):
                self.assertIsNotNone(
                    llm.FORBIDDEN_RE.search(phrase),
                    f"expected FORBIDDEN_RE to match Spanish phrase: {phrase!r}",
                )

    def test_still_catches_english_decision_phrases(self):
        for phrase in self.ENGLISH_DECISION_PHRASES:
            with self.subTest(phrase=phrase):
                self.assertIsNotNone(
                    llm.FORBIDDEN_RE.search(phrase),
                    f"expected FORBIDDEN_RE to match English phrase: {phrase!r}",
                )

    def test_no_false_positive_on_safe_spanish(self):
        for phrase in self.SAFE_SPANISH_PHRASES:
            with self.subTest(phrase=phrase):
                self.assertIsNone(
                    llm.FORBIDDEN_RE.search(phrase),
                    f"did not expect FORBIDDEN_RE to match safe Spanish phrase: {phrase!r}",
                )

    def test_no_false_positive_on_safe_english(self):
        for phrase in self.SAFE_ENGLISH_PHRASES:
            with self.subTest(phrase=phrase):
                self.assertIsNone(
                    llm.FORBIDDEN_RE.search(phrase),
                    f"did not expect FORBIDDEN_RE to match safe English phrase: {phrase!r}",
                )

    def test_guard_helper_agrees_with_forbidden_re(self):
        # _guard() is the function every public entry point actually calls.
        for phrase in self.SPANISH_DECISION_PHRASES + self.ENGLISH_DECISION_PHRASES:
            with self.subTest(phrase=phrase):
                self.assertFalse(llm._guard(phrase))
        for phrase in self.SAFE_SPANISH_PHRASES + self.SAFE_ENGLISH_PHRASES:
            with self.subTest(phrase=phrase):
                self.assertTrue(llm._guard(phrase))


class SpanishDecisionBoundaryPreGuardTests(unittest.TestCase):
    """ask()'s internal pre-model guard must refuse a Spanish
    decision-seeking question deterministically -- no API key, no model
    call, exactly like the English "am I eligible?" case."""

    PROFILE = {
        "household_id": "HH-001",
        "household_size": 1,
        "annualized_income": 56316.0,
        "threshold": 72000.0,
        "comparison": "below_or_equal",
        "readiness_status": "READY_TO_REVIEW",
        "reasons": [],
        "notes": [],
    }

    def setUp(self):
        # Belt-and-suspenders: confirm no key is set for this test process
        # (also stripped at import time above).
        self.assertNotIn("OPENAI_API_KEY", os.environ)

    def test_spanish_am_i_eligible_refused_deterministically(self):
        text, source = llm.ask("¿soy elegible?", self.PROFILE, [])
        self.assertEqual(source, "deterministic_fallback")
        self.assertIsNone(llm.FORBIDDEN_RE.search(text))

    def test_english_am_i_eligible_still_refused_deterministically(self):
        text, source = llm.ask("am I eligible?", self.PROFILE, [])
        self.assertEqual(source, "deterministic_fallback")

    def test_decision_boundary_regex_catches_required_spanish_keywords(self):
        keywords = [
            "soy elegible",
            "elegible",
            "elegibilidad",
            "aprobar",
            "aprobado",
            "me aprueban",
            "deniegan",
            "denegado",
            "rechazan",
            "califico",
            "me toca",
            "garantizan",
        ]
        for kw in keywords:
            with self.subTest(keyword=kw):
                self.assertIsNotNone(
                    llm._DECISION_BOUNDARY_RE.search(kw),
                    f"expected the Spanish decision-boundary pre-guard to match: {kw!r}",
                )

    def test_ask_with_spanish_keyword_never_reaches_model_path(self):
        # With no API key present, ask() can never actually reach OpenAI
        # regardless -- but the pre-guard must fire *before* even trying,
        # returning the exact same deterministic_fallback source tag used
        # by every other refusal branch (not "openai").
        for question in ["soy elegible", "me van a aprobar?", "por qué me deniegan"]:
            with self.subTest(question=question):
                text, source = llm.ask(question, self.PROFILE, [])
                self.assertEqual(source, "deterministic_fallback")


class SpanishDeterministicFallbackTests(unittest.TestCase):
    """The hand-written Spanish deterministic fallbacks must actually be
    reachable with language="es" and no API key, and must contain zero
    forbidden words in either language."""

    PROFILE_READY = {
        "household_id": "HH-001",
        "household_size": 1,
        "annualized_income": 56316.0,
        "threshold": 72000.0,
        "comparison": "below_or_equal",
        "readiness_status": "READY_TO_REVIEW",
        "reasons": [],
        "notes": [],
    }

    PROFILE_NEEDS_REVIEW = {
        "household_id": "HH-002",
        "household_size": 2,
        "annualized_income": 30000.0,
        "threshold": 40000.0,
        "comparison": "below_or_equal",
        "readiness_status": "NEEDS_REVIEW",
        "reasons": ["PAY_STUB_MISSING", "GIG_INCOME_UNCORROBORATED"],
        "notes": [],
    }

    def test_explain_es_fallback_has_no_key_and_no_forbidden_words(self):
        text, source = llm.explain(self.PROFILE_READY, language="es")
        self.assertEqual(source, "deterministic_fallback")
        self.assertIsNone(llm.FORBIDDEN_RE.search(text))
        # sanity: the numbers from CONTEXT appear verbatim, untranslated
        self.assertIn("56,316.00", text)
        self.assertIn("72,000", text)

    def test_coach_es_fallback_has_no_key_and_no_forbidden_words(self):
        text, source = llm.coach(self.PROFILE_NEEDS_REVIEW, language="es")
        self.assertEqual(source, "deterministic_fallback")
        self.assertIsNone(llm.FORBIDDEN_RE.search(text))

    def test_coach_es_fallback_ready_status_has_no_forbidden_words(self):
        text, source = llm.coach(self.PROFILE_READY, language="es")
        self.assertEqual(source, "deterministic_fallback")
        self.assertIsNone(llm.FORBIDDEN_RE.search(text))

    def test_ask_es_generic_fallback_has_no_forbidden_words(self):
        text, source = llm.ask("¿algo que no está en el expediente?", self.PROFILE_READY, [], language="es")
        self.assertEqual(source, "deterministic_fallback")
        self.assertIsNone(llm.FORBIDDEN_RE.search(text))

    def test_unknown_language_coerces_to_english(self):
        text_default, source_default = llm.explain(self.PROFILE_READY)
        text_bad, source_bad = llm.explain(self.PROFILE_READY, language="fr")
        self.assertEqual(source_default, "deterministic_fallback")
        self.assertEqual(source_bad, "deterministic_fallback")
        self.assertEqual(text_default, text_bad)

    def test_english_default_fallback_unchanged_byte_identical(self):
        # language defaults to "en"; explicit "en" must match the default.
        text_a, _ = llm.explain(self.PROFILE_READY)
        text_b, _ = llm.explain(self.PROFILE_READY, language="en")
        self.assertEqual(text_a, text_b)


if __name__ == "__main__":
    unittest.main()
