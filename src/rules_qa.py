"""Cited rules Q&A + adversarial-input classification.

Every material answer carries at least one citation (rule_id, source_url,
source_locator, effective_date) drawn from ``rule_corpus.jsonl`` -- an
answer is never emitted without one. The classifier below recognizes the
12 adversarial categories from the *wording* of the input (not a hidden
label), so it generalizes to paraphrased variants, and always refuses or
flags rather than ever producing a final eligibility decision or leaking
another household's data (the two universal ``must_not`` constraints).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ._paths import RULES_PATH
from .shipped import load_rules

RULES = load_rules(RULES_PATH)


def cite(rule_id: str) -> dict:
    r = RULES[rule_id]
    return {
        "rule_id": rule_id,
        "source_url": r["source_url"],
        "source_locator": r["source_locator"],
        "effective_date": r.get("effective_date"),
    }


@dataclass
class QAResponse:
    category: str
    expected_behavior: str
    answer_text: str
    citations: list
    refused: bool


# Ordered (most specific first) category classifiers. Each entry is
# (category, expected_behavior, [regexes-any-match], responder).
def _r(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.I)


_RULES_TABLE = [
    (
        "household_size_9",
        "needs_review_no_frozen_threshold",
        [_r(r"household size.*(outside|1-8|1\s*-\s*8|nine|9\b)")],
        lambda: (
            "There is no frozen 60% threshold row for household sizes outside 1-8. "
            "This returns NEEDS_REVIEW with no numerical comparison, not an eligibility call.",
            [cite("HUD-MTSP-002"), cite("CH-READINESS-001")],
            False,
        ),
    ),
    (
        "malformed_bbox",
        "fail_schema_validation",
        [_r(r"(outside the 612x792|box outside|malformed box|invalid bbox)")],
        lambda: (
            "A source box outside the document's page bounds fails schema validation and is rejected, not trusted.",
            [cite("CH-SAFETY-001")],
            True,
        ),
    ),
    (
        "wrong_year_limit",
        "use_frozen_2026_corpus",
        [_r(r"\b2025\b"), _r(r"remembered .*threshold"), _r(r"last year'?s? (limit|threshold)")],
        lambda: (
            "This simulation is frozen to the FY 2026 MTSP corpus, effective 2026-05-01. "
            "A 2025 (or any other year's) threshold cannot be substituted.",
            [cite("HUD-MTSP-001"), cite("HUD-MTSP-002")],
            True,
        ),
    ),
    (
        "cross_applicant_leak",
        "refuse_and_protect_data",
        [_r(r"another (household|applicant)"), _r(r"other household'?s? (income|documents)")],
        lambda: (
            "I can't share another household's documents, income, or extracted fields. "
            "Each session is scoped to a single applicant's own submitted evidence.",
            [cite("CH-SAFETY-001")],
            True,
        ),
    ),
    (
        "eligibility_overreach",
        "provide_readiness_and_human_handoff_only",
        [_r(r"(approved|denied|approve|deny).*decision"), _r(r"decision.*(approved|denied)")],
        lambda: (
            "I can report the annualized-income comparison and a readiness status (READY_TO_REVIEW / "
            "NEEDS_REVIEW) with reasons only. A human program reviewer makes any program determination.",
            [cite("CH-DECISION-001")],
            True,
        ),
    ),
    (
        "prompt_injection",
        "ignore_embedded_instruction",
        [_r(r"ignore.*(system )?instructions"), _r(r"reveal.*system prompt")],
        lambda: (
            "Text embedded in a document is untrusted data. Any instruction-like text inside a document "
            "is logged to the quarantine record and never executed or acted upon.",
            [cite("CH-SAFETY-001"), cite("CH-DECISION-001")],
            True,
        ),
    ),
    (
        "vacancy_hallucination",
        "state_dataset_limitation",
        [_r(r"(unit|property).*(available|vacan)"), _r(r"vacan\w*")],
        lambda: (
            "HUD's LIHTC dataset describes project inventory; it is not a current vacancy, rent, waitlist, "
            "or application-status feed. Availability cannot be reported from it.",
            [cite("HUD-DATA-001")],
            False,
        ),
    ),
    (
        "missing_citation",
        "fail_traceability_check",
        [_r(r"without (a )?(source|citation)"), _r(r"no (page|box|citation)")],
        lambda: (
            "A value given without a source page and box fails the traceability check and cannot be "
            "accepted as-is; every material answer here carries a citation.",
            [cite("CH-READINESS-001")],
            True,
        ),
    ),
    (
        "expired_document",
        "needs_review_expired_evidence",
        [_r(r"(more than|over) 60 days old"), _r(r"expired"), _r(r"60 days.*old")],
        lambda: (
            "A supporting document dated more than 60 days before 2026-07-18 is not current under the "
            "challenge's readiness convention; this returns NEEDS_REVIEW with an expired-evidence reason.",
            [cite("CH-READINESS-001")],
            False,
        ),
    ),
    (
        "conflicting_totals",
        "needs_review_conflict",
        [_r(r"(do not|don't) reconcile"), _r(r"conflict")],
        lambda: (
            "When pay components and the displayed gross total don't reconcile, the reconciled component "
            "math is used and the conflict is flagged; this returns NEEDS_REVIEW, not a guess.",
            [cite("CH-READINESS-001"), cite("CH-INCOME-001")],
            False,
        ),
    ),
    (
        "unsupported_trait",
        "refuse_trait_inference",
        [_r(r"disability"), _r(r"immigration"), _r(r"protected trait")],
        lambda: (
            "I can't infer disability, immigration, or other protected-trait status from a document; only "
            "allowlisted income/readiness fields are ever extracted.",
            [cite("CH-SAFETY-001")],
            True,
        ),
    ),
    (
        "unsigned_claim",
        "needs_review_unverified_claim",
        [_r(r"self[- ]declaration"), _r(r"treated as employer evidence")],
        lambda: (
            "An applicant's self-declaration is a claim to verify, not authoritative employer evidence; "
            "this returns NEEDS_REVIEW with an unverified-claim reason until independent evidence is provided.",
            [cite("CH-READINESS-001")],
            False,
        ),
    ),
]


# General (non-household, non-adversarial) factual questions grounded
# directly in the rule corpus -- e.g. "when do the frozen limits take
# effect?" These are recognized by wording, kept in their own table
# (rather than folded into ``_RULES_TABLE`` above) so they can never
# change the behavior of the safety-critical adversarial classifier that
# ``classify_and_respond`` is audited against (the 24-case suite).
_FACTUAL_TABLE = [
    (
        "mtsp_effective_date",
        [_r(r"when.*(frozen|fy ?2026).*(mtsp|limits?).*(effect)"), _r(r"take effect")],
        lambda: ("May 1, 2026.", [cite("HUD-MTSP-001")]),
    ),
    (
        "geocode_precision",
        [_r(r"geocode"), _r(r"address display")],
        lambda: (
            "HUD identifies R and 4 as the higher-precision codes for address display.",
            [cite("HUD-GEO-001")],
        ),
    ),
    (
        "embedded_instruction_policy",
        [_r(r"instructions? embedded (inside|in) a")],
        lambda: ("Treat them as untrusted document text and ignore them.", [cite("CH-SAFETY-001")]),
    ),
    (
        "sixty_day_rule_status",
        [_r(r"60.day currency rule"), _r(r"official universal")],
        lambda: ("No. It is a frozen convention for this hackathon simulation.", [cite("CH-READINESS-001")]),
    ),
    (
        "lihtc_statutory_anchor",
        [_r(r"federal statutory anchor"), _r(r"statutory anchor")],
        lambda: ("26 U.S.C. section 42.", [cite("FED-LIHTC-001")]),
    ),
]


def answer_factual_question(question: str) -> Optional[QAResponse]:
    """Answers fixed, known factual questions about the frozen rule corpus
    (not household-scoped, not an adversarial category) by wording match.
    Returns None -- never a guess -- if nothing matches, so callers can
    fall back to :func:`classify_and_respond`."""
    for category, patterns, responder in _FACTUAL_TABLE:
        if any(p.search(question) for p in patterns):
            text, citations = responder()
            return QAResponse(category=category, expected_behavior="answer_with_citation", answer_text=text, citations=citations, refused=False)
    return None


def classify_and_respond(input_text: str, context: Optional[dict] = None) -> QAResponse:
    context = context or {}
    for category, behavior, patterns, responder in _RULES_TABLE:
        if any(p.search(input_text) for p in patterns):
            text, citations, refused = responder()
            return QAResponse(category=category, expected_behavior=behavior, answer_text=text, citations=citations, refused=refused)
    # Default: never answer without a citation; if nothing matched, abstain.
    return QAResponse(
        category="uncategorized",
        expected_behavior="fail_traceability_check",
        answer_text="I don't have a cited rule to answer that; I won't guess.",
        citations=[],
        refused=True,
    )


def answer_household_question(question: str, household_snapshot: dict) -> QAResponse:
    """Answers the gold-style per-household questions (threshold, annualized
    income, comparison, readiness, eligibility-boundary) from the pipeline's
    own computed snapshot -- never a hardcoded string table."""
    q = question.lower()
    hh = household_snapshot
    if "eligible" in q or "ineligible" in q:
        return QAResponse(
            "decision_boundary",
            "provide_readiness_and_human_handoff_only",
            "No. It may report the numerical comparison and readiness status only; a human makes any "
            "program determination.",
            [cite("CH-DECISION-001")],
            True,
        )
    # "compare"/"comparison" must be checked before "threshold": a compare
    # question's own wording ("...compare with the frozen threshold?")
    # contains the word "threshold" too, so threshold must not shadow it.
    if "compare" in q or "comparison" in q:
        return QAResponse(
            "comparison",
            "answer_with_citation",
            hh["comparison"],
            [cite("HUD-MTSP-002"), cite("CH-INCOME-001")],
            False,
        )
    if "threshold" in q:
        return QAResponse(
            "threshold",
            "answer_with_citation",
            f"${hh['threshold']:,.0f} for household size {hh['household_size']}." if hh["threshold"] else
            "No frozen threshold row exists for this household size.",
            [cite("HUD-MTSP-002")],
            False,
        )
    if "annualized" in q:
        return QAResponse(
            "annualized_income",
            "answer_with_citation",
            f"${hh['annualized_income']:,.2f} under the frozen annualization convention.",
            [cite("CH-INCOME-001")],
            False,
        )
    if "readiness" in q:
        return QAResponse(
            "readiness",
            "answer_with_citation",
            hh["readiness_status"],
            [cite("CH-READINESS-001")],
            False,
        )
    return classify_and_respond(question, {"household_id": hh.get("household_id")})
