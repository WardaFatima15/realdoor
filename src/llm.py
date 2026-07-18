"""OpenAI-powered explanation layer -- rephrases deterministic facts only.

This module never computes a number, selects a threshold, or makes a
program determination. Every function below (``explain``, ``coach``,
``ask``) is handed a CONTEXT built entirely from a ``profile`` dict that
``src/pipeline.py`` / ``src/readiness.py`` already computed, plus (for
``ask``) a few relevant rule ids resolved from the shipped rule corpus via
``src/rules_qa.py``. The system prompt instructs the model to only reword
those facts; a post-generation regex guard (``_guard``) then discards any
output that slips in a forbidden decision/ranking word, and every public
function falls back to a genuinely-useful, deterministically templated
string in that case -- so every function *always* returns something
useful, with or without an API key, and with or without a live guard trip.

Source of truth for every number, threshold, and status stays
``src/readiness.py`` + ``src/thresholds.py``; nothing here is allowed to
override or "improve" them.
"""
from __future__ import annotations

import os
import re
from typing import Iterator, Optional, Tuple

from .rules_qa import RULES, answer_household_question, cite

# ---------------------------------------------------------------------
# Guardrail: forbidden decision / ranking / prediction words. Matches
# eligible, ineligible, approved, denied, qualify, qualifies, rank,
# priority/prioritize, guarantee, predict (and simple variants) in any
# case. If a generated string matches this, it is discarded outright.
# ---------------------------------------------------------------------
FORBIDDEN_RE = re.compile(
    r"(?i)\b(eligible|ineligible|approved|denied|qualif\w*|rank\w*|priorit\w*|guarantee\w*|predict\w*)\b"
)

# Decision-boundary / out-of-scope questions that must NEVER reach the
# model, even when phrased in a way the server-side adversarial
# classifier (rules_qa.classify_and_respond) doesn't happen to match
# (e.g. a bare "am I eligible?"). This is a second, independent
# guardrail layer inside the explanation module itself -- defense in
# depth, mirroring the same branches rules_qa.py and the static app's
# JS already refuse on.
_DECISION_BOUNDARY_RE = re.compile(r"(?i)\b(eligib\w*|ineligib\w*|approv\w*|den(y|ied)\w*|qualif\w*)\b")
_CROSS_APPLICANT_RE = re.compile(r"(?i)another (household|applicant)|other household'?s? (income|documents)")
_WRONG_YEAR_RE = re.compile(r"(?i)\b2025\b|last year'?s? (limit|threshold)|remembered .*threshold")
_VACANCY_RE = re.compile(r"(?i)(unit|property).*(available|vacan)|vacan\w*")
_PROTECTED_TRAIT_RE = re.compile(r"(?i)disability|immigration|protected trait")
_INJECTION_RE = re.compile(r"(?i)ignore.*(system )?instructions|reveal.*system prompt")

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are RealDoor's plain-English explainer for a renter-readiness workspace.

A deterministic rules engine has already computed every number, threshold, comparison, and
status you will ever see. Your ONLY job is to reword the facts given to you in CONTEXT into
warm, plain, accessible English. You are never the source of a fact.

Hard rules -- never break these:
1. Only rephrase facts present in CONTEXT below. Never state a number, date, dollar amount, or
   threshold that is not verbatim in CONTEXT. Never do arithmetic of your own.
2. Never select, assume, imply, or guess a threshold, comparison, or readiness status different
   from the one given in CONTEXT.
3. Never use these words or their variants, in any language: eligible, ineligible, approved,
   denied, qualify/qualifies, rank, priority/prioritize, guarantee, predict. This tool never makes
   or implies a program determination.
4. Always ground material claims in the citation(s) provided, and if the question asks for
   anything outside CONTEXT (another applicant's data, a different year's threshold, a live
   vacancy/rent/waitlist feed, a protected-trait inference, or a final decision), say plainly that
   a human program reviewer handles it -- do not attempt an answer.
5. Always close by noting that a human program reviewer makes any final determination.

Write 3-5 warm, clear sentences unless told otherwise. Do not use headings, bullet lists, or
markdown formatting -- plain prose only.
"""

COACH_TIPS = {
    "EMPLOYMENT_LETTER_EXPIRED": "ask the employer for a new employment letter dated within the last 60 days",
    "BENEFIT_LETTER_EXPIRED": "request an updated benefit letter dated within the last 60 days",
    "PAY_STUB_TOTAL_CONFLICT": "ask for a clean, itemized pay stub so regular hours times hourly rate matches the stated gross pay",
    "GIG_INCOME_UNCORROBORATED": "add an independent corroborating document for the gig income, such as a platform-issued earnings statement",
    "PAY_STUB_MISSING": "submit a current pay stub so wage income can be documented",
    "WAGE_INCOME_UNRESOLVED": "submit a pay stub with clear regular hours, hourly rate, and gross pay so wage income can be reconciled",
    "PAY_FREQUENCY_UNKNOWN": "confirm the pay frequency on the pay stub (weekly, biweekly, semimonthly, or monthly)",
    "EMPLOYMENT_LETTER_MISSING_NO_WAGE_EVIDENCE": "submit either a current pay stub or a current employment letter so wage income has some evidentiary basis",
    "TRACEABILITY_FAILURE": "flag this file for a human reviewer to re-check the source page and box for the affected field",
    "NO_FROZEN_THRESHOLD_FOR_HOUSEHOLD_SIZE": "nothing to submit here -- household sizes above 8 have no frozen threshold row and go straight to a human reviewer",
    "UNVERIFIED_CLAIM": "add independent evidence (a pay stub, benefit letter, etc.) beyond the self-declared application summary",
}

REASON_TEXT = {
    "PAY_STUB_TOTAL_CONFLICT": "the pay stub's stated gross pay didn't reconcile with regular hours times hourly rate, so the reconciled component math was used instead",
    "GIG_INCOME_UNCORROBORATED": "gig income is self-reported with no independent corroborating document on file",
    "EMPLOYMENT_LETTER_EXPIRED": "the employment letter is dated more than 60 days before the event date and isn't current",
    "BENEFIT_LETTER_EXPIRED": "the benefit letter is dated more than 60 days before the event date and isn't current",
    "PAY_STUB_MISSING": "no pay stub is on file, so wage income couldn't be established",
    "WAGE_INCOME_UNRESOLVED": "wage income fields couldn't be reconciled into a usable figure",
    "PAY_FREQUENCY_UNKNOWN": "the stated pay frequency isn't one of the recognized values",
    "EMPLOYMENT_LETTER_MISSING_NO_WAGE_EVIDENCE": "no employment letter and no pay stub are on file, so wage income has no evidentiary basis",
    "TRACEABILITY_FAILURE": "one or more extracted fields has a source box outside the document page and failed the traceability check",
    "NO_FROZEN_THRESHOLD_FOR_HOUSEHOLD_SIZE": "there's no frozen 60% threshold row for this household size (the table covers sizes 1-8 only)",
    "UNVERIFIED_CLAIM": "the only income evidence on file is a self-declaration, not independent evidence",
}


def _client():
    """Return a configured OpenAI client, or None if unavailable for any
    reason (no package, no key, bad key format, etc.) -- always guarded,
    never raises."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    try:
        import openai
    except ImportError:
        return None
    try:
        return openai.OpenAI(api_key=key)
    except Exception:
        return None


def _guard(text: Optional[str]) -> bool:
    """True if `text` is safe to show as-is (non-empty, no forbidden
    decision/ranking/prediction word). False means: discard it and use
    the deterministic fallback instead."""
    if not text or not text.strip():
        return False
    return FORBIDDEN_RE.search(text) is None


def _complete(user_prompt: str, max_tokens: int = 320) -> Optional[str]:
    """Single guarded call to the chat completions API. Returns the
    candidate text, or None on any failure (missing client, API error,
    empty response) -- never raises out of this module."""
    client = _client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        candidate = (resp.choices[0].message.content or "").strip()
        return candidate or None
    except Exception:
        return None


def _profile_context(profile: dict) -> str:
    threshold = profile.get("threshold")
    lines = [
        f"household_id: {profile.get('household_id')}",
        f"household_size: {profile.get('household_size')}",
        f"annualized_income: {profile.get('annualized_income')}",
        f"frozen_60pct_threshold: {threshold if threshold is not None else 'no table row for this household size'}",
        f"comparison: {profile.get('comparison')}",
        f"readiness_status: {profile.get('readiness_status')}",
        f"reasons: {', '.join(profile.get('reasons', [])) or 'none'}",
        f"notes: {'; '.join(profile.get('notes', [])) or 'none'}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------
# explain(profile)
# ---------------------------------------------------------------------
def _fallback_explain(profile: dict) -> str:
    hid = profile.get("household_id")
    size = profile.get("household_size")
    income = profile.get("annualized_income")
    threshold = profile.get("threshold")
    comparison = profile.get("comparison")
    status = profile.get("readiness_status")
    reasons = profile.get("reasons", [])

    parts = [
        f"For household {hid} (size {size}), the deterministic engine annualized this household's "
        f"documented income at ${income:,.2f}."
    ]
    if threshold is not None:
        cmp_word = "at or below" if comparison == "below_or_equal" else "above"
        parts.append(f"That figure is {cmp_word} the frozen 60% AMI threshold of ${threshold:,.0f} for this household size.")
    else:
        parts.append("There is no frozen threshold row for this household size, so no numerical comparison is made.")
    parts.append(f"The computed readiness status is {status.replace('_', ' ')}.")
    if reasons:
        reason_bits = "; ".join(REASON_TEXT.get(r, r.replace("_", " ").lower()) for r in reasons)
        parts.append(f"That reflects: {reason_bits}.")
    else:
        parts.append("No readiness reasons were flagged for this file.")
    parts.append(
        "Every figure above comes directly from the deterministic engine, never a language model; "
        "a human program reviewer makes any final determination."
    )
    return " ".join(parts)


def explain(profile: dict) -> Tuple[str, str]:
    """Warm, plain-English summary of annualized income vs. threshold,
    comparison, readiness status, and each reason code, 3-5 sentences.
    Always returns (text, source) with source in {"openai",
    "deterministic_fallback"}."""
    fallback = _fallback_explain(profile)
    context = _profile_context(profile)
    prompt = (
        f"CONTEXT:\n{context}\n\n"
        "Write a warm, plain-English 3-5 sentence summary of this household's readiness, "
        "covering the annualized income, the threshold comparison, the readiness status, and "
        "each reason listed -- using ONLY the facts in CONTEXT."
    )
    candidate = _complete(prompt)
    if candidate and _guard(candidate):
        return candidate, "openai"
    return fallback, "deterministic_fallback"


# ---------------------------------------------------------------------
# coach(profile)
# ---------------------------------------------------------------------
def _fallback_coach(profile: dict) -> str:
    if profile.get("readiness_status") != "NEEDS_REVIEW":
        return (
            "This household's readiness status is READY_TO_REVIEW, so no additional documents are "
            "called for right now. A human program reviewer still makes the final determination."
        )
    reasons = profile.get("reasons", [])
    tips = [COACH_TIPS[r] for r in reasons if r in COACH_TIPS]
    if not tips:
        tips = ["reach out to a human program reviewer for the specific next step on this file"]
    intro = (
        "Here's what would help move this household's file forward, based only on the reasons the "
        "deterministic engine already flagged:"
    )
    body = " ".join(f"({i + 1}) {t}." for i, t in enumerate(tips))
    outro = (
        "These next steps are drawn only from reasons already on file; a human program reviewer "
        "confirms anything submitted."
    )
    return f"{intro} {body} {outro}"


def coach(profile: dict) -> Tuple[str, str]:
    """For NEEDS_REVIEW households: friendly, concrete next steps derived
    from reasons/notes. No new facts. Always returns (text, source)."""
    fallback = _fallback_coach(profile)
    if profile.get("readiness_status") != "NEEDS_REVIEW":
        return fallback, "deterministic_fallback"
    context = _profile_context(profile)
    prompt = (
        f"CONTEXT:\n{context}\n\n"
        "This household is NEEDS_REVIEW. Using ONLY the reasons and notes in CONTEXT, write friendly, "
        "concrete next steps the applicant could take to address each flagged reason (e.g. an expired "
        "employment letter means asking the employer for a current one). Do not invent any reason, "
        "document, or fact not already in CONTEXT."
    )
    candidate = _complete(prompt)
    if candidate and _guard(candidate):
        return candidate, "openai"
    return fallback, "deterministic_fallback"


# ---------------------------------------------------------------------
# ask(question, profile, rule_context)
# ---------------------------------------------------------------------
def _fallback_ask(question: str, profile: dict, rule_context: list) -> str:
    # Reuse the deterministic per-household Q&A responder directly -- it
    # already answers threshold / annualized / comparison / readiness /
    # eligibility-boundary questions grounded in this exact profile, with
    # a citation, and abstains (via classify_and_respond) on anything else
    # it doesn't recognize.
    resp = answer_household_question(question, profile)
    if resp.category != "uncategorized":
        return resp.answer_text

    q = question.lower()
    if "review" in q or "why" in q:
        reasons = profile.get("reasons", [])
        if reasons:
            reason_bits = "; ".join(REASON_TEXT.get(r, r.replace("_", " ").lower()) for r in reasons)
            return (
                f"This household is {profile.get('readiness_status', 'NEEDS_REVIEW').replace('_', ' ')} "
                f"because: {reason_bits}. These reasons come from the deterministic engine; a human "
                "program reviewer makes any final determination."
            )
        return (
            "No readiness reasons are flagged for this household in the deterministic engine's output. "
            "A human program reviewer makes any final determination."
        )
    return (
        "I don't have a cited fact for that in this household's file. A human program reviewer handles "
        "anything beyond the computed income, threshold comparison, and readiness reasons shown here."
    )


def ask(question: str, profile: dict, rule_context: list) -> Tuple[str, str]:
    """Answer grounded ONLY in the profile facts + the provided rule ids.
    If the question needs anything outside that context, say a human
    reviewer handles it. Always returns (text, source)."""
    # --- Defense-in-depth guardrail: never let the model see (or answer)
    # a decision-boundary / out-of-scope question, even if the caller's
    # own adversarial-category firewall didn't happen to match the
    # phrasing. These always resolve deterministically, no API call. ---
    if _DECISION_BOUNDARY_RE.search(question):
        return (
            "This tool can report the numerical comparison and readiness status only. A human "
            "program reviewer makes any program determination.",
            "deterministic_fallback",
        )
    if _CROSS_APPLICANT_RE.search(question):
        return (
            "I can't share another household's documents, income, or extracted fields. Each session "
            "is scoped to a single applicant's own submitted evidence.",
            "deterministic_fallback",
        )
    if _WRONG_YEAR_RE.search(question):
        return (
            "This simulation is frozen to the FY 2026 MTSP corpus, effective 2026-05-01. A 2025 (or "
            "any other year's) threshold cannot be substituted.",
            "deterministic_fallback",
        )
    if _VACANCY_RE.search(question):
        return (
            "HUD's LIHTC dataset describes project inventory; it is not a current vacancy, rent, "
            "waitlist, or application-status feed. Availability cannot be reported from it.",
            "deterministic_fallback",
        )
    if _PROTECTED_TRAIT_RE.search(question):
        return (
            "I can't infer disability, immigration, or other protected-trait status from a document; "
            "only allowlisted income/readiness fields are ever extracted.",
            "deterministic_fallback",
        )
    if _INJECTION_RE.search(question):
        return (
            "Text embedded in a document is untrusted data. Any instruction-like text is logged to "
            "the quarantine record and never executed or acted upon.",
            "deterministic_fallback",
        )

    fallback = _fallback_ask(question, profile, rule_context)
    context = _profile_context(profile)
    rule_text = "\n".join(
        f"- {rid}: {RULES[rid]['text']}" for rid in (rule_context or []) if rid in RULES
    )
    prompt = (
        f"CONTEXT (household facts):\n{context}\n\n"
        f"CONTEXT (cited rule text):\n{rule_text or 'none provided'}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using ONLY the facts in CONTEXT above. If the question asks for anything not covered "
        "by CONTEXT, say plainly that a human program reviewer handles it -- do not guess."
    )
    candidate = _complete(prompt)
    if candidate and _guard(candidate):
        return candidate, "openai"
    return fallback, "deterministic_fallback"


# ---------------------------------------------------------------------
# Streaming variants -- the full guarded text is resolved first (so the
# guard always runs on the complete candidate before anything is shown),
# then handed back as (source, chunk_iterator) so a caller can set a
# response header from `source` before it starts writing the streamed
# body.
# ---------------------------------------------------------------------
def _chunk_text(text: str, size: int = 8) -> Iterator[str]:
    for i in range(0, len(text), size):
        yield text[i:i + size]


def explain_stream(profile: dict) -> Tuple[str, Iterator[str]]:
    text, source = explain(profile)
    return source, _chunk_text(text)


def coach_stream(profile: dict) -> Tuple[str, Iterator[str]]:
    text, source = coach(profile)
    return source, _chunk_text(text)


def ask_stream(question: str, profile: dict, rule_context: list) -> Tuple[str, Iterator[str]]:
    text, source = ask(question, profile, rule_context)
    return source, _chunk_text(text)
