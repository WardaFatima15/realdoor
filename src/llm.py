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

Every function also accepts an optional ``language`` of ``"en"`` (default)
or ``"es"`` -- when ``"es"``, the model is instructed to write its wording
in plain, warm Spanish and a hand-written Spanish deterministic fallback is
used instead of the English one. The post-generation guard
(``FORBIDDEN_RE``) is bilingual: it discards a candidate that slips in a
forbidden decision/ranking/guarantee/prediction word in *either* English or
Spanish, regardless of which language was requested, because a model can
answer in the wrong language or mix words from both. English behavior with
the default ``language="en"`` is unchanged.

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
# Guardrail: forbidden decision / ranking / prediction words -- BILINGUAL
# (English + Spanish). Matches, in either language and any case:
#   English:  eligible, ineligible, approved, denied, qualify/qualifies,
#             rank, priority/prioritize, guarantee, predict (and variants)
#   Spanish:  elegible(s)/inelegible(s)/elegibilidad, aprobado/aprobar/
#             aprueba/apruebe, denegado/denegar/deniega, rechazado/
#             rechazar/rechaza, califica/calificar/calificado, clasificado/
#             clasificar, prioridad/prioritario/prioriza(r), garantiza(r)/
#             garantía (and unaccented "garantia"), predice/predecir/
#             predicción (and unaccented "prediccion")/pronostica(r)
# Every Spanish stem below is written unaccented so the trailing \w*
# (which is Unicode-aware in Python 3's re) matches both the correctly
# accented form (e.g. "calificaría", "predicción", "garantía") and common
# unaccented misspellings (e.g. "calificaria", "prediccion", "garantia")
# without needing any separate accent-stripping trick. If a generated
# string matches this, in either language, it is discarded outright.
# ---------------------------------------------------------------------
_ES_FORBIDDEN = (
    r"eleg(?:ible|ibles|ibilidad)|ineleg(?:ible|ibles)"
    r"|aprob\w*|aprueb\w*"
    r"|deneg\w*|denieg\w*"
    r"|rechaz\w*"
    r"|calific\w*|clasific\w*"
    r"|priorida\w*|prioritari\w*|prioriz\w*"
    r"|garant(?:[ií]a\w*|iz\w*)"
    r"|predec\w*|predic\w*|pron[oó]stic\w*"
)

FORBIDDEN_RE = re.compile(
    r"(?i)\b(eligible|ineligible|approved|denied|qualif\w*|rank\w*|priorit\w*|guarantee\w*|predict\w*"
    r"|" + _ES_FORBIDDEN + r")\b"
)

# Decision-boundary / out-of-scope questions that must NEVER reach the
# model, even when phrased in a way the server-side adversarial
# classifier (rules_qa.classify_and_respond) doesn't happen to match
# (e.g. a bare "am I eligible?" or its Spanish equivalent "¿soy
# elegible?"). This is a second, independent guardrail layer inside the
# explanation module itself -- defense in depth, mirroring the same
# branches rules_qa.py and the static app's JS already refuse on. It is
# also bilingual: a Spanish decision-seeking question must be refused
# pre-model exactly like an English one. NOTE (known limitation): the
# deterministic refusal *text* returned below stays English even when the
# question was asked in Spanish -- classify_and_respond (the server-side
# firewall in src/rules_qa.py) is English-only-worded too; what matters
# for the no-decisioning boundary is that the refusal still fires before
# any model call, in either language, which it does.
_DECISION_BOUNDARY_RE = re.compile(
    r"(?i)\b(eligib\w*|ineligib\w*|approv\w*|den(y|ied)\w*|qualif\w*"
    r"|eleg(?:ible|ibles|ibilidad)|ineleg(?:ible|ibles)"
    r"|aprob\w*|aprueb\w*"
    r"|deneg\w*|denieg\w*"
    r"|rechaz\w*"
    r"|calific\w*"
    r"|garantiz\w*"
    r"|me toca"
    r")\b"
)
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

# Appended to SYSTEM_PROMPT only when language="es". All hard rules above
# still apply without exception -- this only adds the response-language
# instruction and spells out the Spanish forbidden-word list explicitly
# (English "es" is the only extra language supported; anything else is
# coerced back to "en" before this is ever used).
SYSTEM_PROMPT_ES_SUFFIX = """

RESPONSE LANGUAGE: Spanish (es). Write your entire reply in plain, warm, simple Spanish, as if
speaking directly and kindly to the applicant. Every hard rule above still applies without any
exception in Spanish: never state a new number, date, or threshold not verbatim in CONTEXT, and
never use a Spanish decision/ranking/guarantee/prediction word (or any variant of one) either,
including but not limited to: elegible, inelegible, elegibilidad, aprobado, aprobada, aprobar,
aprueba, denegado, denegar, deniega, rechazado, rechazar, rechaza, califica, calificar, calificado,
clasificado, clasificar, prioridad, prioritario, prioriza, priorizar, garantiza, garantizar,
garantía, predice, predecir, predicción, pronostica. Reuse the exact figures given in CONTEXT
verbatim; never translate a number or do arithmetic of your own.
"""


def _normalize_language(language: str) -> str:
    """Only "en" and "es" are supported; anything else (None, "", a typo,
    an unsupported language code) silently coerces to "en" -- the default,
    byte-identical English behavior is never affected by a bad value."""
    return language if language in ("en", "es") else "en"


def _system_prompt(language: str) -> str:
    return SYSTEM_PROMPT + SYSTEM_PROMPT_ES_SUFFIX if language == "es" else SYSTEM_PROMPT


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

# Spanish equivalents of COACH_TIPS / REASON_TEXT above, hand-written (not
# machine-translated), used only for the Spanish deterministic fallbacks.
# Checked against the bilingual FORBIDDEN_RE (see tests/test_llm_guard.py):
# zero forbidden words, English or Spanish.
COACH_TIPS_ES = {
    "EMPLOYMENT_LETTER_EXPIRED": "pedirle al empleador una nueva carta de empleo fechada dentro de los últimos 60 días",
    "BENEFIT_LETTER_EXPIRED": "solicitar una carta de beneficios actualizada fechada dentro de los últimos 60 días",
    "PAY_STUB_TOTAL_CONFLICT": "pedir un talón de pago claro y detallado para que las horas regulares multiplicadas por la tarifa por hora coincidan con el pago bruto declarado",
    "GIG_INCOME_UNCORROBORATED": "agregar un documento de respaldo independiente para el ingreso por trabajos independientes, como un estado de cuenta de ingresos emitido por la plataforma",
    "PAY_STUB_MISSING": "enviar un talón de pago actual para poder documentar el ingreso salarial",
    "WAGE_INCOME_UNRESOLVED": "enviar un talón de pago con horas regulares, tarifa por hora y pago bruto claros para poder conciliar el ingreso salarial",
    "PAY_FREQUENCY_UNKNOWN": "confirmar la frecuencia de pago en el talón de pago (semanal, quincenal, dos veces al mes o mensual)",
    "EMPLOYMENT_LETTER_MISSING_NO_WAGE_EVIDENCE": "enviar un talón de pago actual o una carta de empleo actual para que el ingreso salarial tenga alguna base documental",
    "TRACEABILITY_FAILURE": "marcar este archivo para que un revisor humano vuelva a verificar la página y el recuadro de origen del campo afectado",
    "NO_FROZEN_THRESHOLD_FOR_HOUSEHOLD_SIZE": "no hay nada que enviar aquí -- los tamaños de hogar mayores a 8 no tienen una fila de umbral fijo y pasan directamente a un revisor humano",
    "UNVERIFIED_CLAIM": "agregar evidencia independiente (un talón de pago, una carta de beneficios, etc.) además del resumen de solicitud autodeclarado",
}

REASON_TEXT_ES = {
    "PAY_STUB_TOTAL_CONFLICT": "el pago bruto declarado en el talón de pago no coincidía con las horas regulares multiplicadas por la tarifa por hora, así que se usó en su lugar el cálculo del componente reconciliado",
    "GIG_INCOME_UNCORROBORATED": "el ingreso por trabajos independientes es autodeclarado y no cuenta con ningún documento de respaldo independiente en el expediente",
    "EMPLOYMENT_LETTER_EXPIRED": "la carta de empleo tiene una fecha de más de 60 días antes de la fecha del evento y ya no está vigente",
    "BENEFIT_LETTER_EXPIRED": "la carta de beneficios tiene una fecha de más de 60 días antes de la fecha del evento y ya no está vigente",
    "PAY_STUB_MISSING": "no hay ningún talón de pago en el expediente, así que no se pudo establecer el ingreso salarial",
    "WAGE_INCOME_UNRESOLVED": "los campos de ingreso salarial no se pudieron conciliar en una cifra utilizable",
    "PAY_FREQUENCY_UNKNOWN": "la frecuencia de pago declarada no corresponde a uno de los valores reconocidos",
    "EMPLOYMENT_LETTER_MISSING_NO_WAGE_EVIDENCE": "no hay carta de empleo ni talón de pago en el expediente, así que el ingreso salarial no tiene ninguna base documental",
    "TRACEABILITY_FAILURE": "uno o más campos extraídos tiene un recuadro de origen fuera de la página del documento y no pasó la verificación de trazabilidad",
    "NO_FROZEN_THRESHOLD_FOR_HOUSEHOLD_SIZE": "no existe una fila de umbral fijo del 60% para este tamaño de hogar (la tabla cubre tamaños de 1 a 8 únicamente)",
    "UNVERIFIED_CLAIM": "la única evidencia de ingreso en el expediente es una autodeclaración, no evidencia independiente",
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


def _complete(user_prompt: str, max_tokens: int = 320, system_prompt: str = SYSTEM_PROMPT) -> Optional[str]:
    """Single guarded call to the chat completions API. Returns the
    candidate text, or None on any failure (missing client, API error,
    empty response) -- never raises out of this module. `system_prompt`
    defaults to the English SYSTEM_PROMPT unchanged; callers pass
    `_system_prompt("es")` for the Spanish variant."""
    client = _client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
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


def _fallback_explain_es(profile: dict) -> str:
    """Hand-written Spanish deterministic fallback -- same facts, same
    placeholders/numbers as `_fallback_explain`, no machine translation.
    Zero forbidden words in either language (see tests/test_llm_guard.py)."""
    hid = profile.get("household_id")
    size = profile.get("household_size")
    income = profile.get("annualized_income")
    threshold = profile.get("threshold")
    comparison = profile.get("comparison")
    status = profile.get("readiness_status")
    reasons = profile.get("reasons", [])

    parts = [
        f"Para el hogar {hid} (tamaño {size}), el motor determinista anualizó el ingreso documentado "
        f"de este hogar en ${income:,.2f}."
    ]
    if threshold is not None:
        cmp_word = "en o por debajo del" if comparison == "below_or_equal" else "por encima del"
        parts.append(
            f"Esa cifra está {cmp_word} umbral fijo del 60% AMI de ${threshold:,.0f} para este tamaño de hogar."
        )
    else:
        parts.append(
            "No existe una fila de umbral fijo para este tamaño de hogar, así que no se hace ninguna "
            "comparación numérica."
        )
    parts.append(f"El estado de preparación calculado es {status.replace('_', ' ').lower()}.")
    if reasons:
        reason_bits = "; ".join(REASON_TEXT_ES.get(r, r.replace("_", " ").lower()) for r in reasons)
        parts.append(f"Eso refleja: {reason_bits}.")
    else:
        parts.append("No se marcó ningún motivo de preparación para este expediente.")
    parts.append(
        "Cada cifra anterior proviene directamente del motor determinista, nunca de un modelo de "
        "lenguaje; un revisor humano del programa toma cualquier determinación final."
    )
    return " ".join(parts)


def explain(profile: dict, language: str = "en") -> Tuple[str, str]:
    """Warm, plain-English (or, when language="es", plain-Spanish)
    summary of annualized income vs. threshold, comparison, readiness
    status, and each reason code, 3-5 sentences. Always returns (text,
    source) with source in {"openai", "deterministic_fallback"}."""
    language = _normalize_language(language)
    context = _profile_context(profile)
    if language == "es":
        fallback = _fallback_explain_es(profile)
        prompt = (
            f"CONTEXTO:\n{context}\n\n"
            "Escribe un resumen cálido y en español sencillo, de 3 a 5 oraciones, sobre la preparación "
            "de este hogar, cubriendo el ingreso anualizado, la comparación con el umbral, el estado de "
            "preparación y cada motivo listado -- usando ÚNICAMENTE los datos en CONTEXTO."
        )
    else:
        fallback = _fallback_explain(profile)
        prompt = (
            f"CONTEXT:\n{context}\n\n"
            "Write a warm, plain-English 3-5 sentence summary of this household's readiness, "
            "covering the annualized income, the threshold comparison, the readiness status, and "
            "each reason listed -- using ONLY the facts in CONTEXT."
        )
    candidate = _complete(prompt, system_prompt=_system_prompt(language))
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


def _fallback_coach_es(profile: dict) -> str:
    """Hand-written Spanish deterministic fallback for `coach`, same
    structure/logic as `_fallback_coach`. Zero forbidden words in either
    language (see tests/test_llm_guard.py)."""
    if profile.get("readiness_status") != "NEEDS_REVIEW":
        return (
            "El estado de preparación de este hogar es READY_TO_REVIEW, así que no se necesita ningún "
            "documento adicional por ahora. Un revisor humano del programa sigue tomando la "
            "determinación final."
        )
    reasons = profile.get("reasons", [])
    tips = [COACH_TIPS_ES[r] for r in reasons if r in COACH_TIPS_ES]
    if not tips:
        tips = [
            "comunicarse con un revisor humano del programa para conocer el siguiente paso "
            "específico de este expediente"
        ]
    intro = (
        "Esto ayudaría a avanzar el expediente de este hogar, basado únicamente en los motivos que "
        "el motor determinista ya marcó:"
    )
    body = " ".join(f"({i + 1}) {t}." for i, t in enumerate(tips))
    outro = (
        "Estos próximos pasos provienen únicamente de motivos ya registrados; un revisor humano del "
        "programa confirma cualquier cosa que se envíe."
    )
    return f"{intro} {body} {outro}"


def coach(profile: dict, language: str = "en") -> Tuple[str, str]:
    """For NEEDS_REVIEW households: friendly, concrete next steps derived
    from reasons/notes. No new facts. Always returns (text, source)."""
    language = _normalize_language(language)
    fallback = _fallback_coach_es(profile) if language == "es" else _fallback_coach(profile)
    if profile.get("readiness_status") != "NEEDS_REVIEW":
        return fallback, "deterministic_fallback"
    context = _profile_context(profile)
    if language == "es":
        prompt = (
            f"CONTEXTO:\n{context}\n\n"
            "Este hogar está en estado NEEDS_REVIEW. Usando ÚNICAMENTE los motivos y notas en "
            "CONTEXTO, escribe pasos siguientes amables y concretos que el solicitante podría tomar "
            "para atender cada motivo marcado (por ejemplo, una carta de empleo vencida significa "
            "pedirle al empleador una nueva). No inventes ningún motivo, documento o dato que no esté "
            "ya en CONTEXTO."
        )
    else:
        prompt = (
            f"CONTEXT:\n{context}\n\n"
            "This household is NEEDS_REVIEW. Using ONLY the reasons and notes in CONTEXT, write friendly, "
            "concrete next steps the applicant could take to address each flagged reason (e.g. an expired "
            "employment letter means asking the employer for a current one). Do not invent any reason, "
            "document, or fact not already in CONTEXT."
        )
    candidate = _complete(prompt, system_prompt=_system_prompt(language))
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


def _fallback_ask_es(question: str, profile: dict, rule_context: list) -> str:
    """Hand-written Spanish deterministic fallback for `ask`. NOTE (known
    limitation): `answer_household_question` (src/rules_qa.py) is itself
    English-worded, so a recognized household-scoped question still comes
    back in English even here -- only the two branches below (the
    NEEDS_REVIEW "why" explanation and the generic "no cited fact"
    abstention) get a hand-written Spanish version. Zero forbidden words
    in either language (see tests/test_llm_guard.py)."""
    resp = answer_household_question(question, profile)
    if resp.category != "uncategorized":
        return resp.answer_text

    q = question.lower()
    if "revis" in q or "por qu" in q or "porque" in q or "why" in q:
        reasons = profile.get("reasons", [])
        if reasons:
            reason_bits = "; ".join(REASON_TEXT_ES.get(r, r.replace("_", " ").lower()) for r in reasons)
            status = profile.get("readiness_status", "NEEDS_REVIEW").replace("_", " ").lower()
            return (
                f"Este hogar está en estado {status} porque: {reason_bits}. Estos motivos provienen del "
                "motor determinista; un revisor humano del programa toma cualquier determinación final."
            )
        return (
            "No se marcó ningún motivo de preparación para este hogar en el resultado del motor "
            "determinista. Un revisor humano del programa toma cualquier determinación final."
        )
    return (
        "No tengo un dato citado sobre eso en el expediente de este hogar. Un revisor humano del "
        "programa atiende cualquier asunto más allá del ingreso calculado, la comparación con el "
        "umbral y los motivos de preparación mostrados aquí."
    )


def ask(question: str, profile: dict, rule_context: list, language: str = "en") -> Tuple[str, str]:
    """Answer grounded ONLY in the profile facts + the provided rule ids.
    If the question needs anything outside that context, say a human
    reviewer handles it. Always returns (text, source)."""
    language = _normalize_language(language)
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

    fallback = _fallback_ask_es(question, profile, rule_context) if language == "es" else _fallback_ask(question, profile, rule_context)
    context = _profile_context(profile)
    rule_text = "\n".join(
        f"- {rid}: {RULES[rid]['text']}" for rid in (rule_context or []) if rid in RULES
    )
    if language == "es":
        prompt = (
            f"CONTEXTO (datos del hogar):\n{context}\n\n"
            f"CONTEXTO (texto de la regla citada):\n{rule_text or 'ninguno proporcionado'}\n\n"
            f"PREGUNTA: {question}\n\n"
            "Responde usando ÚNICAMENTE los datos en CONTEXTO arriba. Si la pregunta pide algo no "
            "cubierto por CONTEXTO, di claramente que un revisor humano del programa lo atiende -- no "
            "adivines."
        )
    else:
        prompt = (
            f"CONTEXT (household facts):\n{context}\n\n"
            f"CONTEXT (cited rule text):\n{rule_text or 'none provided'}\n\n"
            f"QUESTION: {question}\n\n"
            "Answer using ONLY the facts in CONTEXT above. If the question asks for anything not covered "
            "by CONTEXT, say plainly that a human program reviewer handles it -- do not guess."
        )
    candidate = _complete(prompt, system_prompt=_system_prompt(language))
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


def explain_stream(profile: dict, language: str = "en") -> Tuple[str, Iterator[str]]:
    text, source = explain(profile, language)
    return source, _chunk_text(text)


def coach_stream(profile: dict, language: str = "en") -> Tuple[str, Iterator[str]]:
    text, source = coach(profile, language)
    return source, _chunk_text(text)


def ask_stream(question: str, profile: dict, rule_context: list, language: str = "en") -> Tuple[str, Iterator[str]]:
    text, source = ask(question, profile, rule_context, language)
    return source, _chunk_text(text)
