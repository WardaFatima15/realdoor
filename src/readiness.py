"""Document-readiness reasoning: reproduces the 6 gold checklist rows and
generalizes the same rules to new households.

Never emits an eligibility/approval/denial decision -- only a readiness
status (READY_TO_REVIEW / NEEDS_REVIEW) and machine-checkable reason codes.
Uses the shipped, unmodified ``calculate.annualize`` /
``calculate.compare_to_threshold`` for all arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from datetime import date

from .extract import DocumentExtraction, validate_extraction_boxes
from .shipped import FREQUENCY, annualize, compare_to_threshold
from .thresholds import (
    THRESHOLD_EFFECTIVE_DATE,
    THRESHOLD_RULE_ID,
    THRESHOLD_SOURCE_LOCATOR,
    lookup_threshold,
)

EVENT_DATE = date(2026, 7, 18)
CURRENCY_CUTOFF_DAYS = 60


def _parse_iso_date(s: str) -> date:
    parts = s.split("-")
    y = int(parts[0])
    m = int(parts[1])
    d = int(parts[2]) if len(parts) > 2 else 1
    return date(y, m, d)


def is_current(date_str: str, event_date: date = EVENT_DATE, cutoff_days: int = CURRENCY_CUTOFF_DAYS) -> bool:
    """A document is current when dated no more than `cutoff_days` before
    `event_date` (a frozen challenge convention, CH-READINESS-001)."""
    d = _parse_iso_date(date_str)
    return (event_date - d).days <= cutoff_days


def _fields_dict(extraction: DocumentExtraction) -> dict:
    return {f.field: f.value for f in extraction.fields}


@dataclass
class IncomeComponent:
    source_document_id: str
    document_type: str
    basis: str
    amount: float
    frequency: str
    annualized: float


@dataclass
class ReadinessResult:
    household_id: str
    annualized_income: float
    threshold: object  # float or None
    comparison: str
    readiness_status: str
    reasons: list
    notes: list
    income_components: list  # list[IncomeComponent]


def evaluate_household(household_id: str, household_size: int, extractions: list) -> ReadinessResult:
    reasons: list[str] = []
    notes: list[str] = []
    components: list[IncomeComponent] = []

    by_type: dict[str, list[DocumentExtraction]] = {}
    for e in extractions:
        by_type.setdefault(e.document_type, []).append(e)

    # --- Traceability gate: any malformed bbox anywhere fails the check ---
    box_errors = validate_extraction_boxes(extractions)
    if box_errors:
        reasons.append("TRACEABILITY_FAILURE")

    # --- Wage income from the most recent pay stub ---
    pay_stubs = by_type.get("pay_stub", [])
    base_wage = None
    if not pay_stubs:
        reasons.append("PAY_STUB_MISSING")
    else:
        primary = max(pay_stubs, key=lambda e: _fields_dict(e).get("pay_date", ""))
        fd = _fields_dict(primary)
        regular_hours = fd.get("regular_hours")
        hourly_rate = fd.get("hourly_rate")
        gross_pay = fd.get("gross_pay")
        pay_frequency = fd.get("pay_frequency")

        computed = None
        if regular_hours is not None and hourly_rate is not None:
            computed = round(regular_hours * hourly_rate, 2)

        if gross_pay is not None and computed is not None and abs(computed - gross_pay) > 0.01:
            reasons.append("PAY_STUB_TOTAL_CONFLICT")
            base_wage = computed  # trust the reconciled components, never the conflicting stated total
        elif gross_pay is not None:
            base_wage = gross_pay
        elif computed is not None:
            base_wage = computed

        if base_wage is None:
            reasons.append("WAGE_INCOME_UNRESOLVED")
        elif pay_frequency not in FREQUENCY:
            reasons.append("PAY_FREQUENCY_UNKNOWN")
            base_wage = None
        else:
            wage_annual = annualize(base_wage, pay_frequency)
            components.append(
                IncomeComponent(
                    source_document_id=primary.document_id,
                    document_type="pay_stub",
                    basis="regular_hours_x_hourly_rate" if "PAY_STUB_TOTAL_CONFLICT" in reasons else "gross_pay",
                    amount=base_wage,
                    frequency=pay_frequency,
                    annualized=wage_annual,
                )
            )

    # --- Employment letter: supporting/template evidence, never the income source ---
    employment_letters = by_type.get("employment_letter", [])
    if employment_letters:
        for e in employment_letters:
            doc_date = _fields_dict(e).get("document_date")
            if doc_date and not is_current(doc_date):
                reasons.append("EMPLOYMENT_LETTER_EXPIRED")
    else:
        if components:  # wage already documented via a pay stub -- informational only
            notes.append(
                "employment_letter not provided; wage evidence is documented via a current, reconciled pay stub."
            )
        else:
            reasons.append("EMPLOYMENT_LETTER_MISSING_NO_WAGE_EVIDENCE")

    # --- Benefit income ---
    for e in by_type.get("benefit_letter", []):
        fd = _fields_dict(e)
        amount = fd.get("monthly_benefit")
        freq = fd.get("benefit_frequency")
        doc_date = fd.get("document_date")
        if doc_date and not is_current(doc_date):
            reasons.append("BENEFIT_LETTER_EXPIRED")
        if amount is not None and freq in FREQUENCY:
            annual = annualize(amount, freq)
            components.append(
                IncomeComponent(
                    source_document_id=e.document_id,
                    document_type="benefit_letter",
                    basis="monthly_benefit",
                    amount=amount,
                    frequency=freq,
                    annualized=annual,
                )
            )

    # --- Gig income: always flagged uncorroborated (no independent
    # corroboration document type exists in this pack) ---
    gig_corroboration_present = bool(by_type.get("gig_income_corroboration"))
    for e in by_type.get("gig_statement", []):
        fd = _fields_dict(e)
        receipts = fd.get("gross_receipts")
        if receipts is not None:
            annual = annualize(receipts, "monthly")
            components.append(
                IncomeComponent(
                    source_document_id=e.document_id,
                    document_type="gig_statement",
                    basis="gross_receipts",
                    amount=receipts,
                    frequency="monthly",
                    annualized=annual,
                )
            )
            if not gig_corroboration_present:
                reasons.append("GIG_INCOME_UNCORROBORATED")

    # --- Self-declared-only income (unsigned claim): application_summary is
    # a claim to verify, never authoritative evidence on its own. ---
    if not components and by_type.get("application_summary"):
        reasons.append("UNVERIFIED_CLAIM")

    annualized_income = round(sum(c.annualized for c in components), 2)

    # --- Threshold lookup (a lookup-level concern, not compare_to_threshold) ---
    threshold = lookup_threshold(household_size)
    if threshold is None:
        comparison = "no_frozen_threshold"
        reasons.append("NO_FROZEN_THRESHOLD_FOR_HOUSEHOLD_SIZE")
    else:
        comparison = compare_to_threshold(annualized_income, threshold)

    # de-duplicate reasons while preserving order
    seen = set()
    deduped_reasons = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            deduped_reasons.append(r)

    readiness_status = "READY_TO_REVIEW" if not deduped_reasons else "NEEDS_REVIEW"

    return ReadinessResult(
        household_id=household_id,
        annualized_income=annualized_income,
        threshold=threshold,
        comparison=comparison,
        readiness_status=readiness_status,
        reasons=deduped_reasons,
        notes=notes,
        income_components=components,
    )
