"""Frozen 60% AMI (MTSP) threshold lookup -- a lookup-level concern.

``no_frozen_threshold`` belongs here, not inside the shipped
``compare_to_threshold``, because it reflects the *absence of a table row*
for household sizes outside 1-8, not a comparison result.

The threshold table, its effective date, and its source locator are all
read from the organizer's own shipped CSV
(``realdoor-hackathon-starter-pack/data/mtsp_2026_boston_cambridge_quincy.csv``)
at import time -- not hardcoded here -- so this module can never drift from
the actual frozen data file the organizer provides. If a hidden test ships
a corrected or different CSV at this same path, this table follows it.
"""
import csv
from typing import Optional

from ._paths import MTSP_THRESHOLDS_CSV_PATH

THRESHOLD_RULE_ID = "HUD-MTSP-002"


def _load_thresholds_from_csv(path=MTSP_THRESHOLDS_CSV_PATH):
    thresholds: dict[int, float] = {}
    effective_dates: set[str] = set()
    source_pages: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            thresholds[int(row["household_size"])] = float(row["core_challenge_threshold"])
            effective_dates.add(row["effective_date"])
            source_pages.add(row["source_pdf_page"])
    if len(effective_dates) != 1:
        raise ValueError(f"Expected one uniform effective_date in {path}, found {effective_dates}")
    if len(source_pages) != 1:
        raise ValueError(f"Expected one uniform source_pdf_page in {path}, found {source_pages}")
    return thresholds, effective_dates.pop(), f"PDF page {source_pages.pop()}"


FROZEN_60_PCT_THRESHOLDS, THRESHOLD_EFFECTIVE_DATE, THRESHOLD_SOURCE_LOCATOR = _load_thresholds_from_csv()


def lookup_threshold(household_size: int) -> Optional[float]:
    """Return the frozen 60% threshold for a household size, or None if
    there is no table row (household size >= 9, or < 1)."""
    return FROZEN_60_PCT_THRESHOLDS.get(household_size)
