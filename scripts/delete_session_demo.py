"""Demonstrates the session delete() -> zeroize -> verify-empty read
contract used by the Prepare stage's Delete action and by `make delete-session`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.session import SessionStore

STORE = SessionStore()


def main() -> int:
    data = {
        "household_id": "HH-DEMO",
        "annualized_income": 56316.0,
        "documents": [{"document_id": "HH-DEMO-D02", "gross_pay": 2166.0}],
    }
    session_id = STORE.create(data)
    read_back = STORE.read(session_id)
    assert read_back is not None and read_back["household_id"] == "HH-DEMO", "create/read failed"
    print(f"[1/3] session '{session_id}' created and readable: PASS")

    deleted = STORE.delete(session_id)
    assert deleted, "delete() reported nothing to delete"
    print(f"[2/3] session '{session_id}' delete() zeroized: PASS")

    empty = STORE.verify_empty(session_id)
    after_read = STORE.read(session_id)
    ok = empty and after_read is None
    print(f"[3/3] verify-empty read after delete returns nothing: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
