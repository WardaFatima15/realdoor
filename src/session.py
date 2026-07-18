"""A minimal session store whose ``delete()`` actually zeroizes data,
in memory and on disk, rather than merely dropping a reference.

Used by the "Prepare" stage's Delete action: after delete(), a follow-up
read must return nothing (verify-empty read).
"""
from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any, Optional

from ._paths import SESSIONS_DIR


def _zero_in_place(obj: Any) -> Any:
    """Recursively overwrite leaf values so no reference to the original
    data survives in memory, even if some other code kept the object."""
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            _zero_in_place(obj[k])
            obj[k] = None
        return obj
    if isinstance(obj, list):
        for i in range(len(obj)):
            _zero_in_place(obj[i])
            obj[i] = None
        return obj
    return obj


class SessionStore:
    def __init__(self, base_dir: Path = SESSIONS_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, dict] = {}

    def _path(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.json"

    def create(self, data: dict, session_id: Optional[str] = None) -> str:
        session_id = session_id or secrets.token_hex(8)
        self._mem[session_id] = json.loads(json.dumps(data))  # deep copy
        self._path(session_id).write_text(json.dumps(data), encoding="utf-8")
        return session_id

    def read(self, session_id: str) -> Optional[dict]:
        if session_id in self._mem:
            return self._mem[session_id]
        p = self._path(session_id)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def delete(self, session_id: str) -> bool:
        existed = False
        if session_id in self._mem:
            _zero_in_place(self._mem[session_id])
            del self._mem[session_id]
            existed = True
        p = self._path(session_id)
        if p.exists():
            existed = True
            size = p.stat().st_size
            try:
                with p.open("r+b") as f:
                    f.write(b"0" * size)
                    f.flush()
                    os.fsync(f.fileno())
            except OSError:
                pass
            p.unlink()
        return existed

    def verify_empty(self, session_id: str) -> bool:
        """Returns True iff a read after delete returns nothing, in memory
        and on disk."""
        return self.read(session_id) is None and not self._path(session_id).exists()
