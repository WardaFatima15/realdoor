"""Vercel Python entry point -- re-exports the same Flask app from
server.py unchanged, so local (`python server.py`) and deployed
(Vercel) behavior stay identical. Vercel's Python runtime detects the
module-level `app` WSGI callable automatically.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import app  # noqa: E402,F401
