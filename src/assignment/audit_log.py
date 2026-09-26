"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[tuple[str, str | None], dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Start a request; use distinct IDs for overlapping requests by a user."""
        key = (user_id, request_id)
        if key in self._open:
            raise ValueError("Request already open; use a unique request_id")
        self._open[key] = {
            "user_id": user_id,
            "request_id": request_id,
            "input": text,
            "timestamp": utc_now_iso(),
            "started": time.perf_counter(),
        }

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Finish a matching request and record elapsed milliseconds.

        Missing inputs raise ValueError rather than inventing a question or
        latency. This observer records the supplied decision, never makes it.
        """
        key = (user_id, request_id)
        if key not in self._open:
            raise ValueError("No matching input; call record_input first")
        pending = self._open.pop(key)
        started = pending.pop("started")
        self.logs.append({
            **pending,
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "completed_at": utc_now_iso(),
        })

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
