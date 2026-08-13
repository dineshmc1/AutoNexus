"""Durable local metadata storage for the Auto Nexus Studio."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class SQLiteRunStore:
    """Persist owner-scoped run state without requiring a cloud database."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    state_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS runs_owner_created "
                "ON runs(owner_id, created_at DESC)"
            )

    def upsert(self, state: dict[str, Any]) -> None:
        serialized = json.dumps(state, separators=(",", ":"), sort_keys=True)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    id, owner_id, status, created_at, updated_at, state_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    owner_id=excluded.owner_id,
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    state_json=excluded.state_json
                """,
                (
                    str(state["id"]),
                    str(state.get("owner_id", "local-user")),
                    str(state.get("status", "unknown")),
                    str(state.get("created_at", "")),
                    str(state.get("updated_at", "")),
                    serialized,
                ),
            )

    def load_all(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT state_json FROM runs ORDER BY created_at"
            ).fetchall()
        states: list[dict[str, Any]] = []
        for (serialized,) in rows:
            try:
                state = json.loads(serialized)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(state, dict) and state.get("id"):
                states.append(state)
        return states
