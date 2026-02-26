from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3
import threading
import uuid
from typing import Any

import numpy as np

_MARKETS = ("win", "top_3", "top_5", "top_10")
_PROB_EPS = 1e-6
_SNAPSHOT_TYPES = {"manual", "live", "pre_event"}
_LIFECYCLE_STATES = {
    "scheduled",
    "pre_event_snapshot_taken",
    "in_play",
    "complete",
    "awaiting_official",
    "outcomes_synced",
    "retrained",
}


@dataclass(frozen=True)
class PendingOutcomeEvent:
    tour: str
    event_id: str
    event_year: int
    event_name: str | None = None
    event_date: str | None = None


@dataclass(frozen=True)
class CalibrationMetrics:
    market: str
    alpha: float = 0.0
    beta: float = 1.0
    samples: int = 0
    positives: int = 0
    brier_before: float | None = None
    brier_after: float | None = None
    logloss_before: float | None = None
    logloss_after: float | None = None


@dataclass(frozen=True)
class LearningSnapshot:
    tour: str
    version: int = 0
    updated_at: datetime | None = None
    markets: dict[str, CalibrationMetrics] = field(default_factory=dict)

    def apply(self, market: str, probabilities: np.ndarray) -> np.ndarray:
        metric = self.markets.get(market)
        values = np.asarray(probabilities, dtype=np.float64)
        if metric is None or metric.samples <= 0:
            return np.clip(values, _PROB_EPS, 1.0 - _PROB_EPS)
        return _logistic_calibrate(values, alpha=metric.alpha, beta=metric.beta)


class LearningStore:
    def __init__(self, database_path: str):
        self._database_path = Path(database_path).expanduser()
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._database_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _initialize(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS simulation_runs (
                  run_id TEXT PRIMARY KEY,
                  created_at TEXT NOT NULL,
                  tour TEXT NOT NULL,
                  event_id TEXT,
                  event_name TEXT,
                  event_year INTEGER,
                  event_date TEXT,
                  simulation_version INTEGER NOT NULL DEFAULT 1,
                  snapshot_type TEXT NOT NULL DEFAULT 'manual',
                  requested_simulations INTEGER,
                  simulations INTEGER,
                  enable_in_play INTEGER NOT NULL DEFAULT 1,
                  in_play_applied INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS simulation_players (
                  run_id TEXT NOT NULL,
                  player_key TEXT NOT NULL,
                  player_id TEXT,
                  player_name TEXT,
                  win_prob REAL NOT NULL,
                  top3_prob REAL NOT NULL,
                  top5_prob REAL NOT NULL,
                  top10_prob REAL NOT NULL,
                  PRIMARY KEY (run_id, player_key),
                  FOREIGN KEY (run_id) REFERENCES simulation_runs(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS event_outcomes (
                  tour TEXT NOT NULL,
                  event_id TEXT NOT NULL,
                  event_year INTEGER NOT NULL,
                  event_name TEXT,
                  event_completed TEXT,
                  player_key TEXT NOT NULL,
                  player_id TEXT,
                  player_name TEXT,
                  finish_rank INTEGER,
                  won INTEGER NOT NULL,
                  top3 INTEGER NOT NULL,
                  top5 INTEGER NOT NULL,
                  top10 INTEGER NOT NULL,
                  PRIMARY KEY (tour, event_id, event_year, player_key)
                );

                CREATE TABLE IF NOT EXISTS calibration_state (
                  tour TEXT NOT NULL,
                  market TEXT NOT NULL,
                  alpha REAL NOT NULL,
                  beta REAL NOT NULL,
                  samples INTEGER NOT NULL,
                  positives INTEGER NOT NULL,
                  brier_before REAL,
                  brier_after REAL,
                  logloss_before REAL,
                  logloss_after REAL,
                  updated_at TEXT NOT NULL,
                  version INTEGER NOT NULL,
                  PRIMARY KEY (tour, market)
                );

                CREATE TABLE IF NOT EXISTS event_lifecycle (
                  tour TEXT NOT NULL,
                  event_id TEXT NOT NULL,
                  event_year INTEGER NOT NULL,
                  event_name TEXT,
                  event_date TEXT,
                  state TEXT NOT NULL DEFAULT 'scheduled',
                  pre_event_run_id TEXT,
                  pre_event_simulation_version INTEGER,
                  outcomes_source TEXT,
                  retrain_version INTEGER,
                  updated_at TEXT NOT NULL,
                  last_note TEXT,
                  PRIMARY KEY (tour, event_id, event_year)
                );

                CREATE INDEX IF NOT EXISTS idx_simulation_runs_event
                  ON simulation_runs (tour, event_id, event_year, created_at);
                CREATE INDEX IF NOT EXISTS idx_outcomes_event
                  ON event_outcomes (tour, event_id, event_year);
                CREATE INDEX IF NOT EXISTS idx_simulation_players_key
                  ON simulation_players (player_key);
                CREATE INDEX IF NOT EXISTS idx_event_lifecycle_recent
                  ON event_lifecycle (tour, updated_at DESC);
                """
            )
            if not _table_has_column(conn, "simulation_runs", "simulation_version"):
                conn.execute(
                    """
                    ALTER TABLE simulation_runs
                    ADD COLUMN simulation_version INTEGER NOT NULL DEFAULT 1
                    """
                )
            conn.execute(
                """
                UPDATE simulation_runs
                SET simulation_version = 1
                WHERE simulation_version IS NULL
                """
            )
            if not _table_has_column(conn, "simulation_runs", "snapshot_type"):
                conn.execute(
                    """
                    ALTER TABLE simulation_runs
                    ADD COLUMN snapshot_type TEXT NOT NULL DEFAULT 'manual'
                    """
                )
            conn.execute(
                """
                UPDATE simulation_runs
                SET snapshot_type = 'manual'
                WHERE snapshot_type IS NULL OR trim(snapshot_type) = ''
                """
            )
            _reconcile_event_year_mismatches(conn)
            conn.commit()

    @staticmethod
    def _normalize_tour(tour: str) -> str:
        return (tour or "pga").strip().lower()

    def peek_next_simulation_version(
        self,
        *,
        tour: str,
        event_id: str | None,
        event_date: str | None,
    ) -> int:
        normalized_tour = self._normalize_tour(tour)
        normalized_event_id = _normalize_token(event_id)
        with self._lock, self._connect() as conn:
            event_year = self._resolve_event_year_locked(
                conn=conn,
                normalized_tour=normalized_tour,
                normalized_event_id=normalized_event_id,
                event_date=event_date,
            )
            return self._next_simulation_version_locked(
                conn=conn,
                normalized_tour=normalized_tour,
                normalized_event_id=normalized_event_id,
                event_year=event_year,
            )

    def record_prediction(
        self,
        *,
        tour: str,
        event_id: str | None,
        event_name: str | None,
        event_date: str | None,
        requested_simulations: int | None,
        simulations: int | None,
        enable_in_play: bool,
        in_play_applied: bool,
        snapshot_type: str = "manual",
        simulation_version: int | None = None,
        players: list[dict[str, Any]],
    ) -> tuple[str, int]:
        if not players:
            return "", 0

        normalized_tour = self._normalize_tour(tour)
        normalized_event_id = _normalize_token(event_id)
        normalized_snapshot_type = _normalize_snapshot_type(snapshot_type)
        run_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        with self._lock, self._connect() as conn:
            event_year = self._resolve_event_year_locked(
                conn=conn,
                normalized_tour=normalized_tour,
                normalized_event_id=normalized_event_id,
                event_date=event_date,
            )
            resolved_simulation_version = (
                int(simulation_version) if simulation_version is not None else 0
            )
            if resolved_simulation_version <= 0:
                resolved_simulation_version = self._next_simulation_version_locked(
                    conn=conn,
                    normalized_tour=normalized_tour,
                    normalized_event_id=normalized_event_id,
                    event_year=event_year,
                )

            conn.execute(
                """
                INSERT INTO simulation_runs (
                  run_id, created_at, tour, event_id, event_name, event_year, event_date,
                  simulation_version, snapshot_type, requested_simulations, simulations, enable_in_play, in_play_applied
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    created_at,
                    normalized_tour,
                    normalized_event_id,
                    event_name,
                    event_year,
                    event_date,
                    resolved_simulation_version,
                    normalized_snapshot_type,
                    requested_simulations,
                    simulations,
                    int(enable_in_play),
                    int(in_play_applied),
                ),
            )

            player_rows: list[tuple[Any, ...]] = []
            for row in players:
                player_id = str(row.get("player_id") or "").strip()
                player_name = str(row.get("player_name") or "").strip()
                player_key = _canonical_player_key(player_id=player_id, player_name=player_name)
                if not player_key:
                    continue
                player_rows.append(
                    (
                        run_id,
                        player_key,
                        player_id or None,
                        player_name or None,
                        _coerce_probability(row.get("win_probability")),
                        _coerce_probability(row.get("top_3_probability")),
                        _coerce_probability(row.get("top_5_probability")),
                        _coerce_probability(row.get("top_10_probability")),
                    )
                )

            if player_rows:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO simulation_players (
                      run_id, player_key, player_id, player_name, win_prob, top3_prob, top5_prob, top10_prob
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    player_rows,
                )

            if normalized_event_id and event_year is not None:
                lifecycle_state = "scheduled"
                if normalized_snapshot_type == "pre_event":
                    lifecycle_state = "pre_event_snapshot_taken"
                elif in_play_applied or normalized_snapshot_type == "live":
                    lifecycle_state = "in_play"

                self._upsert_event_lifecycle_locked(
                    conn=conn,
                    tour=normalized_tour,
                    event_id=normalized_event_id,
                    event_year=int(event_year),
                    event_name=event_name,
                    event_date=event_date,
                    state=lifecycle_state,
                    pre_event_run_id=run_id if normalized_snapshot_type == "pre_event" else None,
                    pre_event_simulation_version=(
                        int(resolved_simulation_version)
                        if normalized_snapshot_type == "pre_event"
                        else None
                    ),
                )
            conn.commit()

        return run_id, int(resolved_simulation_version)

    def _resolve_event_year_locked(
        self,
        *,
        conn: sqlite3.Connection,
        normalized_tour: str,
        normalized_event_id: str | None,
        event_date: str | None,
    ) -> int | None:
        parsed_year = _year_from_date(event_date)
        if parsed_year is not None:
            return parsed_year
        if not normalized_event_id:
            return None
        row = conn.execute(
            """
            SELECT event_year
            FROM simulation_runs
            WHERE tour = ?
              AND event_id = ?
              AND event_year IS NOT NULL
            ORDER BY event_year DESC, created_at DESC
            LIMIT 1
            """,
            (normalized_tour, normalized_event_id),
        ).fetchone()
        if row is None:
            return None
        return int(row["event_year"])

    def _next_simulation_version_locked(
        self,
        *,
        conn: sqlite3.Connection,
        normalized_tour: str,
        normalized_event_id: str | None,
        event_year: int | None,
    ) -> int:
        if normalized_event_id and event_year is not None:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(simulation_version), 0)
                FROM simulation_runs
                WHERE tour = ? AND event_id = ? AND event_year = ?
                """,
                (normalized_tour, normalized_event_id, int(event_year)),
            ).fetchone()
            return int(row[0] or 0) + 1
        if normalized_event_id:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(simulation_version), 0)
                FROM simulation_runs
                WHERE tour = ? AND event_id = ?
                """,
                (normalized_tour, normalized_event_id),
            ).fetchone()
            return int(row[0] or 0) + 1
        row = conn.execute(
            """
            SELECT COALESCE(MAX(simulation_version), 0)
            FROM simulation_runs
            WHERE tour = ?
            """,
            (normalized_tour,),
        ).fetchone()
        return int(row[0] or 0) + 1

    def _upsert_event_lifecycle_locked(
        self,
        *,
        conn: sqlite3.Connection,
        tour: str,
        event_id: str,
        event_year: int,
        event_name: str | None = None,
        event_date: str | None = None,
        state: str | None = None,
        pre_event_run_id: str | None = None,
        pre_event_simulation_version: int | None = None,
        outcomes_source: str | None = None,
        retrain_version: int | None = None,
        last_note: str | None = None,
    ) -> None:
        normalized_state = _normalize_lifecycle_state(state)
        existing = conn.execute(
            """
            SELECT *
            FROM event_lifecycle
            WHERE tour = ? AND event_id = ? AND event_year = ?
            """,
            (tour, event_id, int(event_year)),
        ).fetchone()
        now_text = datetime.now(timezone.utc).isoformat()
        if existing is None:
            row = {
                "tour": tour,
                "event_id": event_id,
                "event_year": int(event_year),
                "event_name": event_name,
                "event_date": event_date,
                "state": normalized_state or "scheduled",
                "pre_event_run_id": pre_event_run_id,
                "pre_event_simulation_version": pre_event_simulation_version,
                "outcomes_source": outcomes_source,
                "retrain_version": retrain_version,
                "updated_at": now_text,
                "last_note": last_note,
            }
        else:
            row = {
                "tour": str(existing["tour"]),
                "event_id": str(existing["event_id"]),
                "event_year": int(existing["event_year"]),
                "event_name": existing["event_name"],
                "event_date": existing["event_date"],
                "state": existing["state"] or "scheduled",
                "pre_event_run_id": existing["pre_event_run_id"],
                "pre_event_simulation_version": existing["pre_event_simulation_version"],
                "outcomes_source": existing["outcomes_source"],
                "retrain_version": existing["retrain_version"],
                "updated_at": now_text,
                "last_note": existing["last_note"],
            }

            if event_name is not None:
                row["event_name"] = event_name
            if event_date is not None:
                row["event_date"] = event_date
            if normalized_state is not None:
                row["state"] = normalized_state
            if pre_event_run_id is not None:
                row["pre_event_run_id"] = pre_event_run_id
            if pre_event_simulation_version is not None:
                row["pre_event_simulation_version"] = int(pre_event_simulation_version)
            if outcomes_source is not None:
                row["outcomes_source"] = str(outcomes_source).strip().lower()
            if retrain_version is not None:
                row["retrain_version"] = int(retrain_version)
            if last_note is not None:
                row["last_note"] = last_note

        conn.execute(
            """
            INSERT OR REPLACE INTO event_lifecycle (
              tour, event_id, event_year, event_name, event_date, state,
              pre_event_run_id, pre_event_simulation_version,
              outcomes_source, retrain_version, updated_at, last_note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["tour"],
                row["event_id"],
                row["event_year"],
                row["event_name"],
                row["event_date"],
                row["state"],
                row["pre_event_run_id"],
                row["pre_event_simulation_version"],
                row["outcomes_source"],
                row["retrain_version"],
                row["updated_at"],
                row["last_note"],
            ),
        )

    def upsert_event_lifecycle(
        self,
        *,
        tour: str,
        event_id: str | None,
        event_year: int | None,
        event_name: str | None = None,
        event_date: str | None = None,
        state: str | None = None,
        pre_event_run_id: str | None = None,
        pre_event_simulation_version: int | None = None,
        outcomes_source: str | None = None,
        retrain_version: int | None = None,
        last_note: str | None = None,
    ) -> None:
        normalized_tour = self._normalize_tour(tour)
        normalized_event_id = _normalize_token(event_id)
        if not normalized_event_id:
            return

        resolved_year = event_year
        if resolved_year is None:
            resolved_year = _year_from_date(event_date)
        if resolved_year is None:
            return

        with self._lock, self._connect() as conn:
            self._upsert_event_lifecycle_locked(
                conn=conn,
                tour=normalized_tour,
                event_id=normalized_event_id,
                event_year=int(resolved_year),
                event_name=event_name,
                event_date=event_date,
                state=state,
                pre_event_run_id=pre_event_run_id,
                pre_event_simulation_version=pre_event_simulation_version,
                outcomes_source=outcomes_source,
                retrain_version=retrain_version,
                last_note=last_note,
            )
            conn.commit()

    def get_pre_event_snapshot(
        self,
        *,
        tour: str,
        event_id: str | None,
        event_year: int | None,
    ) -> dict[str, Any] | None:
        normalized_tour = self._normalize_tour(tour)
        normalized_event_id = _normalize_token(event_id)
        if not normalized_event_id or event_year is None:
            return None
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT run_id, created_at, simulation_version
                FROM simulation_runs
                WHERE tour = ?
                  AND event_id = ?
                  AND event_year = ?
                  AND snapshot_type = 'pre_event'
                ORDER BY simulation_version DESC, created_at DESC
                LIMIT 1
                """,
                (normalized_tour, normalized_event_id, int(event_year)),
            ).fetchone()
        if row is None:
            return None
        return {
            "run_id": str(row["run_id"]),
            "created_at": _parse_iso_datetime(row["created_at"]),
            "simulation_version": int(row["simulation_version"] or 1),
        }

    def get_latest_prediction_snapshot(
        self,
        *,
        tour: str,
        event_id: str | None = None,
        event_year: int | None = None,
    ) -> dict[str, Any] | None:
        normalized_tour = self._normalize_tour(tour)
        normalized_event_id = _normalize_token(event_id)

        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                  run_id,
                  created_at,
                  tour,
                  event_id,
                  event_name,
                  event_year,
                  event_date,
                  simulation_version,
                  snapshot_type,
                  requested_simulations,
                  simulations,
                  enable_in_play,
                  in_play_applied
                FROM simulation_runs
                WHERE tour = ?
                  AND event_id IS NOT NULL
                  AND (? IS NULL OR event_id = ?)
                  AND (? IS NULL OR event_year = ?)
                ORDER BY created_at DESC, simulation_version DESC
                LIMIT 1
                """,
                (
                    normalized_tour,
                    normalized_event_id,
                    normalized_event_id,
                    event_year,
                    event_year,
                ),
            ).fetchone()

            if row is None:
                return None

            player_rows = conn.execute(
                """
                SELECT
                  player_key,
                  player_id,
                  player_name,
                  win_prob,
                  top3_prob,
                  top5_prob,
                  top10_prob
                FROM simulation_players
                WHERE run_id = ?
                ORDER BY win_prob DESC, player_name ASC, player_key ASC
                """,
                (str(row["run_id"]),),
            ).fetchall()

        return {
            "run_id": str(row["run_id"]),
            "created_at": _parse_iso_datetime(row["created_at"]),
            "tour": str(row["tour"]),
            "event_id": str(row["event_id"]) if row["event_id"] is not None else None,
            "event_name": row["event_name"],
            "event_year": int(row["event_year"]) if row["event_year"] is not None else None,
            "event_date": row["event_date"],
            "simulation_version": int(row["simulation_version"] or 1),
            "snapshot_type": _normalize_snapshot_type(row["snapshot_type"]),
            "requested_simulations": (
                int(row["requested_simulations"])
                if row["requested_simulations"] is not None
                else None
            ),
            "simulations": int(row["simulations"]) if row["simulations"] is not None else None,
            "enable_in_play": bool(int(row["enable_in_play"])),
            "in_play_applied": bool(int(row["in_play_applied"])),
            "players": [
                {
                    "player_key": str(player_row["player_key"]),
                    "player_id": player_row["player_id"],
                    "player_name": player_row["player_name"],
                    "win_probability": float(player_row["win_prob"]),
                    "top_3_probability": float(player_row["top3_prob"]),
                    "top_5_probability": float(player_row["top5_prob"]),
                    "top_10_probability": float(player_row["top10_prob"]),
                }
                for player_row in player_rows
            ],
        }

    def get_run_player_win_deltas(
        self,
        *,
        run_id: str,
    ) -> list[dict[str, Any]]:
        normalized_run_id = str(run_id or "").strip()
        if not normalized_run_id:
            return []

        with self._lock, self._connect() as conn:
            run_row = conn.execute(
                """
                SELECT tour, event_id, event_year, simulation_version
                FROM simulation_runs
                WHERE run_id = ?
                LIMIT 1
                """,
                (normalized_run_id,),
            ).fetchone()
            if run_row is None:
                return []

            tour = str(run_row["tour"])
            event_id = _normalize_token(run_row["event_id"])
            event_year = int(run_row["event_year"]) if run_row["event_year"] is not None else None
            simulation_version = (
                int(run_row["simulation_version"])
                if run_row["simulation_version"] is not None
                else None
            )
            if not event_id or event_year is None or simulation_version is None:
                return []

            first_row = conn.execute(
                """
                SELECT run_id
                FROM simulation_runs
                WHERE tour = ?
                  AND event_id = ?
                  AND event_year = ?
                ORDER BY simulation_version ASC, created_at ASC, run_id ASC
                LIMIT 1
                """,
                (tour, event_id, int(event_year)),
            ).fetchone()
            first_run_id = str(first_row["run_id"]) if first_row is not None else normalized_run_id

            previous_row = conn.execute(
                """
                SELECT run_id
                FROM simulation_runs
                WHERE tour = ?
                  AND event_id = ?
                  AND event_year = ?
                  AND simulation_version < ?
                ORDER BY simulation_version DESC, created_at DESC, run_id DESC
                LIMIT 1
                """,
                (tour, event_id, int(event_year), int(simulation_version)),
            ).fetchone()
            previous_run_id = str(previous_row["run_id"]) if previous_row is not None else None

            lookup_run_ids = [normalized_run_id]
            if first_run_id and first_run_id not in lookup_run_ids:
                lookup_run_ids.append(first_run_id)
            if previous_run_id and previous_run_id not in lookup_run_ids:
                lookup_run_ids.append(previous_run_id)

            placeholders = ", ".join(["?"] * len(lookup_run_ids))
            player_rows = conn.execute(
                f"""
                SELECT run_id, player_key, player_id, player_name, win_prob
                FROM simulation_players
                WHERE run_id IN ({placeholders})
                """,
                lookup_run_ids,
            ).fetchall()

        by_run: dict[str, dict[str, dict[str, Any]]] = {}
        for row in player_rows:
            row_run_id = str(row["run_id"])
            by_run.setdefault(row_run_id, {})[str(row["player_key"])] = {
                "player_id": row["player_id"],
                "player_name": row["player_name"],
                "win_probability": float(row["win_prob"]),
            }

        current_players = by_run.get(normalized_run_id, {})
        first_players = by_run.get(first_run_id, {})
        previous_players = by_run.get(previous_run_id, {}) if previous_run_id else {}

        output: list[dict[str, Any]] = []
        for player_key, current_row in current_players.items():
            current_win = float(current_row["win_probability"])
            first_row_for_player = first_players.get(player_key)
            previous_row_for_player = previous_players.get(player_key)

            delta_start = None
            if first_row_for_player is not None:
                delta_start = current_win - float(first_row_for_player["win_probability"])

            delta_prev = None
            if previous_row_for_player is not None:
                delta_prev = current_win - float(previous_row_for_player["win_probability"])

            output.append(
                {
                    "player_key": player_key,
                    "player_id": current_row.get("player_id"),
                    "player_name": current_row.get("player_name"),
                    "delta_win_since_first": float(delta_start) if delta_start is not None else None,
                    "delta_win_since_previous": float(delta_prev) if delta_prev is not None else None,
                }
            )

        return output

    def list_event_lifecycle(
        self,
        *,
        tour: str = "pga",
        max_events: int = 20,
        min_event_year: int | None = None,
        max_event_year: int | None = None,
    ) -> list[dict[str, Any]]:
        normalized_tour = self._normalize_tour(tour)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                  tour,
                  event_id,
                  event_year,
                  event_name,
                  event_date,
                  state,
                  pre_event_simulation_version,
                  outcomes_source,
                  retrain_version,
                  updated_at,
                  last_note
                FROM event_lifecycle
                WHERE tour = ?
                  AND (? IS NULL OR event_year >= ?)
                  AND (? IS NULL OR event_year <= ?)
                ORDER BY updated_at DESC, event_year DESC, event_id DESC
                LIMIT ?
                """,
                (
                    normalized_tour,
                    min_event_year,
                    min_event_year,
                    max_event_year,
                    max_event_year,
                    max(1, int(max_events)),
                ),
            ).fetchall()

        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "tour": str(row["tour"]),
                    "event_id": str(row["event_id"]),
                    "event_year": int(row["event_year"]),
                    "event_name": row["event_name"],
                    "event_date": row["event_date"],
                    "state": str(row["state"] or "scheduled"),
                    "pre_event_snapshot_version": (
                        int(row["pre_event_simulation_version"])
                        if row["pre_event_simulation_version"] is not None
                        else None
                    ),
                    "outcomes_source": row["outcomes_source"],
                    "retrain_version": (
                        int(row["retrain_version"]) if row["retrain_version"] is not None else None
                    ),
                    "updated_at": _parse_iso_datetime(row["updated_at"]),
                    "last_note": row["last_note"],
                }
            )
        return out

    def list_power_ranking_event_inputs(
        self,
        *,
        tour: str = "pga",
        event_year: int | None = None,
        max_events: int = 16,
    ) -> list[dict[str, Any]]:
        normalized_tour = self._normalize_tour(tour)
        with self._lock, self._connect() as conn:
            event_rows = conn.execute(
                """
                SELECT
                  event_id,
                  event_year,
                  MAX(event_name) AS event_name,
                  MAX(event_date) AS event_date
                FROM simulation_runs
                WHERE tour = ?
                  AND event_id IS NOT NULL
                  AND event_year IS NOT NULL
                  AND (? IS NULL OR event_year = ?)
                GROUP BY event_id, event_year
                ORDER BY
                  CASE
                    WHEN MAX(event_date) IS NULL OR trim(MAX(event_date)) = '' THEN 1
                    ELSE 0
                  END,
                  MAX(event_date) ASC,
                  event_year ASC,
                  event_id ASC
                """,
                (
                    normalized_tour,
                    event_year,
                    event_year,
                ),
            ).fetchall()

            events: list[dict[str, Any]] = []
            for row in event_rows:
                event_id = _normalize_token(row["event_id"])
                if not event_id:
                    continue
                year_value = int(row["event_year"])
                run_row = conn.execute(
                    """
                    SELECT
                      run_id,
                      created_at,
                      simulation_version,
                      snapshot_type,
                      event_name,
                      event_date
                    FROM simulation_runs
                    WHERE tour = ?
                      AND event_id = ?
                      AND event_year = ?
                    ORDER BY
                      CASE
                        WHEN snapshot_type = 'pre_event' THEN 2
                        WHEN snapshot_type = 'manual' THEN 1
                        ELSE 0
                      END DESC,
                      simulation_version DESC,
                      created_at DESC
                    LIMIT 1
                    """,
                    (normalized_tour, event_id, year_value),
                ).fetchone()
                if run_row is None:
                    continue

                player_rows = conn.execute(
                    """
                    SELECT
                      player_key,
                      player_id,
                      player_name,
                      win_prob,
                      top3_prob,
                      top5_prob,
                      top10_prob
                    FROM simulation_players
                    WHERE run_id = ?
                    """,
                    (str(run_row["run_id"]),),
                ).fetchall()
                if not player_rows:
                    continue

                outcome_rows = conn.execute(
                    """
                    SELECT
                      player_key,
                      finish_rank,
                      won,
                      top3,
                      top5,
                      top10
                    FROM event_outcomes
                    WHERE tour = ?
                      AND event_id = ?
                      AND event_year = ?
                    """,
                    (normalized_tour, event_id, year_value),
                ).fetchall()

                events.append(
                    {
                        "event_id": event_id,
                        "event_year": year_value,
                        "event_name": (
                            run_row["event_name"]
                            if run_row["event_name"] is not None
                            else row["event_name"]
                        ),
                        "event_date": (
                            run_row["event_date"]
                            if run_row["event_date"] is not None
                            else row["event_date"]
                        ),
                        "run_id": str(run_row["run_id"]),
                        "created_at": _parse_iso_datetime(run_row["created_at"]),
                        "simulation_version": int(run_row["simulation_version"] or 1),
                        "snapshot_type": _normalize_snapshot_type(run_row["snapshot_type"]),
                        "players": [
                            {
                                "player_key": str(player_row["player_key"]),
                                "player_id": player_row["player_id"],
                                "player_name": player_row["player_name"],
                                "win_probability": float(player_row["win_prob"]),
                                "top_3_probability": float(player_row["top3_prob"]),
                                "top_5_probability": float(player_row["top5_prob"]),
                                "top_10_probability": float(player_row["top10_prob"]),
                            }
                            for player_row in player_rows
                        ],
                        "outcomes": [
                            {
                                "player_key": str(outcome_row["player_key"]),
                                "finish_rank": (
                                    int(outcome_row["finish_rank"])
                                    if outcome_row["finish_rank"] is not None
                                    else None
                                ),
                                "won": bool(int(outcome_row["won"])),
                                "top_3": bool(int(outcome_row["top3"])),
                                "top_5": bool(int(outcome_row["top5"])),
                                "top_10": bool(int(outcome_row["top10"])),
                            }
                            for outcome_row in outcome_rows
                        ],
                    }
                )

        if max_events > 0 and len(events) > int(max_events):
            return events[-int(max_events):]
        return events

    def list_pending_events(
        self,
        *,
        tour: str | None = None,
        max_events: int = 40,
        min_event_year: int | None = None,
        max_event_year: int | None = None,
    ) -> list[PendingOutcomeEvent]:
        normalized_tour = self._normalize_tour(tour or "pga") if tour else None
        current_year = datetime.now(timezone.utc).year
        today_text = datetime.now(timezone.utc).date().isoformat()
        sql = """
            SELECT
              r.tour AS tour,
              r.event_id AS event_id,
              r.event_year AS event_year,
              MAX(r.event_name) AS event_name,
              MAX(r.event_date) AS event_date
            FROM simulation_runs r
            WHERE r.event_id IS NOT NULL
              AND r.event_year IS NOT NULL
              AND r.event_year <= ?
              AND (? IS NULL OR r.event_year >= ?)
              AND (? IS NULL OR r.event_year <= ?)
              AND (r.event_date IS NULL OR substr(r.event_date, 1, 10) <= ?)
              AND (? IS NULL OR r.tour = ?)
              AND NOT EXISTS (
                SELECT 1
                FROM event_outcomes o
                WHERE o.tour = r.tour
                  AND o.event_id = r.event_id
                  AND o.event_year = r.event_year
              )
            GROUP BY r.tour, r.event_id, r.event_year
            ORDER BY r.event_year DESC, MAX(r.event_date) DESC
            LIMIT ?
        """

        with self._lock, self._connect() as conn:
            rows = conn.execute(
                sql,
                (
                    current_year,
                    min_event_year,
                    min_event_year,
                    max_event_year,
                    max_event_year,
                    today_text,
                    normalized_tour,
                    normalized_tour,
                    max(1, int(max_events)),
                ),
            ).fetchall()

        return [
            PendingOutcomeEvent(
                tour=str(row["tour"]),
                event_id=str(row["event_id"]),
                event_year=int(row["event_year"]),
                event_name=row["event_name"],
                event_date=row["event_date"],
            )
            for row in rows
        ]

    def record_outcome_payload(
        self,
        *,
        tour: str,
        event_id: str,
        event_year: int,
        payload: Any,
        outcomes_source: str | None = None,
    ) -> int:
        normalized_tour = self._normalize_tour(tour)
        normalized_event_id = _normalize_token(event_id)
        if not normalized_event_id:
            return 0

        event_rows = _extract_event_stats_rows(payload)
        if not event_rows:
            return 0

        completed = _string_from_payload(payload, ("event_completed", "date", "completed"))
        event_name = _string_from_payload(payload, ("event_name", "tournament_name", "event"))

        outcome_rows: list[tuple[Any, ...]] = []
        for row in event_rows:
            player_id = _string_from_row(row, ("dg_id", "player_id", "id", "player"))
            player_name = _string_from_row(row, ("player_name", "name", "player"))
            player_key = _canonical_player_key(player_id=player_id, player_name=player_name)
            if not player_key:
                continue
            finish_rank = _finish_rank_from_row(row)
            outcome_rows.append(
                (
                    normalized_tour,
                    normalized_event_id,
                    int(event_year),
                    event_name,
                    completed,
                    player_key,
                    player_id,
                    player_name,
                    finish_rank,
                    int(finish_rank == 1 if finish_rank is not None else 0),
                    int(finish_rank is not None and finish_rank <= 3),
                    int(finish_rank is not None and finish_rank <= 5),
                    int(finish_rank is not None and finish_rank <= 10),
                )
            )

        if not outcome_rows:
            return 0

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                DELETE FROM event_outcomes
                WHERE tour = ? AND event_id = ? AND event_year = ?
                """,
                (normalized_tour, normalized_event_id, int(event_year)),
            )
            conn.executemany(
                """
                INSERT INTO event_outcomes (
                  tour, event_id, event_year, event_name, event_completed,
                  player_key, player_id, player_name, finish_rank, won, top3, top5, top10
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                outcome_rows,
            )
            self._upsert_event_lifecycle_locked(
                conn=conn,
                tour=normalized_tour,
                event_id=normalized_event_id,
                event_year=int(event_year),
                event_name=event_name,
                event_date=completed,
                state="outcomes_synced",
                outcomes_source=(
                    str(outcomes_source).strip().lower()
                    if outcomes_source is not None
                    else None
                ),
            )
            conn.commit()
        return len(outcome_rows)

    def get_snapshot(self, *, tour: str) -> LearningSnapshot:
        normalized_tour = self._normalize_tour(tour)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT market, alpha, beta, samples, positives,
                       brier_before, brier_after, logloss_before, logloss_after,
                       updated_at, version
                FROM calibration_state
                WHERE tour = ?
                """,
                (normalized_tour,),
            ).fetchall()

        markets: dict[str, CalibrationMetrics] = {}
        latest_version = 0
        latest_updated_at: datetime | None = None
        for row in rows:
            updated = _parse_iso_datetime(row["updated_at"])
            if updated and (latest_updated_at is None or updated > latest_updated_at):
                latest_updated_at = updated
            latest_version = max(latest_version, int(row["version"]))
            markets[str(row["market"])] = CalibrationMetrics(
                market=str(row["market"]),
                alpha=float(row["alpha"]),
                beta=float(row["beta"]),
                samples=int(row["samples"]),
                positives=int(row["positives"]),
                brier_before=_to_float(row["brier_before"]),
                brier_after=_to_float(row["brier_after"]),
                logloss_before=_to_float(row["logloss_before"]),
                logloss_after=_to_float(row["logloss_after"]),
            )

        return LearningSnapshot(
            tour=normalized_tour,
            version=latest_version,
            updated_at=latest_updated_at,
            markets=markets,
        )

    def retrain(
        self,
        *,
        tour: str,
        bump_version: bool = True,
        min_event_year: int | None = None,
        max_event_year: int | None = None,
        reset_if_empty: bool = False,
    ) -> LearningSnapshot:
        normalized_tour = self._normalize_tour(tour)
        current_snapshot = self.get_snapshot(tour=normalized_tour)
        observations = self._load_training_rows(
            tour=normalized_tour,
            min_event_year=min_event_year,
            max_event_year=max_event_year,
        )
        if not observations:
            if reset_if_empty:
                with self._lock, self._connect() as conn:
                    conn.execute(
                        """
                        DELETE FROM calibration_state
                        WHERE tour = ?
                        """,
                        (normalized_tour,),
                    )
                    conn.commit()
                return self.get_snapshot(tour=normalized_tour)
            return current_snapshot

        if bump_version:
            version = max(1, current_snapshot.version + 1)
        else:
            version = max(1, current_snapshot.version)
        updated_at = datetime.now(timezone.utc).isoformat()

        calibration_rows: list[tuple[Any, ...]] = []
        for market in _MARKETS:
            prob_key, label_key = _market_keys(market)
            probs = np.asarray([row[prob_key] for row in observations], dtype=np.float64)
            labels = np.asarray([row[label_key] for row in observations], dtype=np.float64)
            labels = np.where(labels > 0.5, 1.0, 0.0)
            samples = int(probs.size)
            positives = int(labels.sum()) if samples > 0 else 0

            if samples < 100 or positives == 0 or positives == samples:
                alpha = 0.0
                beta = 1.0
                calibrated = np.clip(probs, _PROB_EPS, 1.0 - _PROB_EPS)
            else:
                alpha, beta = _fit_logistic_parameters(probs=probs, labels=labels)
                calibrated = _logistic_calibrate(probs, alpha=alpha, beta=beta)

            brier_before = _brier_score(probs, labels) if samples > 0 else None
            brier_after = _brier_score(calibrated, labels) if samples > 0 else None
            logloss_before = _logloss(probs, labels) if samples > 0 else None
            logloss_after = _logloss(calibrated, labels) if samples > 0 else None

            # Safety rail: never persist a calibration that materially degrades error metrics.
            if (
                samples > 0
                and brier_before is not None
                and brier_after is not None
                and logloss_before is not None
                and logloss_after is not None
                and (
                    brier_after > (brier_before + 1e-6)
                    or logloss_after > (logloss_before + 1e-6)
                )
            ):
                alpha = 0.0
                beta = 1.0
                calibrated = np.clip(probs, _PROB_EPS, 1.0 - _PROB_EPS)
                brier_after = _brier_score(calibrated, labels)
                logloss_after = _logloss(calibrated, labels)

            calibration_rows.append(
                (
                    normalized_tour,
                    market,
                    float(alpha),
                    float(beta),
                    samples,
                    positives,
                    brier_before,
                    brier_after,
                    logloss_before,
                    logloss_after,
                    updated_at,
                    version,
                )
            )

        with self._lock, self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO calibration_state (
                  tour, market, alpha, beta, samples, positives,
                  brier_before, brier_after, logloss_before, logloss_after, updated_at, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                calibration_rows,
            )
            conn.commit()

        return self.get_snapshot(tour=normalized_tour)

    def status(
        self,
        *,
        tour: str,
        min_event_year: int | None = None,
        max_event_year: int | None = None,
    ) -> dict[str, Any]:
        normalized_tour = self._normalize_tour(tour)
        with self._lock, self._connect() as conn:
            predictions_logged = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM simulation_runs
                    WHERE tour = ?
                      AND (? IS NULL OR event_year >= ?)
                      AND (? IS NULL OR event_year <= ?)
                    """,
                    (
                        normalized_tour,
                        min_event_year,
                        min_event_year,
                        max_event_year,
                        max_event_year,
                    ),
                ).fetchone()[0]
            )
            resolved_events = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM (
                      SELECT DISTINCT event_id, event_year
                      FROM event_outcomes
                      WHERE tour = ?
                        AND (? IS NULL OR event_year >= ?)
                        AND (? IS NULL OR event_year <= ?)
                    )
                    """,
                    (
                        normalized_tour,
                        min_event_year,
                        min_event_year,
                        max_event_year,
                        max_event_year,
                    ),
                ).fetchone()[0]
            )
            pending_events = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM (
                      SELECT DISTINCT r.event_id, r.event_year
                      FROM simulation_runs r
                      WHERE r.tour = ?
                        AND r.event_id IS NOT NULL
                        AND r.event_year IS NOT NULL
                        AND (? IS NULL OR r.event_year >= ?)
                        AND (? IS NULL OR r.event_year <= ?)
                        AND (r.event_date IS NULL OR substr(r.event_date, 1, 10) <= ?)
                        AND NOT EXISTS (
                          SELECT 1
                          FROM event_outcomes o
                          WHERE o.tour = r.tour
                            AND o.event_id = r.event_id
                            AND o.event_year = r.event_year
                        )
                    )
                    """,
                    (
                        normalized_tour,
                        min_event_year,
                        min_event_year,
                        max_event_year,
                        max_event_year,
                        datetime.now(timezone.utc).date().isoformat(),
                    ),
                ).fetchone()[0]
            )

        training_rows = self._load_training_rows(
            tour=normalized_tour,
            min_event_year=min_event_year,
            max_event_year=max_event_year,
        )
        resolved_predictions = len(training_rows)
        snapshot = self.get_snapshot(tour=normalized_tour)
        if resolved_predictions > 0:
            markets = [
                snapshot.markets.get(market, CalibrationMetrics(market=market))
                for market in _MARKETS
            ]
            calibration_version = snapshot.version
            calibration_updated_at = snapshot.updated_at
        else:
            markets = [CalibrationMetrics(market=market) for market in _MARKETS]
            calibration_version = 0
            calibration_updated_at = None

        return {
            "tour": normalized_tour,
            "predictions_logged": predictions_logged,
            "resolved_predictions": resolved_predictions,
            "resolved_events": resolved_events,
            "pending_events": pending_events,
            "calibration_version": calibration_version,
            "calibration_updated_at": calibration_updated_at,
            "markets": markets,
        }

    def event_trends(
        self,
        *,
        tour: str,
        event_id: str,
        event_year: int | None = None,
        max_snapshots: int = 80,
        max_players: int = 40,
    ) -> dict[str, Any]:
        normalized_tour = self._normalize_tour(tour)
        normalized_event_id = _normalize_token(event_id)
        if not normalized_event_id:
            raise ValueError("event_id is required for trend queries.")

        max_snapshots = max(2, min(int(max_snapshots), 300))
        max_players = max(1, min(int(max_players), 200))

        with self._lock, self._connect() as conn:
            selected_year = event_year
            if selected_year is None:
                year_row = conn.execute(
                    """
                    SELECT event_year
                    FROM simulation_runs
                    WHERE tour = ?
                      AND event_id = ?
                      AND event_year IS NOT NULL
                    ORDER BY event_year DESC, created_at DESC
                    LIMIT 1
                    """,
                    (normalized_tour, normalized_event_id),
                ).fetchone()
                if year_row is None:
                    return {
                        "tour": normalized_tour,
                        "event_id": normalized_event_id,
                        "event_year": datetime.now(timezone.utc).year,
                        "event_name": None,
                        "snapshot_count": 0,
                        "latest_run_id": None,
                        "latest_simulation_version": None,
                        "snapshots": [],
                        "players": [],
                    }
                selected_year = int(year_row["event_year"])

            snapshot_rows = conn.execute(
                """
                SELECT
                  run_id,
                  created_at,
                  simulation_version,
                  snapshot_type,
                  simulations,
                  in_play_applied,
                  event_name
                FROM simulation_runs
                WHERE tour = ?
                  AND event_id = ?
                  AND event_year = ?
                ORDER BY created_at DESC, run_id DESC
                LIMIT ?
                """,
                (
                    normalized_tour,
                    normalized_event_id,
                    int(selected_year),
                    max_snapshots,
                ),
            ).fetchall()

            if not snapshot_rows:
                return {
                    "tour": normalized_tour,
                    "event_id": normalized_event_id,
                    "event_year": int(selected_year),
                    "event_name": None,
                    "snapshot_count": 0,
                    "latest_run_id": None,
                    "latest_simulation_version": None,
                    "snapshots": [],
                    "players": [],
                }

            snapshot_rows = list(reversed(snapshot_rows))
            snapshot_ids = [str(row["run_id"]) for row in snapshot_rows]
            placeholders = ", ".join(["?"] * len(snapshot_ids))
            player_rows = conn.execute(
                f"""
                SELECT
                  sp.run_id AS run_id,
                  sp.player_key AS player_key,
                  sp.player_id AS player_id,
                  sp.player_name AS player_name,
                  sp.win_prob AS win_prob,
                  sp.top3_prob AS top3_prob,
                  sp.top5_prob AS top5_prob,
                  sp.top10_prob AS top10_prob,
                  r.created_at AS created_at,
                  r.simulation_version AS simulation_version,
                  r.snapshot_type AS snapshot_type
                FROM simulation_players sp
                JOIN simulation_runs r
                  ON r.run_id = sp.run_id
                WHERE sp.run_id IN ({placeholders})
                ORDER BY r.created_at ASC, sp.player_key ASC
                """,
                snapshot_ids,
            ).fetchall()

        snapshot_by_run = {
            str(row["run_id"]): {
                "run_id": str(row["run_id"]),
                "created_at": _parse_iso_datetime(row["created_at"]),
                "simulation_version": int(row["simulation_version"] or 1),
                "snapshot_type": _normalize_snapshot_type(row["snapshot_type"]),
                "simulations": int(row["simulations"]) if row["simulations"] is not None else None,
                "in_play_applied": bool(int(row["in_play_applied"])),
            }
            for row in snapshot_rows
        }
        run_order = [entry["run_id"] for entry in snapshot_by_run.values()]

        per_player: dict[str, dict[str, Any]] = {}
        for row in player_rows:
            player_key = str(row["player_key"])
            bucket = per_player.setdefault(
                player_key,
                {
                    "player_id": row["player_id"],
                    "player_name": row["player_name"] or "Unknown Player",
                    "points_by_run": {},
                },
            )
            point = {
                "run_id": str(row["run_id"]),
                "created_at": _parse_iso_datetime(row["created_at"]),
                "simulation_version": int(row["simulation_version"] or 1),
                "snapshot_type": _normalize_snapshot_type(row["snapshot_type"]),
                "win_probability": float(row["win_prob"]),
                "top_3_probability": float(row["top3_prob"]),
                "top_5_probability": float(row["top5_prob"]),
                "top_10_probability": float(row["top10_prob"]),
            }
            bucket["points_by_run"][point["run_id"]] = point

        if not per_player:
            event_name = snapshot_rows[-1]["event_name"] if snapshot_rows else None
            return {
                "tour": normalized_tour,
                "event_id": normalized_event_id,
                "event_year": int(selected_year),
                "event_name": event_name,
                "snapshot_count": len(snapshot_rows),
                "latest_run_id": run_order[-1] if run_order else None,
                "latest_simulation_version": snapshot_by_run.get(run_order[-1], {}).get("simulation_version")
                if run_order
                else None,
                "snapshots": list(snapshot_by_run.values()),
                "players": [],
            }

        latest_run = run_order[-1]
        sortable_players: list[tuple[float, str]] = []
        for player_key, info in per_player.items():
            latest_point = info["points_by_run"].get(latest_run)
            if latest_point is None:
                continue
            sortable_players.append((float(latest_point["win_probability"]), player_key))
        sortable_players.sort(key=lambda item: item[0], reverse=True)
        selected_keys = [player_key for _, player_key in sortable_players[:max_players]]

        players_payload: list[dict[str, Any]] = []
        for player_key in selected_keys:
            info = per_player[player_key]
            points: list[dict[str, Any]] = []
            for run_id in run_order:
                point = info["points_by_run"].get(run_id)
                if point is None:
                    continue
                points.append(point)
            if not points:
                continue

            latest = points[-1]
            first = points[0]
            previous = points[-2] if len(points) > 1 else None
            players_payload.append(
                {
                    "player_id": info["player_id"],
                    "player_name": info["player_name"] or "Unknown Player",
                    "latest_win_probability": float(latest["win_probability"]),
                    "delta_win_since_first": float(latest["win_probability"] - first["win_probability"])
                    if first
                    else None,
                    "delta_win_since_previous": float(
                        latest["win_probability"] - previous["win_probability"]
                    )
                    if previous is not None
                    else None,
                    "latest_top_3_probability": float(latest["top_3_probability"]),
                    "latest_top_5_probability": float(latest["top_5_probability"]),
                    "latest_top_10_probability": float(latest["top_10_probability"]),
                    "points": points,
                }
            )

        event_name = snapshot_rows[-1]["event_name"] if snapshot_rows else None
        return {
            "tour": normalized_tour,
            "event_id": normalized_event_id,
            "event_year": int(selected_year),
            "event_name": event_name,
            "snapshot_count": len(snapshot_rows),
            "latest_run_id": run_order[-1] if run_order else None,
            "latest_simulation_version": snapshot_by_run.get(run_order[-1], {}).get("simulation_version")
            if run_order
            else None,
            "snapshots": list(snapshot_by_run.values()),
            "players": players_payload,
        }

    def _load_training_rows(
        self,
        *,
        tour: str,
        min_event_year: int | None = None,
        max_event_year: int | None = None,
    ) -> list[dict[str, Any]]:
        sql = """
            WITH candidate AS (
              SELECT
                r.run_id,
                r.created_at,
                r.tour,
                r.event_id,
                r.event_year,
                sp.player_key,
                sp.win_prob,
                sp.top3_prob,
                sp.top5_prob,
                sp.top10_prob,
                ROW_NUMBER() OVER (
                  PARTITION BY r.tour, r.event_id, r.event_year, sp.player_key
                  ORDER BY r.in_play_applied ASC, r.created_at DESC, r.run_id DESC
                ) AS row_rank
              FROM simulation_runs r
              JOIN simulation_players sp
                ON sp.run_id = r.run_id
              WHERE r.tour = ?
                AND r.event_id IS NOT NULL
                AND r.event_year IS NOT NULL
                AND (? IS NULL OR r.event_year >= ?)
                AND (? IS NULL OR r.event_year <= ?)
            )
            SELECT
              c.win_prob AS win_prob,
              c.top3_prob AS top3_prob,
              c.top5_prob AS top5_prob,
              c.top10_prob AS top10_prob,
              o.won AS won,
              o.top3 AS top3,
              o.top5 AS top5,
              o.top10 AS top10
            FROM candidate c
            JOIN event_outcomes o
              ON o.tour = c.tour
             AND o.event_id = c.event_id
             AND o.event_year = c.event_year
             AND o.player_key = c.player_key
            WHERE c.row_rank = 1
        """

        with self._lock, self._connect() as conn:
            rows = conn.execute(
                sql,
                (
                    tour,
                    min_event_year,
                    min_event_year,
                    max_event_year,
                    max_event_year,
                ),
            ).fetchall()

        return [
            {
                "win_prob": float(row["win_prob"]),
                "top3_prob": float(row["top3_prob"]),
                "top5_prob": float(row["top5_prob"]),
                "top10_prob": float(row["top10_prob"]),
                "won": int(row["won"]),
                "top3": int(row["top3"]),
                "top5": int(row["top5"]),
                "top10": int(row["top10"]),
            }
            for row in rows
        ]


def _extract_event_stats_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        stats = payload.get("event_stats")
        if isinstance(stats, list):
            return [row for row in stats if isinstance(row, dict)]

    rows: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if _string_from_row(node, ("player_name", "name", "player")):
                rows.append(node)
            for value in node.values():
                walk(value)
            return
        if isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    return rows


def _market_keys(market: str) -> tuple[str, str]:
    mapping = {
        "win": ("win_prob", "won"),
        "top_3": ("top3_prob", "top3"),
        "top_5": ("top5_prob", "top5"),
        "top_10": ("top10_prob", "top10"),
    }
    return mapping[market]


def _fit_logistic_parameters(
    probs: np.ndarray,
    labels: np.ndarray,
    ridge: float = 1e-2,
    max_iter: int = 40,
) -> tuple[float, float]:
    clipped = np.clip(probs, _PROB_EPS, 1.0 - _PROB_EPS)
    x = np.log(clipped / (1.0 - clipped))

    alpha = 0.0
    beta = 1.0
    for _ in range(max_iter):
        logits = alpha + (beta * x)
        pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        weights = np.clip(pred * (1.0 - pred), 1e-8, None)

        grad_alpha = float(np.sum(pred - labels) + (ridge * alpha))
        grad_beta = float(np.sum((pred - labels) * x) + (ridge * (beta - 1.0)))
        h_aa = float(np.sum(weights) + ridge)
        h_ab = float(np.sum(weights * x))
        h_bb = float(np.sum(weights * x * x) + ridge)

        det = (h_aa * h_bb) - (h_ab * h_ab)
        if det <= 1e-10:
            break

        step_alpha = ((h_bb * grad_alpha) - (h_ab * grad_beta)) / det
        step_beta = ((-h_ab * grad_alpha) + (h_aa * grad_beta)) / det

        alpha -= step_alpha
        beta -= step_beta
        beta = float(np.clip(beta, 0.05, 4.0))

        if abs(step_alpha) < 1e-6 and abs(step_beta) < 1e-6:
            break

    return float(alpha), float(beta)


def _logistic_calibrate(probs: np.ndarray, *, alpha: float, beta: float) -> np.ndarray:
    clipped = np.clip(probs, _PROB_EPS, 1.0 - _PROB_EPS)
    logits = np.log(clipped / (1.0 - clipped))
    adjusted = alpha + (beta * logits)
    calibrated = 1.0 / (1.0 + np.exp(-np.clip(adjusted, -30.0, 30.0)))
    return np.clip(calibrated, _PROB_EPS, 1.0 - _PROB_EPS)


def _brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64), _PROB_EPS, 1.0 - _PROB_EPS)
    y = np.asarray(labels, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def _logloss(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64), _PROB_EPS, 1.0 - _PROB_EPS)
    y = np.asarray(labels, dtype=np.float64)
    return float(np.mean(-(y * np.log(p)) - ((1.0 - y) * np.log(1.0 - p))))


def _canonical_player_key(player_id: str | None, player_name: str | None) -> str | None:
    normalized_id = _normalize_token(player_id)
    if normalized_id:
        return normalized_id

    normalized_name = _normalize_name(player_name)
    if normalized_name:
        return normalized_name
    return None


def _normalize_name(player_name: str | None) -> str | None:
    if not player_name:
        return None
    cleaned = " ".join(str(player_name).strip().lower().replace(".", "").split())
    if not cleaned:
        return None
    if "," in cleaned:
        pieces = [piece.strip() for piece in cleaned.split(",") if piece.strip()]
        if len(pieces) >= 2:
            reordered = " ".join(pieces[1:] + [pieces[0]])
            cleaned = " ".join(reordered.split())
    return cleaned


def _normalize_token(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = "".join(str(value).strip().lower().split())
    return cleaned or None


def _year_from_date(value: str | None) -> int | None:
    if not value:
        return None
    text = str(value).strip()
    if len(text) >= 4 and text[:4].isdigit():
        year = int(text[:4])
        if 1900 <= year <= 2100:
            return year
    match = re.search(r"(19\d{2}|20\d{2}|21\d{2})", text)
    if match:
        return int(match.group(1))
    return None


def _finish_rank_from_row(row: dict[str, Any]) -> int | None:
    fin_text = _string_from_row(row, ("fin_text", "finish", "position", "pos"))
    if fin_text:
        cleaned = fin_text.strip().upper()
        if cleaned.startswith("T"):
            cleaned = cleaned[1:]
        if cleaned.isdigit():
            return int(cleaned)

    number = _to_float(
        _value_from_row(row, ("finish", "position", "pos", "rank", "result"))
    )
    if number is None:
        return None
    if number < 0:
        return None
    rank = int(round(number))
    return rank if rank > 0 else None


def _string_from_payload(payload: Any, keys: tuple[str, ...]) -> str | None:
    if isinstance(payload, dict):
        direct = _string_from_row(payload, keys)
        if direct:
            return direct
        for value in payload.values():
            nested = _string_from_payload(value, keys)
            if nested:
                return nested
    elif isinstance(payload, list):
        for item in payload:
            nested = _string_from_payload(item, keys)
            if nested:
                return nested
    return None


def _string_from_row(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    value = _value_from_row(row, keys)
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return None
    text = str(value).strip()
    return text or None


def _value_from_row(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    lowered = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        if key in lowered:
            value = lowered[key]
            if value is None:
                continue
            return value
    return None


def _coerce_probability(value: Any) -> float:
    numeric = _to_float(value)
    if numeric is None:
        return 0.0
    if numeric > 1.0:
        numeric /= 100.0
    return float(np.clip(numeric, 0.0, 1.0))


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_snapshot_type(value: Any) -> str:
    if value is None:
        return "manual"
    text = str(value).strip().lower()
    if text in _SNAPSHOT_TYPES:
        return text
    return "manual"


def _normalize_lifecycle_state(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in _LIFECYCLE_STATES:
        return text
    return None


def _reconcile_event_year_mismatches(conn: sqlite3.Connection) -> None:
    # Normalize historical rows that were previously keyed to season year instead
    # of calendar year when date fields clearly indicate the calendar year.
    run_rows = conn.execute(
        """
        SELECT run_id, event_year, event_date
        FROM simulation_runs
        WHERE event_year IS NOT NULL
          AND event_date IS NOT NULL
          AND length(event_date) >= 4
        """
    ).fetchall()
    for row in run_rows:
        event_date_year = _year_from_date(row["event_date"])
        if event_date_year is None:
            continue
        old_year = int(row["event_year"])
        if old_year == event_date_year:
            continue
        conn.execute(
            """
            UPDATE simulation_runs
            SET event_year = ?
            WHERE run_id = ?
            """,
            (int(event_date_year), str(row["run_id"])),
        )

    outcome_rows = conn.execute(
        """
        SELECT DISTINCT tour, event_id, event_year, event_completed
        FROM event_outcomes
        WHERE event_year IS NOT NULL
          AND event_completed IS NOT NULL
          AND length(event_completed) >= 4
        """
    ).fetchall()
    for row in outcome_rows:
        completed_year = _year_from_date(row["event_completed"])
        if completed_year is None:
            continue
        old_year = int(row["event_year"])
        if old_year == completed_year:
            continue
        collision = conn.execute(
            """
            SELECT 1
            FROM event_outcomes
            WHERE tour = ? AND event_id = ? AND event_year = ?
            LIMIT 1
            """,
            (str(row["tour"]), str(row["event_id"]), int(completed_year)),
        ).fetchone()
        if collision is not None:
            continue
        conn.execute(
            """
            UPDATE event_outcomes
            SET event_year = ?
            WHERE tour = ? AND event_id = ? AND event_year = ?
            """,
            (
                int(completed_year),
                str(row["tour"]),
                str(row["event_id"]),
                int(old_year),
            ),
        )

    lifecycle_rows = conn.execute(
        """
        SELECT tour, event_id, event_year, event_date
        FROM event_lifecycle
        WHERE event_year IS NOT NULL
          AND event_date IS NOT NULL
          AND length(event_date) >= 4
        """
    ).fetchall()
    for row in lifecycle_rows:
        event_date_year = _year_from_date(row["event_date"])
        if event_date_year is None:
            continue
        old_year = int(row["event_year"])
        if old_year == event_date_year:
            continue
        collision = conn.execute(
            """
            SELECT 1
            FROM event_lifecycle
            WHERE tour = ? AND event_id = ? AND event_year = ?
            LIMIT 1
            """,
            (str(row["tour"]), str(row["event_id"]), int(event_date_year)),
        ).fetchone()
        if collision is not None:
            continue
        conn.execute(
            """
            UPDATE event_lifecycle
            SET event_year = ?
            WHERE tour = ? AND event_id = ? AND event_year = ?
            """,
            (
                int(event_date_year),
                str(row["tour"]),
                str(row["event_id"]),
                int(old_year),
            ),
        )


def _table_has_column(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    for row in rows:
        if str(row["name"]).strip().lower() == column_name.strip().lower():
            return True
    return False
