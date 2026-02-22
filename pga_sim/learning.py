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

                CREATE INDEX IF NOT EXISTS idx_simulation_runs_event
                  ON simulation_runs (tour, event_id, event_year, created_at);
                CREATE INDEX IF NOT EXISTS idx_outcomes_event
                  ON event_outcomes (tour, event_id, event_year);
                CREATE INDEX IF NOT EXISTS idx_simulation_players_key
                  ON simulation_players (player_key);
                """
            )
            conn.commit()

    @staticmethod
    def _normalize_tour(tour: str) -> str:
        return (tour or "pga").strip().lower()

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
        players: list[dict[str, Any]],
    ) -> str:
        if not players:
            return ""

        normalized_tour = self._normalize_tour(tour)
        normalized_event_id = _normalize_token(event_id)
        event_year = _year_from_date(event_date)
        run_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO simulation_runs (
                  run_id, created_at, tour, event_id, event_name, event_year, event_date,
                  requested_simulations, simulations, enable_in_play, in_play_applied
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    created_at,
                    normalized_tour,
                    normalized_event_id,
                    event_name,
                    event_year,
                    event_date,
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
            conn.commit()

        return run_id

    def list_pending_events(
        self,
        *,
        tour: str | None = None,
        max_events: int = 40,
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
              AND (r.event_date IS NULL OR substr(r.event_date, 1, 10) <= ?)
              AND r.in_play_applied = 0
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

    def retrain(self, *, tour: str, bump_version: bool = True) -> LearningSnapshot:
        normalized_tour = self._normalize_tour(tour)
        current_snapshot = self.get_snapshot(tour=normalized_tour)
        observations = self._load_training_rows(tour=normalized_tour)
        if not observations:
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

    def status(self, *, tour: str) -> dict[str, Any]:
        normalized_tour = self._normalize_tour(tour)
        with self._lock, self._connect() as conn:
            predictions_logged = int(
                conn.execute(
                    "SELECT COUNT(*) FROM simulation_runs WHERE tour = ?",
                    (normalized_tour,),
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
                    )
                    """,
                    (normalized_tour,),
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
                        AND (r.event_date IS NULL OR substr(r.event_date, 1, 10) <= ?)
                        AND r.in_play_applied = 0
                        AND NOT EXISTS (
                          SELECT 1
                          FROM event_outcomes o
                          WHERE o.tour = r.tour
                            AND o.event_id = r.event_id
                            AND o.event_year = r.event_year
                        )
                    )
                    """,
                    (normalized_tour, datetime.now(timezone.utc).date().isoformat()),
                ).fetchone()[0]
            )

        training_rows = self._load_training_rows(tour=normalized_tour)
        snapshot = self.get_snapshot(tour=normalized_tour)
        markets = [snapshot.markets.get(market, CalibrationMetrics(market=market)) for market in _MARKETS]

        return {
            "tour": normalized_tour,
            "predictions_logged": predictions_logged,
            "resolved_predictions": len(training_rows),
            "resolved_events": resolved_events,
            "pending_events": pending_events,
            "calibration_version": snapshot.version,
            "calibration_updated_at": snapshot.updated_at,
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
                        "snapshots": [],
                        "players": [],
                    }
                selected_year = int(year_row["event_year"])

            snapshot_rows = conn.execute(
                """
                SELECT run_id, created_at, simulations, in_play_applied, event_name
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
                  r.created_at AS created_at
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
            "snapshots": list(snapshot_by_run.values()),
            "players": players_payload,
        }

    def _load_training_rows(self, *, tour: str) -> list[dict[str, Any]]:
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
                  ORDER BY r.created_at DESC, r.run_id DESC
                ) AS row_rank
              FROM simulation_runs r
              JOIN simulation_players sp
                ON sp.run_id = r.run_id
              WHERE r.tour = ?
                AND r.in_play_applied = 0
                AND r.event_id IS NOT NULL
                AND r.event_year IS NOT NULL
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
            rows = conn.execute(sql, (tour,)).fetchall()

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
