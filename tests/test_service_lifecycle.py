from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

from pga_sim.datagolf_client import DataGolfAPIError
from pga_sim.learning import LearningStore
from pga_sim.models import SimulationRequest
from pga_sim.service import SimulationService


def _players() -> list[dict[str, str]]:
    return [
        {"player_id": str(9100 + idx), "player_name": f"Lifecycle Player {idx + 1}"}
        for idx in range(8)
    ]


class _ScheduledLifecycleClient:
    async def get_field_updates(self, tour: str = "pga", event_id: Optional[str] = None):
        return {
            "event_id": "200",
            "event_name": "Lifecycle Cup",
            "date": "2026-02-27",
            "field": [
                {
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "position": "",
                    "score_to_par": None,
                    "thru": 0,
                    "today": None,
                }
                for player in _players()
            ],
        }

    async def get_pre_tournament(
        self,
        tour: str = "pga",
        event_id: Optional[str] = None,
        add_position: int = 3,
        odds_format: str = "percent",
    ):
        rows = []
        for idx, player in enumerate(_players(), start=1):
            rows.append(
                {
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "win": max(0.01, 0.22 - (0.015 * idx)),
                    "top_3": max(0.03, 0.42 - (0.02 * idx)),
                    "top_5": max(0.06, 0.58 - (0.03 * idx)),
                    "top_10": max(0.12, 0.80 - (0.05 * idx)),
                }
            )
        return {"predictions": rows}

    async def get_player_decompositions(self, tour: str = "pga", event_id: Optional[str] = None):
        rows = []
        for idx, player in enumerate(_players(), start=1):
            rows.append(
                {
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "sg_total": 1.1 - (0.15 * idx),
                    "sigma": 2.6,
                }
            )
        return {"decomposition": rows}

    async def get_in_play(
        self,
        tour: str = "pga",
        dead_heat: str = "no",
        odds_format: str = "percent",
    ):
        return {}

    async def get_historical_event(self, tour: str, event_id: str, year: int):
        raise DataGolfAPIError("historical endpoint not expected for scheduled test")


class _CompletedLifecycleClient(_ScheduledLifecycleClient):
    async def get_field_updates(self, tour: str = "pga", event_id: Optional[str] = None):
        rows = []
        for idx, player in enumerate(_players(), start=1):
            rows.append(
                {
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "position": str(idx),
                    "score_to_par": float(-16 + idx),
                    "thru": "18",
                    "today": 0,
                    "round_scores": [70, 69, 68, 67 + idx],
                }
            )
        return {
            "event_id": "201",
            "event_name": "Completed Lifecycle Open",
            "date": "2026-02-22",
            "field": rows,
        }

    async def get_historical_event(self, tour: str, event_id: str, year: int):
        event_stats = []
        for rank, player in enumerate(_players(), start=1):
            event_stats.append(
                {
                    "dg_id": int(player["player_id"]),
                    "player_name": player["player_name"],
                    "fin_text": str(rank),
                    "dg_points": float(max(1, 40 - rank)),
                }
            )
        return {
            "event_completed": "2026-02-22",
            "event_id": event_id,
            "event_name": "Completed Lifecycle Open",
            "year": year,
            "event_stats": event_stats,
        }


class _BackfillLifecycleClient(_ScheduledLifecycleClient):
    async def get_field_updates(self, tour: str = "pga", event_id: Optional[str] = None):
        # Active future event (scheduled) so backfill should only target historical events.
        return {
            "event_id": "400",
            "event_name": "Future Automation Classic",
            "date": "2026-03-10",
            "field": [
                {
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "position": "",
                    "score_to_par": None,
                    "thru": 0,
                }
                for player in _players()
            ],
        }

    async def get_historical_event_list(self, tour: str = "pga"):
        return [
            {
                "calendar_year": 2026,
                "date": "2026-01-10",
                "event_id": "301",
                "event_name": "Backfill Event One",
                "tour": "pga",
            },
            {
                "calendar_year": 2026,
                "date": "2026-01-20",
                "event_id": "302",
                "event_name": "Backfill Event Two",
                "tour": "pga",
            },
        ]

    async def get_historical_event(self, tour: str, event_id: str, year: int):
        base_players = _players()
        event_stats = []
        for rank, player in enumerate(base_players, start=1):
            event_stats.append(
                {
                    "dg_id": int(player["player_id"]),
                    "player_name": player["player_name"],
                    "fin_text": str(rank),
                    "dg_points": float(max(1, 35 - rank)),
                }
            )
        return {
            "event_completed": "2026-01-21",
            "event_id": event_id,
            "event_name": "Backfill Event",
            "year": year,
            "event_stats": event_stats,
        }


class _BackfillPreviousYearClient(_BackfillLifecycleClient):
    async def get_historical_event_list(self, tour: str = "pga"):
        return [
            {
                "calendar_year": 2025,
                "date": "2025-01-10",
                "event_id": "501",
                "event_name": "Backfill Prior Year One",
                "tour": "pga",
            },
            {
                "calendar_year": 2025,
                "date": "2025-01-20",
                "event_id": "502",
                "event_name": "Backfill Prior Year Two",
                "tour": "pga",
            },
        ]

    async def get_historical_event(self, tour: str, event_id: str, year: int):
        payload = await super().get_historical_event(tour=tour, event_id=event_id, year=year)
        payload["event_completed"] = "2025-01-21"
        return payload


class _SchedulePlaceholderLifecycleClient(_ScheduledLifecycleClient):
    async def get_schedule(
        self,
        tour: str = "pga",
        upcoming_only: str = "yes",
        season: int | None = None,
    ):
        return [
            {
                "calendar_year": 2026,
                "date": "2026-02-16",
                "event_id": "7",
                "event_name": "The Genesis Invitational",
                "tour": "pga",
            },
            {
                "calendar_year": 2026,
                "date": "2026-02-27",
                "event_id": "200",
                "event_name": "Lifecycle Cup",
                "tour": "pga",
            },
        ]


class _NoActiveFieldLifecycleClient(_ScheduledLifecycleClient):
    async def get_field_updates(self, tour: str = "pga", event_id: Optional[str] = None):
        raise DataGolfAPIError("DataGolf API error for field-updates: no alt event this week.")

    async def get_schedule(
        self,
        tour: str = "pga",
        upcoming_only: str = "yes",
        season: int | None = None,
    ):
        return [
            {
                "calendar_year": 2026,
                "date": "2026-01-20",
                "event_id": "719",
                "event_name": "Hong Kong",
                "tour": "liv",
            },
            {
                "calendar_year": 2026,
                "date": "2026-03-12",
                "event_id": "713",
                "event_name": "Singapore",
                "tour": "liv",
            },
        ]


def test_lifecycle_cycle_captures_pre_event_snapshot_once(tmp_path) -> None:
    learning_store = LearningStore(str(tmp_path / "lifecycle_pre_event.sqlite3"))
    service = SimulationService(
        _ScheduledLifecycleClient(),
        learning_store=learning_store,
        lifecycle_pre_event_simulations=5000,
        lifecycle_pre_event_seed=42,
        lifecycle_sync_max_events=5,
        lifecycle_target_year=2026,
    )

    status_first = asyncio.run(service.run_lifecycle_cycle("pga"))
    assert status_first.pre_event_snapshot_ready is True
    assert status_first.pre_event_snapshot_version == 1
    assert status_first.active_event_state == "pre_event_snapshot_taken"

    learning_first = asyncio.run(service.get_learning_status("pga"))
    assert learning_first.predictions_logged == 1

    status_second = asyncio.run(service.run_lifecycle_cycle("pga"))
    assert status_second.pre_event_snapshot_ready is True
    assert status_second.pre_event_snapshot_version == 1

    learning_second = asyncio.run(service.get_learning_status("pga"))
    assert learning_second.predictions_logged == 1


def test_lifecycle_pre_event_snapshot_handles_large_simulation_caps(tmp_path) -> None:
    learning_store = LearningStore(str(tmp_path / "lifecycle_pre_event_large.sqlite3"))
    service = SimulationService(
        _ScheduledLifecycleClient(),
        learning_store=learning_store,
        lifecycle_pre_event_simulations=1_000_000,
        lifecycle_pre_event_seed=42,
        lifecycle_sync_max_events=5,
        lifecycle_target_year=2026,
    )

    status = asyncio.run(service.run_lifecycle_cycle("pga"))
    assert status.pre_event_snapshot_ready is True
    assert status.pre_event_snapshot_version == 1
    assert status.last_run_note is not None
    assert "validation error" not in status.last_run_note.lower()


def test_lifecycle_cycle_syncs_completed_event_and_retrains(tmp_path) -> None:
    learning_store = LearningStore(str(tmp_path / "lifecycle_complete.sqlite3"))
    service = SimulationService(
        _CompletedLifecycleClient(),
        learning_store=learning_store,
        lifecycle_pre_event_simulations=5000,
        lifecycle_pre_event_seed=99,
        lifecycle_sync_max_events=10,
        lifecycle_target_year=2026,
    )

    # First log a prediction snapshot for the completed event.
    result = asyncio.run(
        service.simulate(
            SimulationRequest(
                tour="pga",
                simulations=5000,
                seed=7,
                enable_in_play_conditioning=False,
                enable_seasonal_form=False,
            )
        )
    )
    assert result.event_id == "201"

    lifecycle_status = asyncio.run(service.run_lifecycle_cycle("pga"))
    assert lifecycle_status.pending_events == 0

    learning_status = asyncio.run(service.get_learning_status("pga"))
    assert learning_status.resolved_events == 1
    assert learning_status.calibration_version == 1

    rows = learning_store.list_event_lifecycle(tour="pga", max_events=5)
    assert len(rows) >= 1
    completed = next(row for row in rows if row["event_id"] == "201")
    assert completed["state"] == "retrained"
    assert completed["outcomes_source"] == "official"
    assert completed["retrain_version"] == 1


def test_lifecycle_cycle_runs_backfill_batch_and_retrains(tmp_path) -> None:
    learning_store = LearningStore(str(tmp_path / "lifecycle_backfill.sqlite3"))
    service = SimulationService(
        _BackfillLifecycleClient(),
        learning_store=learning_store,
        lifecycle_pre_event_simulations=5000,
        lifecycle_pre_event_seed=15,
        lifecycle_sync_max_events=10,
        lifecycle_backfill_enabled=True,
        lifecycle_backfill_batch_size=2,
        lifecycle_target_year=2026,
    )

    lifecycle_status = asyncio.run(service.run_lifecycle_cycle("pga"))
    assert lifecycle_status.last_run_note is not None
    assert "Backfill:" in lifecycle_status.last_run_note

    learning_status = asyncio.run(service.get_learning_status("pga"))
    assert learning_status.resolved_events >= 2
    assert learning_status.calibration_version >= 1

    rows = learning_store.list_event_lifecycle(tour="pga", max_events=10)
    event_ids = {row["event_id"] for row in rows}
    assert "301" in event_ids
    assert "302" in event_ids
    backfill_rows = [row for row in rows if row["event_id"] in {"301", "302"}]
    assert all(row["pre_event_snapshot_version"] is not None for row in backfill_rows)


def test_lifecycle_backfill_respects_target_year_scope(tmp_path) -> None:
    learning_store = LearningStore(str(tmp_path / "lifecycle_backfill_scope.sqlite3"))
    service = SimulationService(
        _BackfillPreviousYearClient(),
        learning_store=learning_store,
        lifecycle_pre_event_simulations=5000,
        lifecycle_pre_event_seed=15,
        lifecycle_sync_max_events=10,
        lifecycle_backfill_enabled=True,
        lifecycle_backfill_batch_size=2,
        lifecycle_target_year=2026,
    )

    lifecycle_status = asyncio.run(service.run_lifecycle_cycle("pga"))
    assert lifecycle_status.last_run_note is not None
    assert "for 2026" in lifecycle_status.last_run_note

    learning_status = asyncio.run(service.get_learning_status("pga"))
    assert learning_status.resolved_events == 0
    assert learning_status.calibration_version == 0

    rows = learning_store.list_event_lifecycle(
        tour="pga",
        max_events=20,
        min_event_year=2026,
        max_event_year=2026,
    )
    row_ids = {row["event_id"] for row in rows}
    assert "501" not in row_ids
    assert "502" not in row_ids


def test_lifecycle_status_includes_completed_schedule_placeholder(tmp_path) -> None:
    learning_store = LearningStore(str(tmp_path / "lifecycle_schedule_placeholder.sqlite3"))
    service = SimulationService(
        _SchedulePlaceholderLifecycleClient(),
        learning_store=learning_store,
        lifecycle_pre_event_simulations=5000,
        lifecycle_pre_event_seed=42,
        lifecycle_sync_max_events=5,
        lifecycle_target_year=2026,
    )

    status = asyncio.run(service.get_lifecycle_status("pga"))
    schedule_rows = [
        row
        for row in status.recent_events
        if row.event_id == "7" and int(row.event_year) == 2026
    ]
    assert len(schedule_rows) == 1
    assert schedule_rows[0].state == "complete"


def test_lifecycle_status_handles_missing_active_field_feed_without_error(tmp_path) -> None:
    learning_store = LearningStore(str(tmp_path / "lifecycle_no_active_field.sqlite3"))
    service = SimulationService(
        _NoActiveFieldLifecycleClient(),
        learning_store=learning_store,
        lifecycle_pre_event_simulations=5000,
        lifecycle_pre_event_seed=42,
        lifecycle_sync_max_events=5,
        lifecycle_target_year=2026,
    )

    status = asyncio.run(service.get_lifecycle_status("liv"))
    assert status.tour == "liv"
    assert status.active_event_id is None
    row_ids = {row.event_id for row in status.recent_events}
    assert "719" in row_ids


def test_lifecycle_last_run_note_scoped_per_tour(tmp_path) -> None:
    learning_store = LearningStore(str(tmp_path / "lifecycle_last_run_scope.sqlite3"))
    service = SimulationService(
        _ScheduledLifecycleClient(),
        learning_store=learning_store,
        lifecycle_pre_event_simulations=5000,
        lifecycle_pre_event_seed=42,
        lifecycle_sync_max_events=5,
        lifecycle_target_year=2026,
    )

    pga_status = asyncio.run(service.run_lifecycle_cycle("pga"))
    assert pga_status.last_run_note is not None
    assert pga_status.last_run_at is not None

    liv_status = asyncio.run(service.get_lifecycle_status("liv"))
    assert liv_status.last_run_note is None
    assert liv_status.last_run_at is None


def test_lifecycle_cycle_forces_retrain_when_calibration_health_degrades(tmp_path) -> None:
    learning_store = LearningStore(str(tmp_path / "lifecycle_health_retrain.sqlite3"))

    seed_players = []
    for idx, player in enumerate(_players(), start=1):
        seed_players.append(
            {
                "player_id": player["player_id"],
                "player_name": player["player_name"],
                "win_probability": max(0.01, 0.22 - (0.015 * idx)),
                "top_3_probability": max(0.03, 0.42 - (0.02 * idx)),
                "top_5_probability": max(0.06, 0.58 - (0.03 * idx)),
                "top_10_probability": max(0.12, 0.80 - (0.05 * idx)),
            }
        )

    learning_store.record_prediction(
        tour="pga",
        event_id="5",
        event_name="AT&T Pebble Beach Pro-Am",
        event_date="2026-02-15",
        requested_simulations=100000,
        simulations=100000,
        enable_in_play=True,
        in_play_applied=False,
        players=seed_players,
    )
    learning_store.record_outcome_payload(
        tour="pga",
        event_id="5",
        event_year=2026,
        payload={
            "event_name": "AT&T Pebble Beach Pro-Am",
            "event_completed": "2026-02-15",
            "event_stats": [
                {
                    "dg_id": int(player["player_id"]),
                    "player_name": player["player_name"],
                    "fin_text": str(idx + 1),
                    "dg_points": float(max(1, 40 - idx)),
                }
                for idx, player in enumerate(_players())
            ],
        },
    )
    learning_store.retrain(tour="pga", bump_version=True, min_event_year=2026, max_event_year=2026)

    now_text = datetime.now(timezone.utc).isoformat()
    with learning_store._lock, learning_store._connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO calibration_state (
              tour, market, alpha, beta, samples, positives,
              brier_before, brier_after, logloss_before, logloss_after, updated_at, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "pga",
                "win",
                0.0,
                1.0,
                8,
                1,
                0.0134,
                0.9914,
                0.05,
                2.5,
                now_text,
                9,
            ),
        )
        conn.commit()

    service = SimulationService(
        _ScheduledLifecycleClient(),
        learning_store=learning_store,
        lifecycle_pre_event_simulations=5000,
        lifecycle_pre_event_seed=42,
        lifecycle_sync_max_events=5,
        lifecycle_target_year=2026,
    )

    status = asyncio.run(service.run_lifecycle_cycle("pga"))
    assert status.last_run_note is not None
    assert "Calibration health check triggered retrain." in status.last_run_note
    assert "retrain=yes" in status.last_run_note

    learning_status = asyncio.run(service.get_learning_status("pga"))
    assert learning_status.calibration_version >= 10
