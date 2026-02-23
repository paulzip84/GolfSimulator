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
        {"player_id": str(3100 + idx), "player_name": f"Player {idx + 1}"}
        for idx in range(8)
    ]


class _LearningClient:
    async def get_field_updates(self, tour: str = "pga", event_id: Optional[str] = None):
        field_rows = []
        for idx, player in enumerate(_players(), start=1):
            field_rows.append(
                {
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "position": f"T{idx}",
                    "score_to_par": float(idx - 2),
                    "thru": 0,
                }
            )
        return {
            "event_id": "14",
            "event_name": "Masters Tournament",
            "date": "2025-04-10",
            "field": field_rows,
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
                    "win": max(0.01, 0.18 - (0.015 * idx)),
                    "top_3": max(0.04, 0.36 - (0.02 * idx)),
                    "top_5": max(0.08, 0.52 - (0.03 * idx)),
                    "top_10": max(0.15, 0.78 - (0.05 * idx)),
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
                    "sg_total": 1.2 - (0.22 * idx),
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
            "event_completed": "2025-04-13",
            "event_id": event_id,
            "event_name": "Masters Tournament",
            "year": year,
            "event_stats": event_stats,
        }


class _HistoricalUnavailableLearningClient(_LearningClient):
    async def get_field_updates(self, tour: str = "pga", event_id: Optional[str] = None):
        field_rows = []
        for idx, player in enumerate(_players(), start=1):
            field_rows.append(
                {
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "position": str(idx),
                    "score_to_par": float(-14 + idx),
                    # Simulate stale thru values despite complete round scorecards.
                    "thru": "15",
                    "today": 0,
                    "round_scores": [70, 69, 68, 67 + idx],
                }
            )
        return {
            "event_id": "99",
            "event_name": "Fallback Open",
            "date": "2026-02-22",
            "field": field_rows,
        }

    async def get_in_play(
        self,
        tour: str = "pga",
        dead_heat: str = "no",
        odds_format: str = "percent",
    ):
        # Stale live row should not override a completed field-updates row.
        return {
            "preds": [
                {
                    "player_id": _players()[0]["player_id"],
                    "player_name": _players()[0]["player_name"],
                    "position": "1",
                    "score_to_par": "-12",
                    "thru": "15",
                    "today": "-3",
                }
            ]
        }

    async def get_historical_event(self, tour: str, event_id: str, year: int):
        raise DataGolfAPIError("historical endpoint not published yet")


class _PreTournamentUnavailableLearningClient(_LearningClient):
    async def get_pre_tournament(
        self,
        tour: str = "pga",
        event_id: Optional[str] = None,
        add_position: int = 3,
        odds_format: str = "percent",
    ):
        raise DataGolfAPIError("pre-tournament feed temporarily unavailable")


class _FieldUnavailableLearningClient(_LearningClient):
    async def get_field_updates(self, tour: str = "pga", event_id: Optional[str] = None):
        raise DataGolfAPIError("field-updates feed temporarily unavailable")


class _ScheduledWithStaleLiveLearningClient(_LearningClient):
    async def get_field_updates(self, tour: str = "pga", event_id: Optional[str] = None):
        return {
            "event_id": "14",
            "event_name": "Masters Tournament",
            "date": "2025-04-10",
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

    async def get_in_play(
        self,
        tour: str = "pga",
        dead_heat: str = "no",
        odds_format: str = "percent",
    ):
        return {
            "preds": [
                {
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "position": str(idx + 1),
                    "score_to_par": float(-15 + idx),
                    "thru": "18",
                    "today": 0,
                    "round_scores": [71, 68, 67, 66 + idx],
                }
                for idx, player in enumerate(_players())
            ]
        }


def test_service_logs_predictions_and_retrains_learning(tmp_path) -> None:
    learning_store = LearningStore(str(tmp_path / "service_learning.sqlite3"))
    service = SimulationService(
        _LearningClient(),
        learning_store=learning_store,
        lifecycle_target_year=2025,
    )

    result = asyncio.run(
        service.simulate(
            SimulationRequest(
                tour="pga",
                simulations=4000,
                seed=11,
                enable_in_play_conditioning=False,
                enable_seasonal_form=False,
            )
        )
    )
    assert result.event_id == "14"
    assert result.simulation_version == 1
    assert result.calibration_applied is False

    before = asyncio.run(service.get_learning_status(tour="pga"))
    assert before.predictions_logged == 1
    assert before.pending_events == 1
    assert before.calibration_version == 0

    sync = asyncio.run(service.sync_learning_and_retrain(tour="pga", max_events=10))
    assert sync.events_processed == 1
    assert sync.outcomes_fetched == 1
    assert sync.provisional_outcomes_fetched == 0
    assert sync.calibration_version == 1
    assert sync.retrain_executed is True
    assert sync.awaiting_outcomes_count == 0

    after = asyncio.run(service.get_learning_status(tour="pga"))
    assert after.pending_events == 0
    assert after.resolved_events == 1
    assert after.resolved_predictions == 8

    sync_again = asyncio.run(service.sync_learning_and_retrain(tour="pga", max_events=10))
    assert sync_again.events_processed == 0
    assert sync_again.outcomes_fetched == 0
    assert sync_again.retrain_executed is False
    assert sync_again.calibration_version == 1

    trends = asyncio.run(
        service.get_learning_event_trends(
            tour="pga",
            event_id="14",
            event_year=2025,
            max_snapshots=20,
            max_players=10,
        )
    )
    assert trends.snapshot_count >= 1
    assert trends.event_id == "14"
    assert len(trends.players) > 0


def test_service_sync_uses_provisional_outcomes_when_official_feed_lags(tmp_path) -> None:
    learning_store = LearningStore(str(tmp_path / "service_learning_fallback.sqlite3"))
    service = SimulationService(
        _HistoricalUnavailableLearningClient(),
        learning_store=learning_store,
        lifecycle_target_year=2026,
    )

    result = asyncio.run(
        service.simulate(
            SimulationRequest(
                tour="pga",
                simulations=4000,
                seed=12,
                enable_in_play_conditioning=False,
                enable_seasonal_form=False,
            )
        )
    )
    assert result.event_id == "99"
    assert result.simulation_version == 1
    assert result.players[0].current_thru == "F"

    sync = asyncio.run(service.sync_learning_and_retrain(tour="pga", max_events=10))
    assert sync.events_processed == 1
    assert sync.outcomes_fetched == 1
    assert sync.provisional_outcomes_fetched == 1
    assert sync.awaiting_outcomes_count == 0
    assert sync.retrain_executed is True
    assert sync.calibration_version == 1
    assert sync.provisional_event_ids == ["99:2026"]


def test_service_simulation_falls_back_when_pre_tournament_feed_errors(tmp_path) -> None:
    learning_store = LearningStore(str(tmp_path / "service_learning_pre_fallback.sqlite3"))
    service = SimulationService(
        _PreTournamentUnavailableLearningClient(),
        learning_store=learning_store,
        lifecycle_target_year=2025,
    )

    result = asyncio.run(
        service.simulate(
            SimulationRequest(
                tour="pga",
                simulations=4000,
                seed=13,
                enable_in_play_conditioning=False,
                enable_seasonal_form=False,
            )
        )
    )
    assert result.event_id == "14"
    assert len(result.players) >= 8
    assert result.in_play_conditioning_note is not None
    assert "Data feed fallback:" in result.in_play_conditioning_note


def test_service_simulation_falls_back_when_field_feed_errors(tmp_path) -> None:
    learning_store = LearningStore(str(tmp_path / "service_learning_field_fallback.sqlite3"))
    service = SimulationService(
        _FieldUnavailableLearningClient(),
        learning_store=learning_store,
        lifecycle_target_year=2025,
    )

    result = asyncio.run(
        service.simulate(
            SimulationRequest(
                tour="pga",
                simulations=4000,
                seed=15,
                enable_in_play_conditioning=False,
                enable_seasonal_form=False,
            )
        )
    )
    assert len(result.players) >= 8
    assert result.in_play_conditioning_note is not None
    assert "field updates unavailable" in result.in_play_conditioning_note.lower()


def test_service_ignores_stale_live_feed_when_event_is_scheduled(tmp_path) -> None:
    learning_store = LearningStore(str(tmp_path / "service_learning_stale_live.sqlite3"))
    service = SimulationService(
        _ScheduledWithStaleLiveLearningClient(),
        learning_store=learning_store,
        lifecycle_target_year=2025,
    )

    result = asyncio.run(
        service.simulate(
            SimulationRequest(
                tour="pga",
                simulations=4000,
                seed=14,
                enable_in_play_conditioning=True,
                enable_seasonal_form=False,
            )
        )
    )
    assert result.in_play_conditioning_applied is False
    assert result.in_play_conditioning_note is not None
    assert "has not started" in result.in_play_conditioning_note.lower()


def test_service_skips_harmful_calibration_snapshot(tmp_path) -> None:
    learning_store = LearningStore(str(tmp_path / "service_learning_harmful_calibration.sqlite3"))

    players = [
        {
            "player_id": player["player_id"],
            "player_name": player["player_name"],
            "win_probability": 0.04 * (idx + 1),
            "top_3_probability": min(0.95, 0.08 * (idx + 1)),
            "top_5_probability": min(0.98, 0.11 * (idx + 1)),
            "top_10_probability": min(0.99, 0.14 * (idx + 1)),
        }
        for idx, player in enumerate(_players())
    ]
    learning_store.record_prediction(
        tour="pga",
        event_id="14",
        event_name="Masters Tournament",
        event_date="2025-04-10",
        requested_simulations=10000,
        simulations=10000,
        enable_in_play=True,
        in_play_applied=False,
        players=players,
    )
    learning_store.record_outcome_payload(
        tour="pga",
        event_id="14",
        event_year=2025,
        payload={
            "event_name": "Masters Tournament",
            "event_completed": "2025-04-13",
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
        _LearningClient(),
        learning_store=learning_store,
        lifecycle_target_year=2025,
    )

    result = asyncio.run(
        service.simulate(
            SimulationRequest(
                tour="pga",
                simulations=4000,
                seed=15,
                enable_in_play_conditioning=False,
                enable_seasonal_form=False,
            )
        )
    )
    assert result.calibration_applied is False
    assert result.calibration_note is not None
    assert "Skipped learning calibration v9" in result.calibration_note


def test_service_sync_retrains_when_existing_calibration_is_harmful(tmp_path) -> None:
    learning_store = LearningStore(str(tmp_path / "service_learning_harmful_sync.sqlite3"))

    players = [
        {
            "player_id": player["player_id"],
            "player_name": player["player_name"],
            "win_probability": 0.04 * (idx + 1),
            "top_3_probability": min(0.95, 0.08 * (idx + 1)),
            "top_5_probability": min(0.98, 0.11 * (idx + 1)),
            "top_10_probability": min(0.99, 0.14 * (idx + 1)),
        }
        for idx, player in enumerate(_players())
    ]
    learning_store.record_prediction(
        tour="pga",
        event_id="14",
        event_name="Masters Tournament",
        event_date="2025-04-10",
        requested_simulations=10000,
        simulations=10000,
        enable_in_play=True,
        in_play_applied=False,
        players=players,
    )
    learning_store.record_outcome_payload(
        tour="pga",
        event_id="14",
        event_year=2025,
        payload={
            "event_name": "Masters Tournament",
            "event_completed": "2025-04-13",
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
        _LearningClient(),
        learning_store=learning_store,
        lifecycle_target_year=2025,
    )
    sync = asyncio.run(service.sync_learning_and_retrain(tour="pga", max_events=10))
    assert sync.events_processed == 0
    assert sync.outcomes_fetched == 0
    assert sync.retrain_executed is True
    assert sync.calibration_version >= 10
