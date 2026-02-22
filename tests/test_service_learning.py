from __future__ import annotations

import asyncio
from typing import Optional

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


def test_service_logs_predictions_and_retrains_learning(tmp_path) -> None:
    learning_store = LearningStore(str(tmp_path / "service_learning.sqlite3"))
    service = SimulationService(_LearningClient(), learning_store=learning_store)

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
    assert result.calibration_applied is False

    before = asyncio.run(service.get_learning_status(tour="pga"))
    assert before.predictions_logged == 1
    assert before.pending_events == 1
    assert before.calibration_version == 0

    sync = asyncio.run(service.sync_learning_and_retrain(tour="pga", max_events=10))
    assert sync.events_processed == 1
    assert sync.outcomes_fetched == 1
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
