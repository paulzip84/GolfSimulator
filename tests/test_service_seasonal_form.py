import asyncio
from typing import Optional

from pga_sim.datagolf_client import DataGolfAPIError
from pga_sim.models import SimulationRequest
from pga_sim.service import SimulationService


def _players():
    return [
        {"player_id": "1001", "player_name": "Player 1", "historical_name": "One, Player"},
        {"player_id": "1002", "player_name": "Player 2", "historical_name": "Two, Player"},
        {"player_id": "1003", "player_name": "Player 3", "historical_name": "Three, Player"},
        {"player_id": "1004", "player_name": "Player 4", "historical_name": "Four, Player"},
        {"player_id": "1005", "player_name": "Player 5", "historical_name": "Five, Player"},
        {"player_id": "1006", "player_name": "Player 6", "historical_name": "Six, Player"},
        {"player_id": "1007", "player_name": "Player 7", "historical_name": "Seven, Player"},
        {"player_id": "1008", "player_name": "Player 8", "historical_name": "Eight, Player"},
    ]


class _SeasonalClient:
    async def get_field_updates(self, tour: str = "pga", event_id: Optional[str] = None):
        field_rows = []
        for idx, row in enumerate(_players(), start=1):
            field_rows.append(
                {
                    "player_id": row["player_id"],
                    "player_name": row["player_name"],
                    "position": f"T{idx}",
                    "score_to_par": -float(idx % 3),
                    "thru": 18,
                    "today": -1 if idx % 2 == 0 else 0,
                    "round_scores": [70 + (idx % 2), 69],
                    "hole_scores": [4, 4, 3, 5, 4, 4, 3, 4, 5],
                }
            )
        return {"event_id": "ACTIVE123", "event_name": "Active Event", "field": field_rows}

    async def get_pre_tournament(
        self,
        tour: str = "pga",
        event_id: Optional[str] = None,
        add_position: int = 3,
        odds_format: str = "percent",
    ):
        rows = []
        for row in _players():
            rows.append(
                {
                    "player_id": row["player_id"],
                    "player_name": row["player_name"],
                    "win": 0.02,
                    "top_10": 0.12,
                }
            )
        return {"predictions": rows}

    async def get_player_decompositions(self, tour: str = "pga", event_id: Optional[str] = None):
        rows = []
        for row in _players():
            rows.append(
                {
                    "player_id": row["player_id"],
                    "player_name": row["player_name"],
                    "sg_total": 0.0,
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
        rows = []
        for idx, row in enumerate(_players(), start=1):
            rows.append(
                {
                    "player_id": row["player_id"],
                    "player_name": row["player_name"],
                    "position": f"T{idx + 1}",
                    "score_to_par": -float((idx % 3) + 1),
                    "thru": "F",
                    "today": 1 if idx % 2 == 0 else -1,
                }
            )
        return {"preds": rows}

    async def get_historical_event_list(self, tour: str = "pga"):
        return [
            {
                "calendar_year": 2025,
                "date": "2025-04-13",
                "event_id": 14,
                "event_name": "Masters Tournament",
                "tour": "pga",
            },
            {
                "calendar_year": 2026,
                "date": "2026-01-25",
                "event_id": 2,
                "event_name": "The American Express",
                "tour": "pga",
            },
        ]

    async def get_historical_event(self, tour: str, event_id: str, year: int):
        event_stats = []
        for row in _players():
            if year == 2025:
                dg_points = 5.0
            elif year == 2026:
                if row["player_id"] == "1001":
                    dg_points = 18.0
                elif row["player_id"] == "1002":
                    dg_points = 1.5
                else:
                    dg_points = 5.0
            else:
                dg_points = 5.0

            event_stats.append(
                {
                    "dg_id": int(row["player_id"]),
                    "player_name": row["historical_name"],
                    "dg_points": dg_points,
                    "earnings": int(max(50_000, dg_points * 100_000)),
                    "fec_points": int(max(10, dg_points * 25)),
                    "fin_text": "1" if dg_points > 10 else "T30",
                }
            )
        return {
            "event_completed": f"{year}-04-13",
            "tour": tour,
            "year": year,
            "event_id": str(event_id),
            "event_name": "Sample Event",
            "event_stats": event_stats,
        }


class _BrokenSeasonalClient(_SeasonalClient):
    async def get_historical_event_list(self, tour: str = "pga"):
        raise DataGolfAPIError("historical feed unavailable")


def test_seasonal_form_changes_player_ordering() -> None:
    service = SimulationService(_SeasonalClient())
    result = asyncio.run(
        service.simulate(
            SimulationRequest(
                tour="pga",
                simulations=12_000,
                seed=42,
                enable_seasonal_form=True,
                baseline_season=2025,
                current_season=2026,
                seasonal_form_weight=1.0,
                current_season_weight=1.0,
                form_delta_weight=0.0,
            )
        )
    )

    by_name = {player.player_name: player for player in result.players}
    assert result.form_adjustment_applied is True
    assert by_name["Player 1"].win_probability > by_name["Player 2"].win_probability
    assert by_name["Player 1"].form_delta_metric > 0.0
    assert by_name["Player 2"].form_delta_metric < 0.0
    assert by_name["Player 1"].baseline_season_metric is not None
    assert by_name["Player 1"].current_season_metric is not None
    # In-play values should override field-updates values when available.
    assert by_name["Player 1"].current_position == "T2"
    assert by_name["Player 1"].current_score_to_par == -2.0
    assert by_name["Player 1"].current_thru == "F"
    assert by_name["Player 1"].today_score_to_par == -1.0
    assert by_name["Player 1"].round_scores == [71.0, 69.0]
    assert by_name["Player 1"].hole_scores == [4, 4, 3, 5, 4, 4, 3, 4, 5]
    assert result.stop_reason is not None
    assert result.requested_simulations == 12_000


def test_seasonal_form_gracefully_falls_back_when_data_missing() -> None:
    service = SimulationService(_BrokenSeasonalClient())
    result = asyncio.run(
        service.simulate(
            SimulationRequest(
                tour="pga",
                simulations=4_000,
                seed=7,
                enable_seasonal_form=True,
            )
        )
    )
    assert result.form_adjustment_applied is False
    assert result.form_adjustment_note is not None
