from __future__ import annotations

import asyncio

import pytest

from pga_sim.datagolf_client import DataGolfAPIError
from pga_sim.learning import LearningStore
from pga_sim.service import SimulationService


class _NoopDataGolfClient:
    pass


def _prediction_players(prefix: str) -> list[dict[str, float | str]]:
    players: list[dict[str, float | str]] = []
    for idx in range(8):
        players.append(
            {
                "player_id": f"{prefix}-{idx + 1}",
                "player_name": f"{prefix.upper()} Player {idx + 1}",
                "win_probability": max(0.01, 0.18 - (0.015 * idx)),
                "top_3_probability": max(0.04, 0.38 - (0.02 * idx)),
                "top_5_probability": max(0.07, 0.55 - (0.03 * idx)),
                "top_10_probability": max(0.12, 0.78 - (0.04 * idx)),
            }
        )
    return players


class _WarmStartClient:
    @staticmethod
    def _event_id(tour: str) -> str:
        return f"{tour}-event-1"

    @staticmethod
    def _event_name(tour: str) -> str:
        labels = {
            "pga": "PGA Warm Event",
            "liv": "LIV Warm Event",
            "euro": "DP Warm Event",
            "kft": "KFT Warm Event",
        }
        return labels.get(tour, f"{tour.upper()} Warm Event")

    async def get_schedule(
        self,
        tour: str = "pga",
        upcoming_only: str = "yes",
        season: int | None = None,
    ):
        return [
            {
                "event_id": self._event_id(tour),
                "event_name": self._event_name(tour),
                "date": "2026-03-10",
                "tour": tour,
            }
        ]

    async def get_field_updates(self, tour: str = "pga", event_id: str | None = None):
        return {
            "tour": tour,
            "event_id": self._event_id(tour),
            "event_name": self._event_name(tour),
            "date": "2026-03-10",
            "field": [
                {
                    "player_id": str(player["player_id"]),
                    "player_name": str(player["player_name"]),
                    "position": "",
                    "score_to_par": None,
                    "thru": 0,
                }
                for player in _prediction_players(tour)
            ],
        }

    async def get_pre_tournament(
        self,
        tour: str = "pga",
        event_id: str | None = None,
        add_position: int = 3,
        odds_format: str = "percent",
    ):
        return {
            "tour": tour,
            "event_id": self._event_id(tour),
            "event_name": self._event_name(tour),
            "pred": _prediction_players(tour),
        }

    async def get_player_decompositions(self, tour: str = "pga", event_id: str | None = None):
        return {
            "tour": tour,
            "event_id": self._event_id(tour),
            "event_name": self._event_name(tour),
            "decomposition": [
                {
                    "player_id": str(player["player_id"]),
                    "player_name": str(player["player_name"]),
                    "sg_total": 1.1 - (0.12 * idx),
                    "sigma": 2.6,
                }
                for idx, player in enumerate(_prediction_players(tour), start=1)
            ],
        }

    async def get_in_play(
        self,
        tour: str = "pga",
        dead_heat: str = "no",
        odds_format: str = "percent",
    ):
        return {}


class _WarmStartMissingDatesClient(_WarmStartClient):
    async def get_schedule(
        self,
        tour: str = "pga",
        upcoming_only: str = "yes",
        season: int | None = None,
    ):
        return [
            {
                "event_id": self._event_id(tour),
                "event_name": self._event_name(tour),
                "tour": tour,
            }
        ]

    async def get_field_updates(self, tour: str = "pga", event_id: str | None = None):
        payload = await super().get_field_updates(tour=tour, event_id=event_id)
        payload.pop("date", None)
        return payload

    async def get_pre_tournament(
        self,
        tour: str = "pga",
        event_id: str | None = None,
        add_position: int = 3,
        odds_format: str = "percent",
    ):
        payload = await super().get_pre_tournament(
            tour=tour,
            event_id=event_id,
            add_position=add_position,
            odds_format=odds_format,
        )
        payload.pop("date", None)
        return payload


class _WarmStartNoEventIdClient(_WarmStartClient):
    async def get_field_updates(self, tour: str = "pga", event_id: str | None = None):
        raise DataGolfAPIError("field-updates unavailable")

    async def get_pre_tournament(
        self,
        tour: str = "pga",
        event_id: str | None = None,
        add_position: int = 3,
        odds_format: str = "percent",
    ):
        payload = await super().get_pre_tournament(
            tour=tour,
            event_id=event_id,
            add_position=add_position,
            odds_format=odds_format,
        )
        payload.pop("event_id", None)
        payload["event_name"] = f"{tour.upper()} Name-Only Event"
        return payload

    async def get_player_decompositions(self, tour: str = "pga", event_id: str | None = None):
        payload = await super().get_player_decompositions(tour=tour, event_id=event_id)
        payload.pop("event_id", None)
        payload["event_name"] = f"{tour.upper()} Name-Only Event"
        return payload


def _outcome_payload(event_name: str, rows: list[tuple[int, str, int]]) -> dict:
    return {
        "event_name": event_name,
        "event_completed": "2026-01-21",
        "event_stats": [
            {
                "dg_id": player_id,
                "player_name": player_name,
                "fin_text": str(rank),
                "dg_points": float(max(1, 35 - rank)),
            }
            for player_id, player_name, rank in rows
        ],
    }


def test_power_rankings_report_builds_series_from_learning_history(tmp_path) -> None:
    learning_store = LearningStore(str(tmp_path / "service_reports.sqlite3"))
    service = SimulationService(
        _NoopDataGolfClient(),
        learning_store=learning_store,
        lifecycle_target_year=2026,
    )

    event_one_players = [
        {
            "player_id": "5001",
            "player_name": "Alpha",
            "win_probability": 0.20,
            "top_3_probability": 0.45,
            "top_5_probability": 0.58,
            "top_10_probability": 0.76,
        },
        {
            "player_id": "5002",
            "player_name": "Beta",
            "win_probability": 0.14,
            "top_3_probability": 0.34,
            "top_5_probability": 0.50,
            "top_10_probability": 0.71,
        },
        {
            "player_id": "5003",
            "player_name": "Gamma",
            "win_probability": 0.09,
            "top_3_probability": 0.24,
            "top_5_probability": 0.40,
            "top_10_probability": 0.63,
        },
    ]
    event_two_players = [
        {
            "player_id": "5001",
            "player_name": "Alpha",
            "win_probability": 0.16,
            "top_3_probability": 0.39,
            "top_5_probability": 0.54,
            "top_10_probability": 0.74,
        },
        {
            "player_id": "5002",
            "player_name": "Beta",
            "win_probability": 0.23,
            "top_3_probability": 0.47,
            "top_5_probability": 0.62,
            "top_10_probability": 0.80,
        },
        {
            "player_id": "5003",
            "player_name": "Gamma",
            "win_probability": 0.08,
            "top_3_probability": 0.21,
            "top_5_probability": 0.36,
            "top_10_probability": 0.60,
        },
    ]

    learning_store.record_prediction(
        tour="pga",
        event_id="100",
        event_name="Event One",
        event_date="2026-01-10",
        requested_simulations=100000,
        simulations=100000,
        enable_in_play=True,
        in_play_applied=False,
        snapshot_type="pre_event",
        players=event_one_players,
    )
    learning_store.record_prediction(
        tour="pga",
        event_id="101",
        event_name="Event Two",
        event_date="2026-01-20",
        requested_simulations=100000,
        simulations=100000,
        enable_in_play=True,
        in_play_applied=False,
        snapshot_type="pre_event",
        players=event_two_players,
    )
    learning_store.record_outcome_payload(
        tour="pga",
        event_id="100",
        event_year=2026,
        payload=_outcome_payload(
            "Event One",
            [
                (5001, "Alpha", 1),
                (5002, "Beta", 2),
                (5003, "Gamma", 3),
            ],
        ),
    )

    report = asyncio.run(
        service.get_power_rankings_report(
            tour="pga",
            lookback_events=12,
            top_n=2,
            event_year=2026,
        )
    )
    assert report.tour == "pga"
    assert report.event_year == 2026
    assert len(report.events) == 2
    assert len(report.players) == 2
    assert report.players[0].latest_rank == 1
    assert len(report.players[0].points) == 2
    assert report.players[0].points[-1].rank == report.players[0].latest_rank
    assert report.events[0].source_snapshot_type == "pre_event"


def test_power_rankings_report_raises_when_no_history(tmp_path) -> None:
    learning_store = LearningStore(str(tmp_path / "service_reports_empty.sqlite3"))
    service = SimulationService(
        _NoopDataGolfClient(),
        learning_store=learning_store,
        lifecycle_target_year=2026,
    )
    with pytest.raises(LookupError):
        asyncio.run(
            service.get_power_rankings_report(
                tour="pga",
                lookback_events=10,
                top_n=10,
                event_year=2026,
            )
        )


def test_power_rankings_warm_start_seeds_missing_tours(tmp_path) -> None:
    learning_store = LearningStore(str(tmp_path / "service_reports_warm_start.sqlite3"))
    service = SimulationService(
        _WarmStartClient(),
        learning_store=learning_store,
        lifecycle_target_year=2026,
        simulation_max_sync_simulations=50_000,
    )

    learning_store.record_prediction(
        tour="pga",
        event_id="seeded-pga",
        event_name="Already Seeded PGA",
        event_date="2026-01-08",
        requested_simulations=10_000,
        simulations=10_000,
        enable_in_play=False,
        in_play_applied=False,
        snapshot_type="pre_event",
        players=_prediction_players("pga-seed"),
    )

    payload = asyncio.run(
        service.warm_start_power_rankings(
            tours=["pga", "liv"],
            event_year=2026,
            simulations=12_000,
            force=False,
        )
    )

    by_tour = {row.tour: row for row in payload.results}
    assert by_tour["pga"].status == "skipped_existing"
    assert by_tour["liv"].status == "simulated"
    assert by_tour["liv"].event_id is not None
    assert (by_tour["liv"].simulation_version or 0) >= 1

    liv_report = asyncio.run(
        service.get_power_rankings_report(
            tour="liv",
            lookback_events=12,
            top_n=8,
            event_year=2026,
        )
    )
    assert len(liv_report.events) >= 1
    assert len(liv_report.players) >= 1


def test_power_rankings_warm_start_persists_year_when_feed_dates_missing(tmp_path) -> None:
    learning_store = LearningStore(str(tmp_path / "service_reports_warm_start_missing_dates.sqlite3"))
    service = SimulationService(
        _WarmStartMissingDatesClient(),
        learning_store=learning_store,
        lifecycle_target_year=2026,
        simulation_max_sync_simulations=50_000,
    )

    payload = asyncio.run(
        service.warm_start_power_rankings(
            tours=["euro"],
            event_year=2026,
            simulations=12_000,
            force=True,
        )
    )

    assert len(payload.results) == 1
    assert payload.results[0].status == "simulated"
    assert payload.results[0].event_year == 2026

    euro_report = asyncio.run(
        service.get_power_rankings_report(
            tour="euro",
            lookback_events=12,
            top_n=8,
            event_year=2026,
        )
    )
    assert len(euro_report.events) >= 1
    assert len(euro_report.players) >= 1


def test_power_rankings_warm_start_seeds_synthetic_event_id_when_missing(tmp_path) -> None:
    learning_store = LearningStore(
        str(tmp_path / "service_reports_warm_start_synthetic_event.sqlite3")
    )
    service = SimulationService(
        _WarmStartNoEventIdClient(),
        learning_store=learning_store,
        lifecycle_target_year=2026,
        simulation_max_sync_simulations=50_000,
    )

    payload = asyncio.run(
        service.warm_start_power_rankings(
            tours=["liv"],
            event_year=2026,
            simulations=12_000,
            force=True,
        )
    )
    assert len(payload.results) == 1
    assert payload.results[0].status == "simulated"
    assert payload.results[0].event_id is not None
    assert payload.results[0].event_id.startswith("warm-2026-")
    assert payload.results[0].note is not None
    assert "synthetic event_id" in payload.results[0].note

    liv_report = asyncio.run(
        service.get_power_rankings_report(
            tour="liv",
            lookback_events=12,
            top_n=8,
            event_year=2026,
        )
    )
    assert len(liv_report.events) >= 1
    assert len(liv_report.players) >= 1
