import asyncio
from typing import Optional

import pytest

from pga_sim.datagolf_client import DataGolfAPIError
from pga_sim.models import SimulationRequest
from pga_sim.service import SimulationService


class _FakeDataGolfClient:
    async def get_schedule(
        self, tour: str = "pga", upcoming_only: str = "yes", season: Optional[int] = None
    ):
        return {
            "schedule": [
                {
                    "event_id": "ACTIVE123",
                    "event_name": "Active Event",
                    "start_date": "2026-02-19",
                },
                {
                    "event_id": "FUTURE999",
                    "event_name": "Future Event",
                    "start_date": "2026-04-01",
                },
            ]
        }

    async def get_field_updates(self, tour: str = "pga", event_id: Optional[str] = None):
        return {"event_id": "ACTIVE123", "event_name": "Active Event", "field": []}

    async def get_pre_tournament(
        self,
        tour: str = "pga",
        event_id: Optional[str] = None,
        add_position: int = 3,
        odds_format: str = "percent",
    ):
        return {"predictions": []}

    async def get_player_decompositions(self, tour: str = "pga", event_id: Optional[str] = None):
        return {"decompositions": []}


class _FakeMixedTourScheduleClient(_FakeDataGolfClient):
    async def get_schedule(
        self, tour: str = "pga", upcoming_only: str = "yes", season: Optional[int] = None
    ):
        return {
            "schedule": [
                {
                    "event_id": "PGA111",
                    "event_name": "PGA Event",
                    "tour": "pga",
                    "start_date": "2026-02-01",
                },
                {
                    "event_id": "LIV222",
                    "event_name": "LIV Event",
                    "tour": "liv",
                    "start_date": "2026-02-08",
                },
                {
                    "event_id": "LIV333",
                    "event_name": "LIV Future",
                    "tour": "alt",
                    "start_date": "2026-03-01",
                },
            ]
        }

    async def get_field_updates(self, tour: str = "pga", event_id: Optional[str] = None):
        return {"event_id": "LIV222", "event_name": "LIV Event", "tour": "liv", "field": []}


class _FakeNoTourScheduleMismatchedFieldClient(_FakeDataGolfClient):
    async def get_schedule(
        self, tour: str = "pga", upcoming_only: str = "yes", season: Optional[int] = None
    ):
        return {
            "schedule": [
                {
                    "event_id": "MIXED001",
                    "event_name": "Unknown Tour Event",
                    "start_date": "2026-02-05",
                }
            ]
        }

    async def get_field_updates(self, tour: str = "pga", event_id: Optional[str] = None):
        return {"event_id": "PGA555", "event_name": "PGA Active", "tour": "pga", "field": []}


class _FakeStalePreEventFeedClient(_FakeDataGolfClient):
    async def get_schedule(
        self, tour: str = "pga", upcoming_only: str = "yes", season: Optional[int] = None
    ):
        return {
            "schedule": [
                {
                    "event_id": "719",
                    "event_name": "Hong Kong",
                    "tour": "liv",
                    "start_date": "2026-03-05",
                },
                {
                    "event_id": "713",
                    "event_name": "Singapore",
                    "tour": "liv",
                    "start_date": "2026-03-12",
                },
            ]
        }

    async def get_field_updates(self, tour: str = "pga", event_id: Optional[str] = None):
        raise DataGolfAPIError("DataGolf API error for field-updates: no alt event this week.")

    async def get_pre_tournament(
        self,
        tour: str = "pga",
        event_id: Optional[str] = None,
        add_position: int = 3,
        odds_format: str = "percent",
    ):
        rows = []
        for idx in range(8):
            rows.append(
                {
                    "player_id": f"LIV{idx + 1}",
                    "player_name": f"LIV Player {idx + 1}",
                    "win": max(0.01, 0.18 - (0.02 * idx)),
                    "top_3": max(0.02, 0.35 - (0.025 * idx)),
                    "top_5": max(0.03, 0.45 - (0.03 * idx)),
                    "top_10": max(0.05, 0.62 - (0.04 * idx)),
                }
            )
        return {"event_id": "55", "event_name": "LIV Adelaide", "predictions": rows}

    async def get_player_decompositions(self, tour: str = "pga", event_id: Optional[str] = None):
        rows = []
        for idx in range(8):
            rows.append(
                {
                    "player_id": f"LIV{idx + 1}",
                    "player_name": f"LIV Player {idx + 1}",
                    "sg_total": 1.0 - (0.1 * idx),
                    "sigma": 2.7,
                }
            )
        return {"event_id": "55", "event_name": "LIV Adelaide", "decomposition": rows}


class _FakePredictiveFeedForScheduledEventClient(_FakeDataGolfClient):
    async def get_schedule(
        self, tour: str = "pga", upcoming_only: str = "yes", season: Optional[int] = None
    ):
        return {
            "schedule": [
                {
                    "event_id": "719",
                    "event_name": "Hong Kong",
                    "tour": "liv",
                    "start_date": "2026-03-05",
                },
                {
                    "event_id": "713",
                    "event_name": "Singapore",
                    "tour": "liv",
                    "start_date": "2026-03-12",
                },
            ]
        }

    async def get_field_updates(self, tour: str = "pga", event_id: Optional[str] = None):
        raise DataGolfAPIError("DataGolf API error for field-updates: no alt event this week.")

    async def get_pre_tournament(
        self,
        tour: str = "pga",
        event_id: Optional[str] = None,
        add_position: int = 3,
        odds_format: str = "percent",
    ):
        return {"event_id": "719", "event_name": "Hong Kong", "predictions": []}

    async def get_player_decompositions(self, tour: str = "pga", event_id: Optional[str] = None):
        return {"event_id": "719", "event_name": "Hong Kong", "decomposition": []}


def test_list_upcoming_marks_only_active_event_simulatable() -> None:
    service = SimulationService(_FakeDataGolfClient())
    events = asyncio.run(service.list_upcoming_events(tour="pga", limit=10))
    assert len(events) == 2
    assert events[0].event_id == "ACTIVE123"
    assert events[0].simulatable is True
    assert events[1].event_id == "FUTURE999"
    assert events[1].simulatable is False


def test_list_upcoming_filters_schedule_rows_to_requested_tour() -> None:
    service = SimulationService(_FakeMixedTourScheduleClient())
    events = asyncio.run(service.list_upcoming_events(tour="liv", limit=10))
    event_ids = [event.event_id for event in events]
    assert "PGA111" not in event_ids
    assert "LIV222" in event_ids
    assert "LIV333" in event_ids


def test_list_upcoming_skips_mismatched_active_event_for_requested_tour() -> None:
    service = SimulationService(_FakeNoTourScheduleMismatchedFieldClient())
    events = asyncio.run(service.list_upcoming_events(tour="liv", limit=10))
    assert events == []


def test_list_upcoming_uses_predictive_event_when_field_feed_missing() -> None:
    service = SimulationService(_FakePredictiveFeedForScheduledEventClient())
    events = asyncio.run(service.list_upcoming_events(tour="liv", limit=10))
    assert len(events) == 2
    assert events[0].event_id == "719"
    assert events[0].simulatable is True
    assert events[1].simulatable is False


def test_list_upcoming_no_simulatable_when_predictive_feed_is_stale() -> None:
    service = SimulationService(_FakeStalePreEventFeedClient())
    events = asyncio.run(service.list_upcoming_events(tour="liv", limit=10))
    assert len(events) == 2
    assert all(event.simulatable is False for event in events)


def test_simulate_rejects_non_active_event_selection() -> None:
    service = SimulationService(_FakeDataGolfClient())
    request = SimulationRequest(
        tour="pga",
        event_id="FUTURE999",
        simulations=1_000,
    )

    with pytest.raises(ValueError, match="Selected event is not available"):
        asyncio.run(service.simulate(request))


def test_simulate_rejects_stale_feed_for_selected_event_without_active_field() -> None:
    service = SimulationService(_FakeStalePreEventFeedClient())
    request = SimulationRequest(
        tour="liv",
        event_id="719",
        simulations=1_000,
        enable_seasonal_form=False,
    )

    with pytest.raises(ValueError, match="Selected event is not yet available"):
        asyncio.run(service.simulate(request))
