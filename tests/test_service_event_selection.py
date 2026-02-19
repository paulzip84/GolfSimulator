import asyncio
from typing import Optional

import pytest

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


def test_list_upcoming_marks_only_active_event_simulatable() -> None:
    service = SimulationService(_FakeDataGolfClient())
    events = asyncio.run(service.list_upcoming_events(tour="pga", limit=10))
    assert len(events) == 2
    assert events[0].event_id == "ACTIVE123"
    assert events[0].simulatable is True
    assert events[1].event_id == "FUTURE999"
    assert events[1].simulatable is False


def test_simulate_rejects_non_active_event_selection() -> None:
    service = SimulationService(_FakeDataGolfClient())
    request = SimulationRequest(
        tour="pga",
        event_id="FUTURE999",
        simulations=1_000,
    )

    with pytest.raises(ValueError, match="Selected event is not available"):
        asyncio.run(service.simulate(request))
