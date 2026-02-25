import asyncio

from pga_sim.datagolf_client import DataGolfAPIError
from pga_sim.datagolf_client import DataGolfClient


def test_normalize_tour_maps_known_aliases() -> None:
    assert DataGolfClient._normalize_tour("dpwt") == "euro"
    assert DataGolfClient._normalize_tour("european") == "euro"
    assert DataGolfClient._normalize_tour("alt") == "alt"


def test_normalize_tour_keeps_liv_code() -> None:
    assert DataGolfClient._normalize_tour("liv") == "liv"


def _make_client_with_stub(stub):
    client = DataGolfClient.__new__(DataGolfClient)
    client._get_json = stub
    return client


def test_get_json_for_tour_fallback_uses_alt_when_liv_fails() -> None:
    calls = []

    async def stub(path, params=None):
        tour = str((params or {}).get("tour", ""))
        calls.append(tour)
        if tour == "liv":
            raise DataGolfAPIError("DataGolf request failed (404) for get-schedule")
        return {"ok": True, "tour": tour}

    client = _make_client_with_stub(stub)
    payload = asyncio.run(
        client._get_json_for_tour(
            "get-schedule",
            tour="liv",
            params={"upcoming_only": "yes"},
        )
    )
    assert calls == ["liv", "alt"]
    assert payload["ok"] is True
    assert payload["tour"] == "alt"


def test_get_json_for_tour_single_try_for_non_liv_tour() -> None:
    calls = []

    async def stub(path, params=None):
        tour = str((params or {}).get("tour", ""))
        calls.append(tour)
        return {"ok": True}

    client = _make_client_with_stub(stub)
    payload = asyncio.run(client._get_json_for_tour("get-schedule", tour="pga"))
    assert payload["ok"] is True
    assert calls == ["pga"]


def test_get_json_for_tour_ignores_mismatched_schedule_payload_and_uses_fallback() -> None:
    calls = []

    async def stub(path, params=None):
        tour = str((params or {}).get("tour", ""))
        calls.append(tour)
        if tour == "liv":
            return {
                "schedule": [
                    {"event_id": "PGA111", "event_name": "PGA Event", "tour": "pga"},
                ]
            }
        return {
            "schedule": [
                {"event_id": "LIV222", "event_name": "LIV Event", "tour": "liv"},
            ]
        }

    client = _make_client_with_stub(stub)
    payload = asyncio.run(client._get_json_for_tour("get-schedule", tour="liv"))
    assert calls == ["liv", "alt"]
    schedule = payload["schedule"]
    assert schedule[0]["event_id"] == "LIV222"
