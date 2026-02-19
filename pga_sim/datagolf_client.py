from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from .config import Settings


class DataGolfAPIError(RuntimeError):
    pass


class DataGolfClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.datagolf_base_url.rstrip("/"),
            timeout=settings.http_timeout_seconds,
        )

    async def __aenter__(self) -> "DataGolfClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get_json(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        api_key = self._settings.datagolf_api_key.strip()
        if not api_key:
            raise DataGolfAPIError(
                "DATAGOLF_API_KEY is not configured. "
                "Create /Users/paulzip84/Documents/New project/.env and set DATAGOLF_API_KEY=..."
            )

        query_params: dict[str, Any] = {
            "file_format": "json",
            "key": api_key,
        }
        if params:
            query_params.update({k: v for k, v in params.items() if v is not None})

        response = await self._client.get(f"/{path.lstrip('/')}", params=query_params)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise DataGolfAPIError(
                f"DataGolf request failed ({exc.response.status_code}) for {path}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise DataGolfAPIError(f"DataGolf returned non-JSON payload for {path}") from exc

        if isinstance(payload, dict) and payload.get("error"):
            raise DataGolfAPIError(f"DataGolf API error for {path}: {payload['error']}")
        return payload

    @staticmethod
    def _normalize_tour(tour: str) -> str:
        normalized = tour.strip().lower()
        aliases = {
            "dpwt": "euro",
            "european": "euro",
            "liv": "alt",
        }
        return aliases.get(normalized, normalized)

    async def get_schedule(
        self,
        tour: str = "pga",
        upcoming_only: str = "yes",
        season: int | None = None,
    ) -> Any:
        tour_code = self._normalize_tour(tour)
        return await self._get_json(
            "get-schedule",
            params={"tour": tour_code, "upcoming_only": upcoming_only, "season": season},
        )

    async def get_field_updates(self, tour: str = "pga", event_id: str | None = None) -> Any:
        tour_code = self._normalize_tour(tour)
        # Current-week field feed is keyed by tour. `event_id` is accepted in method
        # signature to keep a stable interface but is not sent.
        return await self._get_json("field-updates", params={"tour": tour_code})

    async def get_pre_tournament(
        self,
        tour: str = "pga",
        event_id: str | None = None,
        add_position: int = 3,
        odds_format: str = "percent",
    ) -> Any:
        tour_code = self._normalize_tour(tour)
        # Current-week pre-tournament feed is keyed by tour only.
        return await self._get_json(
            "preds/pre-tournament",
            params={
                "tour": tour_code,
                "add_position": add_position,
                "odds_format": odds_format,
            },
        )

    async def get_player_decompositions(
        self, tour: str = "pga", event_id: str | None = None
    ) -> Any:
        tour_code = self._normalize_tour(tour)
        # Current-week decomposition feed is keyed by tour only.
        return await self._get_json("preds/player-decompositions", params={"tour": tour_code})

    async def get_historical_event_list(self, tour: str = "pga") -> Any:
        tour_code = self._normalize_tour(tour)
        return await self._get_json("historical-event-data/event-list", params={"tour": tour_code})

    async def get_historical_event(
        self,
        tour: str,
        event_id: str,
        year: int,
    ) -> Any:
        tour_code = self._normalize_tour(tour)
        return await self._get_json(
            "historical-event-data/events",
            params={
                "tour": tour_code,
                "event_id": event_id,
                "year": year,
            },
        )

    async def get_historical_rounds(
        self,
        tour: str = "pga",
        season: int | None = None,
    ) -> Any:
        # Backwards-compatible helper. Uses historical event list + events endpoints.
        if season is None:
            raise DataGolfAPIError("season is required for historical rounds requests.")

        event_list_payload = await self.get_historical_event_list(tour=tour)

        rows: list[dict[str, Any]] = []
        if isinstance(event_list_payload, dict):
            candidate_rows = event_list_payload.get("event_list") or event_list_payload.get("events")
            if isinstance(candidate_rows, list):
                rows = [row for row in candidate_rows if isinstance(row, dict)]
        elif isinstance(event_list_payload, list):
            rows = [row for row in event_list_payload if isinstance(row, dict)]

        events: list[tuple[str, int]] = []
        for row in rows:
            event_id = None
            for key in ("event_id", "id", "event"):
                value = row.get(key)
                if value:
                    event_id = str(value)
                    break
            year = row.get("year") or row.get("season") or season
            try:
                year_int = int(year)
            except (TypeError, ValueError):
                continue
            if year_int != season or not event_id:
                continue
            events.append((event_id, year_int))

        if not events:
            raise DataGolfAPIError(
                f"No historical events found for tour={tour} season={season}."
            )

        payloads = []
        for event_id, year in events:
            try:
                payloads.append(await self.get_historical_event(tour=tour, event_id=event_id, year=year))
            except DataGolfAPIError:
                continue
        return {"events": payloads}
