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
        }
        return aliases.get(normalized, normalized)

    def _tour_candidates(self, tour: str) -> list[str]:
        raw = str(tour or "").strip().lower()
        normalized = self._normalize_tour(raw)
        if normalized in {"liv", "alt"}:
            if raw == "alt":
                return ["alt", "liv"]
            return ["liv", "alt"]
        return [normalized]

    @staticmethod
    def _normalize_tour_code(value: Any) -> str | None:
        if value is None:
            return None
        text = " ".join(str(value).strip().lower().split())
        if not text:
            return None
        compact = text.replace("-", "").replace("_", "").replace(" ", "")
        if compact in {"pga", "pgatour"}:
            return "pga"
        if compact in {"liv", "livgolf", "alt"}:
            return "liv"
        if compact in {"euro", "dpwt", "dpworldtour", "european", "europeantour"}:
            return "euro"
        if compact in {"kft", "kornferry", "kornferrytour"}:
            return "kft"
        if "pga" in text:
            return "pga"
        if "liv" in text:
            return "liv"
        if ("dp" in text and "world" in text) or "european" in text:
            return "euro"
        if "korn" in text and "ferry" in text:
            return "kft"
        return compact

    @staticmethod
    def _extract_schedule_rows(payload: Any) -> list[Mapping[str, Any]]:
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, Mapping)]
        if not isinstance(payload, Mapping):
            return []
        candidates = (
            payload.get("schedule"),
            payload.get("event"),
            payload.get("events"),
        )
        for candidate in candidates:
            if isinstance(candidate, list):
                return [row for row in candidate if isinstance(row, Mapping)]
        return []

    @staticmethod
    def _payload_tour_codes(payload: Any) -> set[str]:
        values: set[str] = set()
        if isinstance(payload, Mapping):
            for key in ("tour", "tour_name", "tour_code", "tour_nm"):
                normalized = DataGolfClient._normalize_tour_code(payload.get(key))
                if normalized:
                    values.add(normalized)
            for nested_key in ("event", "tournament", "meta", "metadata"):
                nested = payload.get(nested_key)
                if isinstance(nested, Mapping):
                    for key in ("tour", "tour_name", "tour_code", "tour_nm"):
                        normalized = DataGolfClient._normalize_tour_code(nested.get(key))
                        if normalized:
                            values.add(normalized)
        return values

    def _payload_tour_match_status(self, path: str, payload: Any, tour_code: str) -> str:
        # Returns "match", "mismatch", or "unknown".
        requested = self._normalize_tour_code(tour_code)
        if not requested:
            return "unknown"

        path_key = str(path).strip().lower()
        if "get-schedule" in path_key:
            rows = self._extract_schedule_rows(payload)
            row_tours: set[str] = set()
            for row in rows:
                for key in ("tour", "tour_name", "tour_code", "tour_nm"):
                    normalized = self._normalize_tour_code(row.get(key))
                    if normalized:
                        row_tours.add(normalized)
                        break
            if not row_tours:
                return "unknown"
            return "match" if requested in row_tours else "mismatch"

        payload_tours = self._payload_tour_codes(payload)
        if not payload_tours:
            return "unknown"
        return "match" if requested in payload_tours else "mismatch"

    async def _get_json_for_tour(
        self,
        path: str,
        *,
        tour: str,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        candidates = self._tour_candidates(tour)
        last_error: DataGolfAPIError | None = None
        fallback_payload: Any | None = None
        saw_mismatch = False
        extra_params = dict(params or {})
        for tour_code in candidates:
            query = dict(extra_params)
            query["tour"] = tour_code
            try:
                payload = await self._get_json(path, params=query)
            except DataGolfAPIError as exc:
                last_error = exc
                continue
            match_status = self._payload_tour_match_status(path=path, payload=payload, tour_code=tour_code)
            if match_status == "match":
                return payload
            if match_status == "unknown" and fallback_payload is None:
                fallback_payload = payload
                continue
            saw_mismatch = True
            continue
        if fallback_payload is not None:
            return fallback_payload
        if last_error is not None:
            raise last_error
        if saw_mismatch:
            raise DataGolfAPIError(
                f"DataGolf returned payload for unexpected tour for {path}"
            )
        raise DataGolfAPIError(f"Unable to resolve tour for {path}.")

    async def get_schedule(
        self,
        tour: str = "pga",
        upcoming_only: str = "yes",
        season: int | None = None,
    ) -> Any:
        return await self._get_json_for_tour(
            "get-schedule",
            tour=tour,
            params={"upcoming_only": upcoming_only, "season": season},
        )

    async def get_field_updates(self, tour: str = "pga", event_id: str | None = None) -> Any:
        # Current-week field feed is keyed by tour. `event_id` is accepted in method
        # signature to keep a stable interface but is not sent.
        return await self._get_json_for_tour("field-updates", tour=tour)

    async def get_pre_tournament(
        self,
        tour: str = "pga",
        event_id: str | None = None,
        add_position: int = 3,
        odds_format: str = "percent",
    ) -> Any:
        # Current-week pre-tournament feed is keyed by tour only.
        return await self._get_json_for_tour(
            "preds/pre-tournament",
            tour=tour,
            params={
                "add_position": add_position,
                "odds_format": odds_format,
            },
        )

    async def get_player_decompositions(
        self, tour: str = "pga", event_id: str | None = None
    ) -> Any:
        # Current-week decomposition feed is keyed by tour only.
        return await self._get_json_for_tour("preds/player-decompositions", tour=tour)

    async def get_in_play(
        self,
        tour: str = "pga",
        dead_heat: str = "no",
        odds_format: str = "percent",
    ) -> Any:
        return await self._get_json_for_tour(
            "preds/in-play",
            tour=tour,
            params={
                "dead_heat": dead_heat,
                "odds_format": odds_format,
            },
        )

    async def get_historical_event_list(self, tour: str = "pga") -> Any:
        return await self._get_json_for_tour(
            "historical-event-data/event-list",
            tour=tour,
        )

    async def get_historical_event(
        self,
        tour: str,
        event_id: str,
        year: int,
    ) -> Any:
        return await self._get_json_for_tour(
            "historical-event-data/events",
            tour=tour,
            params={
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
