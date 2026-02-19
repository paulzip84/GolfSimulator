from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import re
from statistics import NormalDist
from typing import Any

import numpy as np

from .datagolf_client import DataGolfAPIError, DataGolfClient
from .models import EventSummary, PlayerSimulationOutput, SimulationRequest, SimulationResponse
from .simulation import HybridMarkovSimulator, MarkovSimulationConfig, SimulationInputs

_ID_KEYS = ("dg_id", "player_id", "id", "player")
_NAME_KEYS = ("player_name", "name", "player", "golfer")
_EVENT_ID_KEYS = ("event_id", "eventid", "tournament_id", "event")
_EVENT_NAME_KEYS = ("event_name", "tournament_name", "event")
_DATE_KEYS = ("date", "start_date", "start", "event_start")
_COURSE_KEYS = ("course", "course_name", "course_nm")
_WIN_KEYS = ("win", "win_prob", "win_probability", "outright", "winner")
_TOP3_KEYS = ("top_3", "top3", "position_3", "pos_3", "top_three")
_TOP5_KEYS = ("top_5", "top5", "position_5", "pos_5")
_TOP10_KEYS = ("top_10", "top10", "position_10", "pos_10")
_SKILL_KEYS = (
    "sg_total",
    "skill",
    "true_skill",
    "pred_skill",
    "baseline_skill",
    "total",
    "projection",
)
_SIGMA_KEYS = ("sigma", "sd", "round_sd", "round_std", "stdev", "volatility")
_ROUND_METRIC_RULES: tuple[tuple[str, float], ...] = (
    ("sg_total", 1.0),
    ("strokes_gained_total", 1.0),
    ("strokes_gained", 1.0),
    ("sg", 1.0),
    ("adj_score_to_par", -1.0),
    ("score_to_par", -1.0),
    ("round_score_to_par", -1.0),
    ("score_relative_to_field", -1.0),
)


@dataclass
class _PlayerRecord:
    player_id: str
    player_name: str
    skill: float | None = None
    sigma: float | None = None
    baseline_win_probability: float | None = None
    baseline_top_3_probability: float | None = None
    baseline_top_5_probability: float | None = None
    baseline_top_10_probability: float | None = None
    baseline_season_metric: float | None = None
    current_season_metric: float | None = None
    form_delta_metric: float | None = None
    baseline_season_rounds: int = 0
    current_season_rounds: int = 0
    baseline_season_volatility: float | None = None
    current_season_volatility: float | None = None


@dataclass
class _SeasonLoadResult:
    metrics: dict[str, dict[str, Any]]
    event_count: int
    successful_event_count: int
    player_metric_count: int
    observation_count: int
    from_cache: bool = False


class SimulationService:
    def __init__(self, datagolf: DataGolfClient):
        self._datagolf = datagolf
        self._season_metrics_cache: dict[tuple[str, int], _SeasonLoadResult] = {}

    async def list_upcoming_events(self, tour: str = "pga", limit: int = 12) -> list[EventSummary]:
        schedule_payload = await self._datagolf.get_schedule(tour=tour)
        rows = _extract_rows(schedule_payload, ("schedule", "event"))

        active_event_id: str | None = None
        active_event_name: str | None = None
        active_event_date: str | None = None
        active_event_course: str | None = None
        try:
            field_payload = await self._datagolf.get_field_updates(tour=tour)
            active_event_id = _normalized_event_id(
                _string_from_payload(field_payload, _EVENT_ID_KEYS)
            )
            active_event_name = _string_from_payload(field_payload, _EVENT_NAME_KEYS)
            active_event_date = _string_from_payload(field_payload, _DATE_KEYS)
            active_event_course = _string_from_payload(field_payload, _COURSE_KEYS)
        except DataGolfAPIError:
            pass

        events: list[EventSummary] = []
        for row in rows:
            event_id = _string_from_keys(row, _EVENT_ID_KEYS)
            event_name = _string_from_keys(row, _EVENT_NAME_KEYS)
            if not event_id or not event_name:
                continue

            normalized_event_id = _normalized_event_id(event_id)
            simulatable = bool(active_event_id and normalized_event_id == active_event_id)
            events.append(
                EventSummary(
                    event_id=event_id,
                    event_name=event_name,
                    start_date=_string_from_keys(row, _DATE_KEYS),
                    course=_string_from_keys(row, _COURSE_KEYS),
                    simulatable=simulatable,
                    unavailable_reason=(
                        None
                        if simulatable
                        else "DataGolf current-week simulation feeds are keyed to the active event for this tour."
                    ),
                )
            )

        active_already_listed = any(
            _normalized_event_id(event.event_id) == active_event_id for event in events
        )
        if active_event_id and not active_already_listed:
            events.append(
                EventSummary(
                    event_id=active_event_id,
                    event_name=active_event_name or "Active Tour Event",
                    start_date=active_event_date,
                    course=active_event_course,
                    simulatable=True,
                    unavailable_reason=None,
                )
            )

        if not active_event_id and events:
            events[0].simulatable = True
            events[0].unavailable_reason = None

        events.sort(key=lambda e: (not e.simulatable, _safe_date_sort_key(e.start_date)))
        return events[:limit]

    async def simulate(self, request: SimulationRequest) -> SimulationResponse:
        field_payload, pre_payload, decomp_payload = await asyncio.gather(
            self._datagolf.get_field_updates(tour=request.tour, event_id=request.event_id),
            self._datagolf.get_pre_tournament(
                tour=request.tour,
                event_id=request.event_id,
                add_position=3,
                odds_format="percent",
            ),
            self._datagolf.get_player_decompositions(
                tour=request.tour,
                event_id=request.event_id,
            ),
        )

        selected_event_id = _normalized_event_id(request.event_id)
        active_event_id = _normalized_event_id(_string_from_payload(field_payload, _EVENT_ID_KEYS))
        if selected_event_id and active_event_id and selected_event_id != active_event_id:
            active_event_name = _string_from_payload(field_payload, _EVENT_NAME_KEYS) or "Unknown"
            raise ValueError(
                "Selected event is not available for this tour in DataGolf current-week simulation "
                f"feeds. Active event: {active_event_name} (event_id={active_event_id})."
            )

        field_rows = _extract_rows(field_payload, ("field", "player"))
        pre_rows = _extract_rows(pre_payload, ("pred", "tournament", "player"))
        decomp_rows = _extract_rows(decomp_payload, ("decomposition", "player"))

        players = self._merge_player_records(field_rows, pre_rows, decomp_rows)
        if len(players) < 8:
            raise ValueError(
                "Unable to build enough players from DataGolf payloads. "
                "Verify tour/event_id and API access."
            )

        baseline_season, current_season = self._resolve_season_window(request)
        form_adjustment_applied = False
        form_adjustment_note: str | None = None
        if request.enable_seasonal_form:
            try:
                baseline_result, current_result = await asyncio.gather(
                    self._load_season_round_metrics(
                        tour=request.tour,
                        season=baseline_season,
                    ),
                    self._load_season_round_metrics(
                        tour=request.tour,
                        season=current_season,
                    ),
                )
                baseline_metrics = baseline_result.metrics
                current_metrics = current_result.metrics
                applied_count = self._apply_seasonal_form_metrics(
                    players=players,
                    baseline_metrics=baseline_metrics,
                    current_metrics=current_metrics,
                )
                form_adjustment_applied = applied_count > 0
                summary = (
                    f"Seasonal data loaded: baseline {baseline_season} -> "
                    f"{baseline_result.player_metric_count} players "
                    f"({baseline_result.successful_event_count}/{baseline_result.event_count} events), "
                    f"current {current_season} -> {current_result.player_metric_count} players "
                    f"({current_result.successful_event_count}/{current_result.event_count} events). "
                    f"Matched active field players: {applied_count}."
                )
                if form_adjustment_applied:
                    form_adjustment_note = summary
                else:
                    form_adjustment_note = (
                        "No overlapping players between active field and seasonal metrics. "
                        + summary
                    )
            except DataGolfAPIError as exc:
                form_adjustment_note = (
                    "Seasonal form adjustment not applied because historical event data could not be "
                    f"loaded: {exc}"
                )
        else:
            form_adjustment_note = "Seasonal form adjustment disabled."

        model_inputs = self._build_simulation_inputs(
            players=players,
            seasonal_form_weight=request.seasonal_form_weight,
            current_season_weight=request.current_season_weight,
            form_delta_weight=request.form_delta_weight,
        )
        simulator = HybridMarkovSimulator(
            MarkovSimulationConfig(mean_reversion=request.mean_reversion)
        )
        outputs = simulator.simulate(
            inputs=model_inputs,
            n_simulations=request.simulations,
            seed=request.seed,
            cut_size=request.cut_size,
        )

        rankings = np.argsort(outputs.win_probability)[::-1]
        result_rows: list[PlayerSimulationOutput] = []
        for idx in rankings:
            record = players[idx]
            result_rows.append(
                PlayerSimulationOutput(
                    player_id=record.player_id,
                    player_name=record.player_name,
                    win_probability=float(outputs.win_probability[idx]),
                    top_3_probability=float(outputs.top_3_probability[idx]),
                    top_5_probability=float(outputs.top_5_probability[idx]),
                    top_10_probability=float(outputs.top_10_probability[idx]),
                    mean_finish=float(outputs.mean_finish[idx]),
                    mean_total_relative_to_field=float(
                        outputs.mean_total_relative_to_field[idx]
                    ),
                    baseline_win_probability=record.baseline_win_probability,
                    baseline_top_3_probability=record.baseline_top_3_probability,
                    baseline_top_5_probability=record.baseline_top_5_probability,
                    baseline_top_10_probability=record.baseline_top_10_probability,
                    baseline_season_metric=record.baseline_season_metric,
                    current_season_metric=record.current_season_metric,
                    form_delta_metric=record.form_delta_metric,
                    baseline_season_rounds=record.baseline_season_rounds,
                    current_season_rounds=record.current_season_rounds,
                )
            )

        return SimulationResponse(
            generated_at=datetime.now(timezone.utc),
            tour=request.tour,
            event_id=(
                request.event_id
                or _string_from_payload(field_payload, _EVENT_ID_KEYS)
                or _string_from_payload(pre_payload, _EVENT_ID_KEYS)
            ),
            event_name=(
                _string_from_payload(field_payload, _EVENT_NAME_KEYS)
                or _string_from_payload(pre_payload, _EVENT_NAME_KEYS)
            ),
            simulations=request.simulations,
            baseline_season=baseline_season if request.enable_seasonal_form else None,
            current_season=current_season if request.enable_seasonal_form else None,
            form_adjustment_applied=form_adjustment_applied,
            form_adjustment_note=form_adjustment_note,
            players=result_rows,
        )

    def _merge_player_records(
        self,
        field_rows: list[dict[str, Any]],
        pre_rows: list[dict[str, Any]],
        decomp_rows: list[dict[str, Any]],
    ) -> list[_PlayerRecord]:
        merged: dict[str, _PlayerRecord] = {}
        name_index: dict[str, str] = {}

        for row in field_rows:
            key, player_id, player_name = _player_identity(row)
            if not key:
                continue
            resolved_key = _resolve_player_key(key, player_name, merged, name_index)
            merged[resolved_key] = _PlayerRecord(player_id=player_id, player_name=player_name)
            name_index[_normalized_name(player_name)] = resolved_key

        for row in pre_rows:
            key, player_id, player_name = _player_identity(row)
            if not key:
                continue
            resolved_key = _resolve_player_key(key, player_name, merged, name_index)
            record = merged.setdefault(
                resolved_key,
                _PlayerRecord(player_id=player_id, player_name=player_name),
            )
            name_index[_normalized_name(player_name)] = resolved_key
            record.baseline_win_probability = _probability_from_keys(row, _WIN_KEYS)
            record.baseline_top_3_probability = _probability_from_keys(row, _TOP3_KEYS)
            record.baseline_top_5_probability = _probability_from_keys(row, _TOP5_KEYS)
            record.baseline_top_10_probability = _probability_from_keys(row, _TOP10_KEYS)
            record.skill = record.skill or _skill_from_row(row)
            record.sigma = record.sigma or _numeric_from_keys(row, _SIGMA_KEYS)

        for row in decomp_rows:
            key, player_id, player_name = _player_identity(row)
            if not key:
                continue
            resolved_key = _resolve_player_key(key, player_name, merged, name_index)
            record = merged.setdefault(
                resolved_key,
                _PlayerRecord(player_id=player_id, player_name=player_name),
            )
            name_index[_normalized_name(player_name)] = resolved_key
            row_skill = _skill_from_row(row)
            row_sigma = _numeric_from_keys(row, _SIGMA_KEYS)
            if row_skill is not None:
                record.skill = row_skill
            if row_sigma is not None:
                record.sigma = row_sigma

        return list(merged.values())

    @staticmethod
    def _resolve_season_window(request: SimulationRequest) -> tuple[int, int]:
        now_year = datetime.now(timezone.utc).year
        baseline_season = request.baseline_season
        current_season = request.current_season

        if baseline_season is not None and current_season is not None:
            return baseline_season, current_season
        if current_season is not None and baseline_season is None:
            return current_season - 1, current_season
        if baseline_season is not None and current_season is None:
            return baseline_season, baseline_season + 1
        return now_year - 1, now_year

    async def _load_season_round_metrics(
        self,
        tour: str,
        season: int,
    ) -> _SeasonLoadResult:
        cache_key = (tour.strip().lower(), season)
        cached = self._season_metrics_cache.get(cache_key)
        if cached is not None:
            return _SeasonLoadResult(
                metrics=cached.metrics,
                event_count=cached.event_count,
                successful_event_count=cached.successful_event_count,
                player_metric_count=cached.player_metric_count,
                observation_count=cached.observation_count,
                from_cache=True,
            )

        event_list_payload = await self._datagolf.get_historical_event_list(tour=tour)
        event_descriptors = _historical_event_descriptors(event_list_payload, season=season)
        if not event_descriptors:
            raise DataGolfAPIError(
                f"No historical events found for tour={tour}, season={season}."
            )

        semaphore = asyncio.Semaphore(8)

        async def fetch_event(event_id: str, year: int) -> Any:
            async with semaphore:
                return await self._datagolf.get_historical_event(
                    tour=tour,
                    event_id=event_id,
                    year=year,
                )

        payload_tasks = [fetch_event(event_id, year) for event_id, year in event_descriptors]
        payloads = await asyncio.gather(*payload_tasks, return_exceptions=True)

        per_player: dict[str, dict[str, Any]] = {}
        successful_events = 0
        total_observations = 0
        for payload in payloads:
            if isinstance(payload, Exception):
                continue

            observations = _extract_player_metric_observations(payload)
            if observations:
                successful_events += 1
            total_observations += len(observations)
            for canonical_key, player_id, player_name, metric in observations:
                bucket = per_player.setdefault(
                    canonical_key,
                    {
                        "player_id": player_id,
                        "player_name": player_name,
                        "metrics": [],
                    },
                )
                bucket["metrics"].append(metric)

        out: dict[str, dict[str, Any]] = {}
        for key, values in per_player.items():
            metrics = np.asarray(values["metrics"], dtype=np.float64)
            if metrics.size == 0:
                continue
            out[key] = {
                "player_id": values.get("player_id"),
                "player_name": values.get("player_name"),
                "metric": float(metrics.mean()),
                "rounds": int(metrics.size),
                "volatility": float(metrics.std(ddof=1)) if metrics.size > 1 else None,
            }

        if not out:
            raise DataGolfAPIError(
                f"No usable player round metrics found for tour={tour}, season={season}."
            )

        result = _SeasonLoadResult(
            metrics=out,
            event_count=len(event_descriptors),
            successful_event_count=successful_events,
            player_metric_count=len(out),
            observation_count=total_observations,
            from_cache=False,
        )
        self._season_metrics_cache[cache_key] = result
        return result

    @staticmethod
    def _apply_seasonal_form_metrics(
        players: list[_PlayerRecord],
        baseline_metrics: dict[str, dict[str, Any]],
        current_metrics: dict[str, dict[str, Any]],
    ) -> int:
        baseline_by_name = {
            _normalized_name(str(v.get("player_name"))): v
            for v in baseline_metrics.values()
            if v.get("player_name")
        }
        current_by_name = {
            _normalized_name(str(v.get("player_name"))): v
            for v in current_metrics.values()
            if v.get("player_name")
        }

        applied = 0
        for player in players:
            id_key = _normalized_player_key(player.player_id)
            name_key = _normalized_name(player.player_name)
            base = baseline_metrics.get(id_key) or baseline_by_name.get(name_key)
            curr = current_metrics.get(id_key) or current_by_name.get(name_key)

            if base:
                player.baseline_season_metric = _to_float(base.get("metric"))
                player.baseline_season_rounds = int(base.get("rounds") or 0)
                player.baseline_season_volatility = _to_float(base.get("volatility"))
            if curr:
                player.current_season_metric = _to_float(curr.get("metric"))
                player.current_season_rounds = int(curr.get("rounds") or 0)
                player.current_season_volatility = _to_float(curr.get("volatility"))
            if (
                player.baseline_season_metric is not None
                and player.current_season_metric is not None
            ):
                player.form_delta_metric = (
                    player.current_season_metric - player.baseline_season_metric
                )
            if base or curr:
                applied += 1

        return applied

    @staticmethod
    def _build_simulation_inputs(
        players: list[_PlayerRecord],
        seasonal_form_weight: float = 0.35,
        current_season_weight: float = 0.60,
        form_delta_weight: float = 0.25,
    ) -> SimulationInputs:
        player_ids = [p.player_id for p in players]
        player_names = [p.player_name for p in players]

        raw_skill = np.array(
            [
                p.skill if p.skill is not None else _skill_from_baseline_probabilities(p)
                for p in players
            ],
            dtype=np.float64,
        )

        if np.isnan(raw_skill).all():
            raw_skill = np.zeros_like(raw_skill)
        else:
            skill_mean = np.nanmean(raw_skill)
            raw_skill = np.where(np.isnan(raw_skill), skill_mean, raw_skill)

        normalized_skill = _zscore_with_nan(raw_skill)

        baseline_metric = np.array(
            [
                p.baseline_season_metric
                if p.baseline_season_metric is not None
                else np.nan
                for p in players
            ],
            dtype=np.float64,
        )
        current_metric = np.array(
            [
                p.current_season_metric
                if p.current_season_metric is not None
                else np.nan
                for p in players
            ],
            dtype=np.float64,
        )
        baseline_rounds = np.array(
            [max(0, p.baseline_season_rounds) for p in players], dtype=np.float64
        )
        current_rounds = np.array(
            [max(0, p.current_season_rounds) for p in players], dtype=np.float64
        )

        baseline_z = _zscore_with_nan(baseline_metric)
        current_z = _zscore_with_nan(current_metric)

        has_baseline = ~np.isnan(baseline_z)
        has_current = ~np.isnan(current_z)

        baseline_signal = np.where(has_baseline, baseline_z, 0.0)
        current_signal = np.where(has_current, current_z, baseline_signal)

        current_reliability = np.clip(current_rounds / 24.0, 0.0, 1.0)
        baseline_reliability = np.clip(baseline_rounds / 24.0, 0.0, 1.0)

        current_weight = np.where(
            has_current,
            np.clip(current_season_weight, 0.0, 1.0) * current_reliability,
            0.0,
        )
        baseline_weight = np.where(
            has_baseline,
            (1.0 - np.clip(current_season_weight, 0.0, 1.0)) * baseline_reliability,
            0.0,
        )
        weight_sum = current_weight + baseline_weight

        weighted_signal = (current_weight * current_signal) + (baseline_weight * baseline_signal)
        seasonal_anchor = np.divide(
            weighted_signal,
            weight_sum,
            out=np.zeros_like(weight_sum, dtype=np.float64),
            where=weight_sum > 1e-8,
        )
        seasonal_delta = np.where(has_current & has_baseline, current_signal - baseline_signal, 0.0)
        seasonal_signal = seasonal_anchor + (
            np.clip(form_delta_weight, 0.0, 1.0) * seasonal_delta
        )

        blended_skill = np.where(
            weight_sum > 1e-8,
            (1.0 - np.clip(seasonal_form_weight, 0.0, 1.0)) * normalized_skill
            + np.clip(seasonal_form_weight, 0.0, 1.0) * seasonal_signal,
            normalized_skill,
        )

        # Better players get lower expected round deltas (negative is better).
        mu_round = -0.85 * blended_skill

        sigma_round = np.array(
            [p.sigma if p.sigma is not None else np.nan for p in players], dtype=np.float64
        )
        if np.isnan(sigma_round).all():
            sigma_round = 2.7 - (0.10 * blended_skill)
        else:
            fallback = 2.7 - (0.10 * blended_skill)
            sigma_round = np.where(np.isnan(sigma_round), fallback, sigma_round)

        sigma_round = np.clip(sigma_round, 2.0, 3.8)

        return SimulationInputs(
            player_ids=player_ids,
            player_names=player_names,
            mu_round=mu_round,
            sigma_round=sigma_round,
        )


def _extract_rows(payload: Any, preferred_terms: tuple[str, ...]) -> list[dict[str, Any]]:
    candidates: list[tuple[tuple[str, ...], list[dict[str, Any]]]] = []

    def walk(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, list):
            dict_rows = [row for row in node if isinstance(row, dict)]
            if dict_rows and len(dict_rows) == len(node):
                candidates.append((path, dict_rows))
            return
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, path + (str(key).lower(),))

    walk(payload, tuple())

    if not candidates:
        return []

    def score(candidate: tuple[tuple[str, ...], list[dict[str, Any]]]) -> tuple[int, int]:
        path, rows = candidate
        path_score = sum(4 for term in preferred_terms if any(term in p for p in path))
        sample = rows[0]
        row_score = 0
        if any("player" in str(k).lower() for k in sample.keys()):
            row_score += 2
        if any("event" in str(k).lower() for k in sample.keys()):
            row_score += 1
        return path_score + row_score, len(rows)

    best = max(candidates, key=score)
    return best[1]


def _string_from_payload(payload: Any, keys: tuple[str, ...]) -> str | None:
    if isinstance(payload, dict):
        value = _string_from_keys(payload, keys)
        if value:
            return value
        for nested in payload.values():
            value = _string_from_payload(nested, keys)
            if value:
                return value
    elif isinstance(payload, list):
        for item in payload:
            value = _string_from_payload(item, keys)
            if value:
                return value
    return None


def _player_identity(row: dict[str, Any]) -> tuple[str | None, str, str]:
    player_id = _string_from_keys(row, _ID_KEYS)
    player_name = _string_from_keys(row, _NAME_KEYS)
    if player_name and player_name.strip().startswith(("{", "[")):
        player_name = None
    if player_id and player_id.strip().startswith(("{", "[")):
        player_id = None
    if not player_name:
        return None, "", ""
    if not player_id:
        player_id = player_name.lower().replace(" ", "-")
    key = player_id or player_name.lower()
    return key, player_id, player_name


def _resolve_player_key(
    key: str,
    player_name: str,
    merged: dict[str, _PlayerRecord],
    name_index: dict[str, str],
) -> str:
    if key in merged:
        return key
    normalized_name = _normalized_name(player_name)
    if normalized_name in name_index:
        return name_index[normalized_name]
    return key


def _normalized_name(player_name: str) -> str:
    cleaned = " ".join(player_name.strip().lower().replace(".", "").split())
    if "," in cleaned:
        parts = [part.strip() for part in cleaned.split(",") if part.strip()]
        if len(parts) >= 2:
            # Convert "last, first" into "first last" for cross-feed matching.
            reordered = " ".join(parts[1:] + [parts[0]])
            return " ".join(reordered.split())
    return cleaned


def _normalized_event_id(event_id: str | None) -> str | None:
    if event_id is None:
        return None
    normalized = "".join(event_id.strip().lower().split())
    return normalized or None


def _season_from_row(row: dict[str, Any]) -> int | None:
    year_text = _string_from_keys(row, ("year", "season", "calendar_year"))
    if year_text:
        parsed = _parse_season_value(year_text)
        if parsed is not None:
            return parsed

    date_text = _string_from_keys(row, _DATE_KEYS)
    if date_text and len(date_text) >= 4:
        prefix = date_text[:4]
        if prefix.isdigit():
            return int(prefix)
    return None


def _historical_event_descriptors(
    event_list_payload: Any,
    season: int,
    limit: int = 72,
) -> list[tuple[str, int]]:
    rows = _extract_rows(event_list_payload, ("event", "list", "historical", "schedule"))
    if not rows:
        rows = _collect_dict_nodes(event_list_payload)

    descriptors: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        event_id = _string_from_keys(row, _EVENT_ID_KEYS)
        if not event_id:
            continue
        year = _season_from_row(row) or season
        if year != season:
            continue
        key = (event_id, year)
        if key in seen:
            continue
        seen.add(key)
        descriptors.append(key)
        if len(descriptors) >= limit:
            break
    return descriptors


def _normalized_player_key(player_key: str | None) -> str | None:
    if player_key is None:
        return None
    normalized = "".join(player_key.strip().lower().split())
    return normalized or None


def _historical_player_identity(row: dict[str, Any]) -> tuple[str | None, str, str]:
    key, player_id, player_name = _player_identity(row)
    if key and player_name:
        normalized = _normalized_player_key(key)
        return normalized, player_id, player_name

    lowered = {str(k).lower(): v for k, v in row.items()}
    for carrier in ("player", "golfer", "competitor"):
        nested = lowered.get(carrier)
        if not isinstance(nested, dict):
            continue
        nested_id = _string_from_keys(nested, _ID_KEYS + ("dg_id",))
        nested_name = _string_from_keys(
            nested,
            _NAME_KEYS + ("full_name", "display_name", "player_display_name"),
        )
        if not nested_name:
            first = _string_from_keys(nested, ("first_name", "firstname"))
            last = _string_from_keys(nested, ("last_name", "lastname"))
            combined = " ".join(part for part in (first, last) if part).strip()
            nested_name = combined or None
        if not nested_name:
            continue
        if not nested_id:
            nested_id = nested_name.lower().replace(" ", "-")
        canonical_key = _normalized_player_key(nested_id or nested_name)
        return canonical_key, nested_id, nested_name

    first = _string_from_keys(row, ("first_name", "firstname"))
    last = _string_from_keys(row, ("last_name", "lastname"))
    combined = " ".join(part for part in (first, last) if part).strip()
    if combined:
        inferred_id = combined.lower().replace(" ", "-")
        return _normalized_player_key(inferred_id), inferred_id, combined

    return None, "", ""


def _collect_dict_nodes(payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            rows.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    return rows


def _metrics_from_nested_list(value: Any) -> list[float]:
    out: list[float] = []
    if isinstance(value, list):
        for item in value:
            out.extend(_metrics_from_nested_list(item))
    elif isinstance(value, dict):
        metric = _round_metric_from_row(value)
        if metric is not None:
            out.append(metric)
        for nested in value.values():
            if isinstance(nested, (list, dict)):
                out.extend(_metrics_from_nested_list(nested))
    return out


def _extract_player_metric_observations(payload: Any) -> list[tuple[str, str, str, float]]:
    observations: list[tuple[str, str, str, float]] = []
    direct_metric_players: set[str] = set()
    for row in _collect_dict_nodes(payload):
        canonical_key, player_id, player_name = _historical_player_identity(row)
        if not canonical_key:
            continue

        nested_metrics: list[float] = []
        for value in row.values():
            if isinstance(value, (list, dict)):
                nested_metrics.extend(_metrics_from_nested_list(value))

        if nested_metrics:
            for metric in nested_metrics:
                observations.append((canonical_key, player_id, player_name, metric))
            direct_metric_players.add(canonical_key)
            continue

        direct_metric = _round_metric_from_row(row)
        if direct_metric is not None:
            observations.append((canonical_key, player_id, player_name, direct_metric))
            direct_metric_players.add(canonical_key)
            continue

        event_metric = _event_level_metric_from_row(row)
        if event_metric is not None:
            observations.append((canonical_key, player_id, player_name, event_metric))
            direct_metric_players.add(canonical_key)

    score_observations = _extract_score_array_observations(
        payload,
        exclude_players=direct_metric_players,
    )
    observations.extend(score_observations)

    return observations


def _extract_score_array_observations(
    payload: Any,
    exclude_players: set[str],
) -> list[tuple[str, str, str, float]]:
    player_scores: list[tuple[str, str, str, list[float]]] = []
    for row in _collect_dict_nodes(payload):
        canonical_key, player_id, player_name = _historical_player_identity(row)
        if not canonical_key or canonical_key in exclude_players:
            continue
        scores = _round_scores_from_row(row)
        if not scores:
            continue
        player_scores.append((canonical_key, player_id, player_name, scores))

    if not player_scores:
        return []

    max_rounds = max(len(scores) for _, _, _, scores in player_scores)
    round_means: list[float] = []
    for round_idx in range(max_rounds):
        values = [scores[round_idx] for _, _, _, scores in player_scores if round_idx < len(scores)]
        if not values:
            round_means.append(float("nan"))
            continue
        round_means.append(float(np.mean(values)))

    observations: list[tuple[str, str, str, float]] = []
    for canonical_key, player_id, player_name, scores in player_scores:
        for round_idx, score in enumerate(scores):
            baseline = round_means[round_idx]
            if np.isnan(baseline):
                continue
            # Lower score than round field mean is positive form.
            metric = -(score - baseline)
            observations.append((canonical_key, player_id, player_name, metric))
    return observations


def _round_scores_from_row(row: dict[str, Any]) -> list[float]:
    lowered = {str(k).lower(): v for k, v in row.items()}

    key_candidates = (
        "round_scores",
        "scores",
        "rounds",
        "round_scores_raw",
        "strokes_by_round",
        "scores_by_round",
    )
    for key in key_candidates:
        if key not in lowered:
            continue
        extracted = _numeric_sequence_from_value(lowered[key])
        if extracted:
            return extracted

    # Common flattened shapes like r1/r2/r3/r4.
    round_keys = (
        "r1",
        "r2",
        "r3",
        "r4",
        "round1",
        "round2",
        "round3",
        "round4",
        "score_r1",
        "score_r2",
        "score_r3",
        "score_r4",
    )
    flattened: list[float] = []
    for key in round_keys:
        value = _to_float(lowered.get(key))
        if value is not None:
            flattened.append(value)
    return flattened


def _event_level_metric_from_row(row: dict[str, Any]) -> float | None:
    dg_points = _numeric_from_keys(row, ("dg_points",))
    if dg_points is not None:
        return dg_points

    fec_points = _numeric_from_keys(row, ("fec_points", "fedex_points"))
    if fec_points is not None:
        # Typical event winner is 500-750 points.
        return fec_points / 25.0

    earnings = _numeric_from_keys(row, ("earnings", "prize_money"))
    if earnings is not None and earnings >= 0:
        return float(np.log1p(earnings) / 2.0)

    fin_text = _string_from_keys(row, ("fin_text", "finish", "position", "pos"))
    if fin_text:
        rank = _finish_text_to_rank(fin_text)
        if rank is not None:
            return -float(rank)
    return None


def _numeric_sequence_from_value(value: Any) -> list[float]:
    out: list[float] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (int, float)):
                out.append(float(item))
                continue
            if isinstance(item, str):
                number = _to_float(item)
                if number is not None:
                    out.append(number)
                continue
            if isinstance(item, dict):
                score = _numeric_from_keys(
                    item,
                    (
                        "score",
                        "round_score",
                        "strokes",
                        "gross_score",
                        "adj_score",
                        "round_total",
                    ),
                )
                if score is not None:
                    out.append(score)
    elif isinstance(value, dict):
        nested = _numeric_sequence_from_value(list(value.values()))
        out.extend(nested)
    return out


def _finish_text_to_rank(fin_text: str) -> int | None:
    normalized = fin_text.strip().upper()
    if not normalized:
        return None
    if normalized.startswith("T"):
        normalized = normalized[1:]
    if normalized.isdigit():
        return int(normalized)
    return None


def _parse_season_value(value: str) -> int | None:
    text = value.strip()
    try:
        parsed = int(float(text))
        if 1900 <= parsed <= 2100:
            return parsed
    except (TypeError, ValueError):
        pass

    mixed = re.search(r"(19\d{2}|20\d{2}|21\d{2})\D+(\d{2})\b", text)
    if mixed:
        start_year = int(mixed.group(1))
        end_two = int(mixed.group(2))
        century = (start_year // 100) * 100
        candidate = century + end_two
        if candidate < start_year:
            candidate += 100
        if 1900 <= candidate <= 2100:
            return candidate

    years = [int(match) for match in re.findall(r"(19\d{2}|20\d{2}|21\d{2})", text)]
    if years:
        return max(years)

    compact = re.findall(r"\b(\d{2})\b", text)
    if len(compact) >= 2:
        yr = int(compact[-1])
        return 2000 + yr if yr <= 79 else 1900 + yr
    return None


def _string_from_keys(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    lowered = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        if key in lowered:
            value = lowered[key]
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                continue
            text = str(value).strip()
            if text:
                return text
    return None


def _numeric_from_keys(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    lowered = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        if key in lowered:
            value = _to_float(lowered[key])
            if value is not None:
                return value
    return None


def _probability_from_keys(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    value = _numeric_from_keys(row, keys)
    if value is None:
        return None
    if value > 1.0:
        value /= 100.0
    if value < 0.0 or value > 1.0:
        return None
    return value


def _skill_from_row(row: dict[str, Any]) -> float | None:
    direct = _numeric_from_keys(row, _SKILL_KEYS)
    if direct is not None:
        return direct

    component_keys = ("driving", "approach", "around_green", "putting", "ott", "app", "atg", "putt")
    lowered = {str(k).lower(): v for k, v in row.items()}
    components: list[float] = []
    for key in component_keys:
        if key in lowered:
            val = _to_float(lowered[key])
            if val is not None:
                components.append(val)
    if components:
        return float(sum(components))
    return None


def _skill_from_baseline_probabilities(player: _PlayerRecord) -> float:
    normal = NormalDist()
    if player.baseline_win_probability:
        prob = min(max(player.baseline_win_probability, 1e-5), 0.6)
        return normal.inv_cdf(prob)
    if player.baseline_top_10_probability:
        prob = min(max(player.baseline_top_10_probability, 1e-4), 0.98)
        return 0.65 * normal.inv_cdf(prob)
    return np.nan


def _round_metric_from_row(row: dict[str, Any]) -> float | None:
    lowered = {str(k).lower(): v for k, v in row.items()}
    for key, direction in _ROUND_METRIC_RULES:
        if key not in lowered:
            continue
        value = _to_float(lowered[key])
        if value is None:
            continue
        return direction * value

    round_score = _to_float(lowered.get("round_score"))
    field_avg = _to_float(lowered.get("field_avg"))
    if round_score is not None and field_avg is not None:
        # Lower score than the field average should increase player skill metric.
        return -(round_score - field_avg)
    return None


def _zscore_with_nan(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64)
    mask = ~np.isnan(values)
    if not mask.any():
        return np.zeros_like(values, dtype=np.float64)
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if std <= 1e-8:
        out = np.zeros_like(values, dtype=np.float64)
        return np.where(mask, out, np.nan)
    out = (values - mean) / std
    return np.where(mask, out, np.nan)


def _safe_date_sort_key(date_str: str | None) -> datetime:
    if not date_str:
        return datetime.max.replace(tzinfo=timezone.utc)
    text = date_str.strip()
    formats = ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S")
    for fmt in formats:
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime.max.replace(tzinfo=timezone.utc)


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
