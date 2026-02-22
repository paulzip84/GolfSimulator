from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from statistics import NormalDist
from typing import Any

import numpy as np

from .datagolf_client import DataGolfAPIError, DataGolfClient
from .learning import CalibrationMetrics, LearningStore
from .models import (
    CalibrationMarketStatus,
    EventSummary,
    LearningEventTrendsResponse,
    LearningStatusResponse,
    LearningSyncResponse,
    PlayerSimulationOutput,
    SimulationRequest,
    SimulationResponse,
)
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
_LIVE_POSITION_KEYS = (
    "position",
    "current_position",
    "pos",
    "current_pos",
    "rank",
    "place",
    "leaderboard_position",
)
_LIVE_SCORE_TO_PAR_KEYS = (
    "score_to_par",
    "total_to_par",
    "total_score_to_par",
    "event_score_to_par",
    "tournament_score_to_par",
    "event_to_par",
    "overall_score_to_par",
    "overall_to_par",
    "cum_to_par",
)
_LIVE_SCORE_TO_PAR_FALLBACK_KEYS = ("to_par", "tot")
_LIVE_SCORE_NESTED_CONTAINERS = (
    "status",
    "leaderboard",
    "scoring",
    "live",
    "current",
    "event",
    "tournament",
    "totals",
)
_LIVE_THRU_KEYS = (
    "thru",
    "through",
    "holes_completed",
    "holes_played",
    "current_hole",
)
_LIVE_TODAY_KEYS = (
    "today",
    "today_to_par",
    "round_score_to_par",
    "round_to_par",
    "current_round_to_par",
)
_LIVE_ROUND_SCORE_KEYS = (
    "round_scores",
    "scores_by_round",
    "strokes_by_round",
    "round_scorecard",
    "scorecard_rounds",
)
_LIVE_HOLE_SCORE_KEYS = (
    "hole_scores",
    "scores_by_hole",
    "strokes_by_hole",
    "hole_by_hole",
    "scorecard_holes",
)
_ROUND_SCORE_FLAT_KEYS = (
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
_HOLE_SCORE_FLAT_KEYS = tuple(
    [f"h{idx}" for idx in range(1, 19)]
    + [f"hole{idx}" for idx in range(1, 19)]
    + [f"hole_{idx}" for idx in range(1, 19)]
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
    baseline_phase_metric: float | None = None
    current_season_metric: float | None = None
    current_recent_metric: float | None = None
    current_recent_finish_score: float | None = None
    form_delta_metric: float | None = None
    baseline_season_rounds: int = 0
    current_season_rounds: int = 0
    baseline_season_starts: int = 0
    current_season_starts: int = 0
    baseline_season_volatility: float | None = None
    current_season_volatility: float | None = None
    baseline_hot_streak_score: float | None = None
    dynamic_current_weight: float | None = None
    current_position: str | None = None
    current_score_to_par: float | None = None
    current_thru: str | None = None
    today_score_to_par: float | None = None
    round_scores: list[float] = field(default_factory=list)
    hole_scores: list[int] = field(default_factory=list)


@dataclass
class _LiveScoreSnapshot:
    current_position: str | None = None
    current_score_to_par: float | None = None
    current_thru: str | None = None
    today_score_to_par: float | None = None
    round_scores: list[float] = field(default_factory=list)
    hole_scores: list[int] = field(default_factory=list)


@dataclass
class _SeasonLoadResult:
    metrics: dict[str, dict[str, Any]]
    event_count: int
    successful_event_count: int
    player_metric_count: int
    observation_count: int
    season_phase_hint: float | None = None
    from_cache: bool = False


@dataclass
class _InPlayConditioningResult:
    applied: bool
    note: str | None
    initial_totals: np.ndarray
    round_fractions: np.ndarray
    round_numbers: np.ndarray


class SimulationService:
    def __init__(self, datagolf: DataGolfClient, learning_store: LearningStore | None = None):
        self._datagolf = datagolf
        self._learning = learning_store
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
        live_payload: Any = {}
        get_in_play = getattr(self._datagolf, "get_in_play", None)
        if callable(get_in_play):
            try:
                live_payload = await get_in_play(
                    tour=request.tour,
                    odds_format="percent",
                )
            except DataGolfAPIError:
                live_payload = {}

        selected_event_id = _normalized_event_id(request.event_id)
        active_event_id = _normalized_event_id(_string_from_payload(field_payload, _EVENT_ID_KEYS))
        if selected_event_id and active_event_id and selected_event_id != active_event_id:
            active_event_name = _string_from_payload(field_payload, _EVENT_NAME_KEYS) or "Unknown"
            raise ValueError(
                "Selected event is not available for this tour in DataGolf current-week simulation "
                f"feeds. Active event: {active_event_name} (event_id={active_event_id})."
            )

        event_date = (
            _string_from_payload(field_payload, _DATE_KEYS)
            or _string_from_payload(pre_payload, _DATE_KEYS)
        )
        season_phase = _season_phase_from_date(event_date)

        field_rows = _extract_rows(field_payload, ("field", "player"))
        live_rows = _extract_rows(live_payload, ("in-play", "player", "pred"))
        pre_rows = _extract_rows(pre_payload, ("pred", "tournament", "player"))
        decomp_rows = _extract_rows(decomp_payload, ("decomposition", "player"))

        players = self._merge_player_records(field_rows, pre_rows, decomp_rows, live_rows)
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
                    season_phase=(
                        season_phase
                        if season_phase is not None
                        else current_result.season_phase_hint
                    ),
                )
                form_adjustment_applied = applied_count > 0
                summary = (
                    f"Seasonal data loaded: baseline {baseline_season} -> "
                    f"{baseline_result.player_metric_count} players "
                    f"({baseline_result.successful_event_count}/{baseline_result.event_count} events), "
                    f"current {current_season} -> {current_result.player_metric_count} players "
                    f"({current_result.successful_event_count}/{current_result.event_count} events). "
                    f"Matched active field players: {applied_count}. "
                    f"Season phase={((season_phase if season_phase is not None else current_result.season_phase_hint) or 0.5):.2f}."
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

        in_play_context = self._build_in_play_conditioning_context(
            players=players,
            total_rounds=4,
            enable=request.enable_in_play_conditioning,
        )

        model_inputs = self._build_simulation_inputs(
            players=players,
            base_mean_reversion=request.mean_reversion,
            seasonal_form_weight=request.seasonal_form_weight,
            current_season_weight=request.current_season_weight,
            form_delta_weight=request.form_delta_weight,
            initial_totals=in_play_context.initial_totals,
            round_fractions=in_play_context.round_fractions,
            round_numbers=in_play_context.round_numbers,
        )
        simulator = HybridMarkovSimulator(
            MarkovSimulationConfig(
                mean_reversion=request.mean_reversion,
                round_shock_sigma=request.shared_round_shock_sigma,
            )
        )
        resolution_mode = (request.resolution_mode or "auto_target").strip().lower()
        adaptive_enabled = bool(
            request.enable_adaptive_simulation and resolution_mode != "fixed_cap"
        )
        outputs = simulator.simulate(
            inputs=model_inputs,
            n_simulations=request.simulations,
            seed=request.seed,
            cut_size=request.cut_size,
            adaptive=adaptive_enabled,
            min_simulations=request.min_simulations,
            batch_size=request.simulation_batch_size,
            ci_confidence=request.ci_confidence,
            ci_half_width_target=request.ci_half_width_target,
            ci_top_n=request.ci_top_n,
        )

        event_id = (
            request.event_id
            or _string_from_payload(field_payload, _EVENT_ID_KEYS)
            or _string_from_payload(pre_payload, _EVENT_ID_KEYS)
        )
        event_name = (
            _string_from_payload(field_payload, _EVENT_NAME_KEYS)
            or _string_from_payload(pre_payload, _EVENT_NAME_KEYS)
        )

        raw_win_probability = outputs.win_probability.astype(np.float64, copy=True)
        raw_top_3_probability = outputs.top_3_probability.astype(np.float64, copy=True)
        raw_top_5_probability = outputs.top_5_probability.astype(np.float64, copy=True)
        raw_top_10_probability = outputs.top_10_probability.astype(np.float64, copy=True)

        display_win_probability = raw_win_probability
        display_top_3_probability = raw_top_3_probability
        display_top_5_probability = raw_top_5_probability
        display_top_10_probability = raw_top_10_probability

        calibration_applied = False
        calibration_version = 0
        calibration_note = "Learning calibration not configured."
        if self._learning is not None:
            snapshot = self._learning.get_snapshot(tour=request.tour)
            calibration_version = snapshot.version
            if snapshot.version > 0 and snapshot.markets:
                display_win_probability = snapshot.apply("win", raw_win_probability)
                win_total = float(display_win_probability.sum())
                if win_total > 1e-12:
                    display_win_probability = display_win_probability / win_total
                else:
                    display_win_probability = raw_win_probability

                display_top_3_probability = snapshot.apply("top_3", raw_top_3_probability)
                display_top_5_probability = snapshot.apply("top_5", raw_top_5_probability)
                display_top_10_probability = snapshot.apply("top_10", raw_top_10_probability)

                display_top_3_probability = np.maximum(
                    display_top_3_probability,
                    display_win_probability,
                )
                display_top_5_probability = np.maximum(
                    display_top_5_probability,
                    display_top_3_probability,
                )
                display_top_10_probability = np.maximum(
                    display_top_10_probability,
                    display_top_5_probability,
                )
                display_top_3_probability = np.clip(display_top_3_probability, 0.0, 1.0)
                display_top_5_probability = np.clip(display_top_5_probability, 0.0, 1.0)
                display_top_10_probability = np.clip(display_top_10_probability, 0.0, 1.0)
                calibration_applied = True
                calibration_note = (
                    f"Applied learning calibration v{snapshot.version} from historical outcomes."
                )
            elif snapshot.version <= 0:
                calibration_note = (
                    "Learning calibration is not trained yet. Run outcome sync + retrain."
                )
            else:
                calibration_note = (
                    "Learning calibration loaded, but no per-market parameters were available."
                )

        rankings = np.argsort(display_win_probability)[::-1]
        result_rows: list[PlayerSimulationOutput] = []
        for idx in rankings:
            record = players[idx]
            result_rows.append(
                PlayerSimulationOutput(
                    player_id=record.player_id,
                    player_name=record.player_name,
                    win_probability=float(display_win_probability[idx]),
                    top_3_probability=float(display_top_3_probability[idx]),
                    top_5_probability=float(display_top_5_probability[idx]),
                    top_10_probability=float(display_top_10_probability[idx]),
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
                    current_position=record.current_position,
                    current_score_to_par=record.current_score_to_par,
                    current_thru=record.current_thru,
                    today_score_to_par=record.today_score_to_par,
                    round_scores=record.round_scores,
                    hole_scores=record.hole_scores,
                )
            )

        response = SimulationResponse(
            generated_at=datetime.now(timezone.utc),
            tour=request.tour,
            event_id=event_id,
            event_name=event_name,
            simulations=outputs.simulations_run,
            requested_simulations=request.simulations,
            adaptive_stopped_early=outputs.adaptive_stopped_early,
            win_ci_half_width_top_n=outputs.win_ci_half_width_top_n,
            ci_target_met=outputs.ci_target_met,
            stop_reason=outputs.stop_reason,
            recommended_simulations=outputs.recommended_simulations,
            in_play_conditioning_applied=in_play_context.applied,
            in_play_conditioning_note=in_play_context.note,
            baseline_season=baseline_season if request.enable_seasonal_form else None,
            current_season=current_season if request.enable_seasonal_form else None,
            form_adjustment_applied=form_adjustment_applied,
            form_adjustment_note=form_adjustment_note,
            calibration_applied=calibration_applied,
            calibration_version=calibration_version,
            calibration_note=calibration_note,
            players=result_rows,
        )
        self._record_learning_prediction(
            request=request,
            response=response,
            players=players,
            raw_win_probability=raw_win_probability,
            raw_top_3_probability=raw_top_3_probability,
            raw_top_5_probability=raw_top_5_probability,
            raw_top_10_probability=raw_top_10_probability,
            event_date=event_date,
            in_play_applied=in_play_context.applied,
        )
        return response

    async def get_learning_status(self, tour: str = "pga") -> LearningStatusResponse:
        if self._learning is None:
            return LearningStatusResponse(tour=tour.strip().lower())
        status = self._learning.status(tour=tour)
        return LearningStatusResponse(
            tour=status["tour"],
            predictions_logged=int(status["predictions_logged"]),
            resolved_predictions=int(status["resolved_predictions"]),
            resolved_events=int(status["resolved_events"]),
            pending_events=int(status["pending_events"]),
            calibration_version=int(status["calibration_version"]),
            calibration_updated_at=status["calibration_updated_at"],
            markets=self._market_status_models(status["markets"]),
        )

    async def get_learning_event_trends(
        self,
        *,
        tour: str = "pga",
        event_id: str,
        event_year: int | None = None,
        max_snapshots: int = 80,
        max_players: int = 40,
    ) -> LearningEventTrendsResponse:
        if self._learning is None:
            raise ValueError("Learning store is not configured.")
        payload = self._learning.event_trends(
            tour=tour,
            event_id=event_id,
            event_year=event_year,
            max_snapshots=max_snapshots,
            max_players=max_players,
        )
        return LearningEventTrendsResponse(**payload)

    async def sync_learning_and_retrain(
        self,
        tour: str = "pga",
        max_events: int = 40,
    ) -> LearningSyncResponse:
        if self._learning is None:
            raise ValueError("Learning store is not configured.")

        baseline_snapshot = self._learning.get_snapshot(tour=tour)
        baseline_status = self._learning.status(tour=tour)
        pending_events = self._learning.list_pending_events(tour=tour, max_events=max_events)
        processed_event_ids: list[str] = []
        awaiting_outcomes_event_ids: list[str] = []
        outcomes_fetched = 0

        def event_label(event: Any) -> str:
            name = str(event.event_name).strip() if getattr(event, "event_name", None) else ""
            base = f"{event.event_id}:{event.event_year}"
            return f"{name} ({base})" if name else base

        for event in pending_events:
            try:
                payload = await self._datagolf.get_historical_event(
                    tour=event.tour,
                    event_id=event.event_id,
                    year=event.event_year,
                )
            except DataGolfAPIError:
                awaiting_outcomes_event_ids.append(event_label(event))
                continue
            outcome_rows = self._learning.record_outcome_payload(
                tour=event.tour,
                event_id=event.event_id,
                event_year=event.event_year,
                payload=payload,
            )
            if outcome_rows > 0:
                outcomes_fetched += 1
                processed_event_ids.append(f"{event.event_id}:{event.event_year}")
            else:
                awaiting_outcomes_event_ids.append(event_label(event))

        retrain_executed = False
        should_retrain = outcomes_fetched > 0 or (
            int(baseline_snapshot.version) <= 0
            and int(baseline_status.get("resolved_predictions") or 0) > 0
        )
        if should_retrain:
            self._learning.retrain(tour=tour, bump_version=True)
            retrain_executed = True

        status = self._learning.status(tour=tour)
        current_version = int(status["calibration_version"])
        previous_version = int(baseline_snapshot.version)

        note_parts: list[str] = []
        if outcomes_fetched > 0:
            note_parts.append(f"Fetched outcomes for {outcomes_fetched} events.")
        else:
            note_parts.append("No new outcomes fetched.")

        if awaiting_outcomes_event_ids:
            preview = ", ".join(awaiting_outcomes_event_ids[:4])
            if len(awaiting_outcomes_event_ids) > 4:
                preview += ", ..."
            note_parts.append(
                "Awaiting official historical outcomes for "
                f"{len(awaiting_outcomes_event_ids)} event(s): {preview}"
            )

        if retrain_executed and current_version > previous_version:
            note_parts.append(
                f"Retrain complete: Learning v{previous_version} -> v{current_version}."
            )
        elif retrain_executed:
            note_parts.append(
                f"Retrain ran with existing calibration version v{current_version}."
            )
        else:
            note_parts.append(
                f"Retrain skipped: Learning version remains v{current_version} (no new resolved outcomes)."
            )
        note = " ".join(note_parts)

        return LearningSyncResponse(
            tour=status["tour"],
            predictions_logged=int(status["predictions_logged"]),
            resolved_predictions=int(status["resolved_predictions"]),
            resolved_events=int(status["resolved_events"]),
            pending_events=int(status["pending_events"]),
            calibration_version=int(status["calibration_version"]),
            calibration_updated_at=status["calibration_updated_at"],
            markets=self._market_status_models(status["markets"]),
            outcomes_fetched=outcomes_fetched,
            events_processed=len(pending_events),
            event_ids_processed=processed_event_ids,
            awaiting_outcomes_count=len(awaiting_outcomes_event_ids),
            awaiting_outcomes_event_ids=awaiting_outcomes_event_ids,
            retrain_executed=retrain_executed,
            sync_note=note,
        )

    @staticmethod
    def _market_status_models(
        metrics: list[CalibrationMetrics],
    ) -> list[CalibrationMarketStatus]:
        out: list[CalibrationMarketStatus] = []
        for metric in metrics:
            out.append(
                CalibrationMarketStatus(
                    market=metric.market,
                    alpha=metric.alpha,
                    beta=metric.beta,
                    samples=metric.samples,
                    positives=metric.positives,
                    brier_before=metric.brier_before,
                    brier_after=metric.brier_after,
                    logloss_before=metric.logloss_before,
                    logloss_after=metric.logloss_after,
                )
            )
        return out

    def _record_learning_prediction(
        self,
        *,
        request: SimulationRequest,
        response: SimulationResponse,
        players: list[_PlayerRecord],
        raw_win_probability: np.ndarray,
        raw_top_3_probability: np.ndarray,
        raw_top_5_probability: np.ndarray,
        raw_top_10_probability: np.ndarray,
        event_date: str | None,
        in_play_applied: bool,
    ) -> None:
        if self._learning is None:
            return
        try:
            player_rows: list[dict[str, Any]] = []
            for idx, record in enumerate(players):
                player_rows.append(
                    {
                        "player_id": record.player_id,
                        "player_name": record.player_name,
                        "win_probability": float(raw_win_probability[idx]),
                        "top_3_probability": float(raw_top_3_probability[idx]),
                        "top_5_probability": float(raw_top_5_probability[idx]),
                        "top_10_probability": float(raw_top_10_probability[idx]),
                    }
                )
            self._learning.record_prediction(
                tour=response.tour,
                event_id=response.event_id,
                event_name=response.event_name,
                event_date=event_date,
                requested_simulations=response.requested_simulations,
                simulations=response.simulations,
                enable_in_play=request.enable_in_play_conditioning,
                in_play_applied=in_play_applied,
                players=player_rows,
            )
        except Exception:
            # Never block simulation responses on local learning persistence.
            return

    def _merge_player_records(
        self,
        field_rows: list[dict[str, Any]],
        pre_rows: list[dict[str, Any]],
        decomp_rows: list[dict[str, Any]],
        live_rows: list[dict[str, Any]] | None = None,
    ) -> list[_PlayerRecord]:
        merged: dict[str, _PlayerRecord] = {}
        name_index: dict[str, str] = {}

        for row in field_rows:
            key, player_id, player_name = _player_identity(row)
            if not key:
                continue
            resolved_key = _resolve_player_key(key, player_name, merged, name_index)
            live_snapshot = _extract_live_score_snapshot(row)
            record = merged.get(resolved_key)
            if record is None:
                record = _PlayerRecord(player_id=player_id, player_name=player_name)
                merged[resolved_key] = record
            else:
                if not record.player_id and player_id:
                    record.player_id = player_id
                if not record.player_name and player_name:
                    record.player_name = player_name

            if live_snapshot.current_position is not None:
                record.current_position = live_snapshot.current_position
            if live_snapshot.current_score_to_par is not None:
                record.current_score_to_par = live_snapshot.current_score_to_par
            if live_snapshot.current_thru is not None:
                record.current_thru = live_snapshot.current_thru
            if live_snapshot.today_score_to_par is not None:
                record.today_score_to_par = live_snapshot.today_score_to_par
            if live_snapshot.round_scores:
                record.round_scores = live_snapshot.round_scores
            if live_snapshot.hole_scores:
                record.hole_scores = live_snapshot.hole_scores
            name_index[_normalized_name(player_name)] = resolved_key

        for row in live_rows or []:
            key, player_id, player_name = _player_identity(row)
            if not key:
                continue
            resolved_key = _resolve_player_key(key, player_name, merged, name_index)
            record = merged.setdefault(
                resolved_key,
                _PlayerRecord(player_id=player_id, player_name=player_name),
            )
            if not record.player_id and player_id:
                record.player_id = player_id
            if not record.player_name and player_name:
                record.player_name = player_name
            live_snapshot = _extract_live_score_snapshot(row)
            if live_snapshot.current_position is not None:
                record.current_position = live_snapshot.current_position
            if live_snapshot.current_score_to_par is not None:
                record.current_score_to_par = live_snapshot.current_score_to_par
            if live_snapshot.current_thru is not None:
                record.current_thru = live_snapshot.current_thru
            if live_snapshot.today_score_to_par is not None:
                record.today_score_to_par = live_snapshot.today_score_to_par
            if live_snapshot.round_scores:
                record.round_scores = live_snapshot.round_scores
            if live_snapshot.hole_scores:
                record.hole_scores = live_snapshot.hole_scores
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

        records = list(merged.values())
        _reconcile_merged_live_scores(records)
        return records

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
                season_phase_hint=cached.season_phase_hint,
                from_cache=True,
            )

        event_list_payload = await self._datagolf.get_historical_event_list(tour=tour)
        event_records = _historical_event_records(event_list_payload, season=season)
        if not event_records:
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

        payload_tasks = [
            fetch_event(str(event["event_id"]), int(event["year"])) for event in event_records
        ]
        payloads = await asyncio.gather(*payload_tasks, return_exceptions=True)

        per_player: dict[str, dict[str, Any]] = {}
        successful_events = 0
        total_observations = 0
        latest_event_phase = 0.5

        for event_record, payload in zip(event_records, payloads):
            if isinstance(payload, Exception):
                continue
            event_phase = _to_float(event_record.get("phase")) or 0.5
            latest_event_phase = event_phase
            observations = _extract_player_metric_observations(payload)
            event_features = _extract_player_event_features(payload, event_phase=event_phase)
            if observations or event_features:
                successful_events += 1
            total_observations += len(observations)

            grouped_event_observations: dict[str, dict[str, Any]] = {}
            for canonical_key, player_id, player_name, metric in observations:
                bucket = per_player.setdefault(
                    canonical_key,
                    {
                        "player_id": player_id,
                        "player_name": player_name,
                        "metrics": [],
                        "event_metrics": [],
                        "event_phases": [],
                        "finish_scores": [],
                        "win_phases": [],
                        "starts": 0,
                        "wins": 0,
                        "top10": 0,
                    },
                )
                bucket["metrics"].append(metric)
                grouped = grouped_event_observations.setdefault(
                    canonical_key,
                    {
                        "player_id": player_id,
                        "player_name": player_name,
                        "metrics": [],
                    },
                )
                grouped["metrics"].append(metric)

            if event_features:
                for feature in event_features:
                    canonical_key = str(feature["canonical_key"])
                    player_id = str(feature.get("player_id") or "")
                    player_name = str(feature.get("player_name") or "")
                    metric = _to_float(feature.get("metric"))
                    if metric is None:
                        continue
                    bucket = per_player.setdefault(
                        canonical_key,
                        {
                            "player_id": player_id,
                            "player_name": player_name,
                            "metrics": [],
                            "event_metrics": [],
                            "event_phases": [],
                            "finish_scores": [],
                            "win_phases": [],
                            "starts": 0,
                            "wins": 0,
                            "top10": 0,
                        },
                    )
                    if bucket.get("player_id") in (None, "") and player_id:
                        bucket["player_id"] = player_id
                    if bucket.get("player_name") in (None, "") and player_name:
                        bucket["player_name"] = player_name

                    bucket["event_metrics"].append(float(metric))
                    bucket["event_phases"].append(float(event_phase))
                    bucket["starts"] += 1
                    finish_rank = _to_float(feature.get("finish_rank"))
                    finish_score = _finish_score_from_rank(finish_rank)
                    if finish_score is not None:
                        bucket["finish_scores"].append(float(finish_score))
                    won = bool(feature.get("won"))
                    top10 = bool(feature.get("top10"))
                    if won:
                        bucket["wins"] += 1
                        bucket["win_phases"].append(float(event_phase))
                    if top10:
                        bucket["top10"] += 1
            else:
                # Fallback for round-only payloads: synthesize one event-level datapoint per player.
                for canonical_key, grouped in grouped_event_observations.items():
                    metrics = grouped.get("metrics") or []
                    if not metrics:
                        continue
                    metric = float(np.mean(np.asarray(metrics, dtype=np.float64)))
                    bucket = per_player.setdefault(
                        canonical_key,
                        {
                            "player_id": grouped.get("player_id"),
                            "player_name": grouped.get("player_name"),
                            "metrics": [],
                            "event_metrics": [],
                            "event_phases": [],
                            "finish_scores": [],
                            "win_phases": [],
                            "starts": 0,
                            "wins": 0,
                            "top10": 0,
                        },
                    )
                    bucket["event_metrics"].append(metric)
                    bucket["event_phases"].append(float(event_phase))
                    bucket["starts"] += 1

        out: dict[str, dict[str, Any]] = {}
        for key, values in per_player.items():
            metrics = np.asarray(values["metrics"], dtype=np.float64)
            event_metrics = np.asarray(values.get("event_metrics") or [], dtype=np.float64)
            event_phases = np.asarray(values.get("event_phases") or [], dtype=np.float64)
            finish_scores = np.asarray(values.get("finish_scores") or [], dtype=np.float64)
            win_phases = np.asarray(values.get("win_phases") or [], dtype=np.float64)

            metric_source = event_metrics if event_metrics.size > 0 else metrics
            if metric_source.size == 0:
                continue
            starts = int(values.get("starts") or 0)
            if starts <= 0 and event_metrics.size > 0:
                starts = int(event_metrics.size)
            if starts <= 0:
                starts = 0

            recent_metric = _ewma(metric_source.tolist(), alpha=0.55)
            recent_finish_score = _ewma(finish_scores.tolist(), alpha=0.60)
            hot_streak_score = _hot_streak_score(win_phases.tolist())
            out[key] = {
                "player_id": values.get("player_id"),
                "player_name": values.get("player_name"),
                "metric": float(metric_source.mean()),
                "rounds": int(metrics.size) if metrics.size > 0 else int(metric_source.size),
                "starts": starts,
                "wins": int(values.get("wins") or 0),
                "top10": int(values.get("top10") or 0),
                "volatility": (
                    float(metric_source.std(ddof=1))
                    if metric_source.size > 1
                    else None
                ),
                "recent_metric": recent_metric,
                "recent_finish_score": recent_finish_score,
                "hot_streak_score": hot_streak_score,
                "event_metrics": [float(v) for v in event_metrics.tolist()],
                "event_phases": [float(v) for v in event_phases.tolist()],
                "win_phases": [float(v) for v in win_phases.tolist()],
            }

        if not out:
            raise DataGolfAPIError(
                f"No usable player round metrics found for tour={tour}, season={season}."
            )

        result = _SeasonLoadResult(
            metrics=out,
            event_count=len(event_records),
            successful_event_count=successful_events,
            player_metric_count=len(out),
            observation_count=total_observations,
            season_phase_hint=latest_event_phase,
            from_cache=False,
        )
        self._season_metrics_cache[cache_key] = result
        return result

    @staticmethod
    def _apply_seasonal_form_metrics(
        players: list[_PlayerRecord],
        baseline_metrics: dict[str, dict[str, Any]],
        current_metrics: dict[str, dict[str, Any]],
        season_phase: float | None = None,
    ) -> int:
        phase_value = 0.5 if season_phase is None else float(np.clip(season_phase, 0.0, 1.0))
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
                baseline_mean_metric = _to_float(base.get("metric"))
                baseline_phase_metric = _phase_weighted_metric_from_profile(
                    base,
                    target_phase=phase_value,
                    default_value=baseline_mean_metric,
                )
                player.baseline_season_metric = baseline_mean_metric
                player.baseline_phase_metric = baseline_phase_metric
                player.baseline_season_rounds = int(base.get("rounds") or 0)
                baseline_starts_value = _to_float(base.get("starts"))
                if baseline_starts_value is None:
                    baseline_starts_value = float(player.baseline_season_rounds)
                player.baseline_season_starts = int(max(0.0, baseline_starts_value))
                player.baseline_season_volatility = _to_float(base.get("volatility"))
                player.baseline_hot_streak_score = _to_float(base.get("hot_streak_score"))
            if curr:
                player.current_season_metric = _to_float(curr.get("metric"))
                player.current_recent_metric = _to_float(curr.get("recent_metric")) or player.current_season_metric
                player.current_recent_finish_score = _to_float(curr.get("recent_finish_score"))
                player.current_season_rounds = int(curr.get("rounds") or 0)
                current_starts_value = _to_float(curr.get("starts"))
                if current_starts_value is None:
                    current_starts_value = float(player.current_season_rounds)
                player.current_season_starts = int(max(0.0, current_starts_value))
                player.current_season_volatility = _to_float(curr.get("volatility"))

            anchor_metric = (
                player.baseline_phase_metric
                if player.baseline_phase_metric is not None
                else player.baseline_season_metric
            )
            current_metric = (
                player.current_recent_metric
                if player.current_recent_metric is not None
                else player.current_season_metric
            )

            if anchor_metric is not None and current_metric is not None:
                # Bayesian shrinkage against start count avoids penalizing players with zero starts.
                starts = float(max(0, player.current_season_starts))
                tau = 3.5
                reliability = starts / (starts + tau)
                posterior_current = (
                    (reliability * current_metric)
                    + ((1.0 - reliability) * anchor_metric)
                )
                player.form_delta_metric = posterior_current - anchor_metric
            elif (
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
        base_mean_reversion: float = 0.10,
        seasonal_form_weight: float = 0.35,
        current_season_weight: float = 0.60,
        form_delta_weight: float = 0.25,
        initial_totals: np.ndarray | None = None,
        round_fractions: np.ndarray | None = None,
        round_numbers: np.ndarray | None = None,
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
        baseline_phase_metric = np.array(
            [
                p.baseline_phase_metric
                if p.baseline_phase_metric is not None
                else p.baseline_season_metric
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
        current_recent_metric = np.array(
            [
                p.current_recent_metric
                if p.current_recent_metric is not None
                else p.current_season_metric
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
        baseline_starts = np.array(
            [max(0, p.baseline_season_starts) for p in players], dtype=np.float64
        )
        current_starts = np.array(
            [max(0, p.current_season_starts) for p in players], dtype=np.float64
        )
        recent_finish_score = np.array(
            [
                p.current_recent_finish_score
                if p.current_recent_finish_score is not None
                else np.nan
                for p in players
            ],
            dtype=np.float64,
        )
        baseline_hot_streak = np.array(
            [
                p.baseline_hot_streak_score
                if p.baseline_hot_streak_score is not None
                else 0.0
                for p in players
            ],
            dtype=np.float64,
        )

        baseline_z = _zscore_with_nan(baseline_metric)
        baseline_phase_z = _zscore_with_nan(baseline_phase_metric)
        current_z = _zscore_with_nan(current_metric)
        current_recent_z = _zscore_with_nan(current_recent_metric)
        recent_finish_z = _zscore_with_nan(recent_finish_score)

        has_baseline = ~np.isnan(baseline_z)
        has_current = ~np.isnan(current_z)
        has_current_recent = ~np.isnan(current_recent_z)
        has_season_signal = has_baseline | has_current

        baseline_signal = np.where(has_baseline, baseline_z, 0.0)
        baseline_phase_signal = np.where(
            ~np.isnan(baseline_phase_z),
            baseline_phase_z,
            baseline_signal,
        )
        current_signal = np.where(has_current, current_z, baseline_phase_signal)
        current_recent_signal = np.where(
            has_current_recent,
            current_recent_z,
            current_signal,
        )

        current_start_reliability = np.divide(
            current_starts,
            current_starts + 3.5,
            out=np.zeros_like(current_starts, dtype=np.float64),
            where=(current_starts + 3.5) > 1e-8,
        )

        # Phase-specific baseline anchor captures player seasonality from the baseline season.
        seasonal_anchor = baseline_phase_signal

        # Bayesian shrinkage prevents zero/low-start players from being unfairly penalized.
        current_shrunk_signal = (
            (current_start_reliability * current_recent_signal)
            + ((1.0 - current_start_reliability) * seasonal_anchor)
        )

        phase_strength = seasonal_anchor - baseline_signal
        finish_signal = np.where(~np.isnan(recent_finish_z), recent_finish_z, 0.0)
        hot_signal = np.where(np.isnan(baseline_hot_streak), 0.0, baseline_hot_streak)

        base_current_weight = float(np.clip(current_season_weight, 0.0, 1.0))
        base_current_weight = float(np.clip(base_current_weight, 0.02, 0.98))
        base_logit = np.log(base_current_weight / (1.0 - base_current_weight))

        dynamic_logit = (
            base_logit
            + (1.15 * finish_signal)
            + (0.85 * (current_start_reliability - 0.5))
            - (0.70 * phase_strength)
            + (0.45 * hot_signal)
        )
        dynamic_current_weight = 1.0 / (
            1.0 + np.exp(-np.clip(dynamic_logit, -8.0, 8.0))
        )
        dynamic_current_weight = np.clip(dynamic_current_weight, 0.05, 0.95)

        seasonal_delta = current_shrunk_signal - seasonal_anchor
        delta_gain = np.clip(form_delta_weight, 0.0, 1.0) * np.sqrt(current_start_reliability)
        seasonal_signal = (
            ((1.0 - dynamic_current_weight) * seasonal_anchor)
            + (dynamic_current_weight * current_shrunk_signal)
            + (delta_gain * seasonal_delta)
        )

        for idx, player in enumerate(players):
            player.dynamic_current_weight = float(dynamic_current_weight[idx])

        blended_skill = np.where(
            has_season_signal,
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

        volatility_signal = np.array(
            [
                (
                    p.current_season_volatility
                    if p.current_season_volatility is not None
                    else p.baseline_season_volatility
                    if p.baseline_season_volatility is not None
                    else np.nan
                )
                for p in players
            ],
            dtype=np.float64,
        )
        volatility_z = _zscore_with_nan(volatility_signal)
        observation_reliability = np.clip((baseline_starts + current_starts) / 20.0, 0.0, 1.0)
        reversion_adjustment = np.where(~np.isnan(volatility_z), -0.03 * volatility_z, 0.0)
        player_mean_reversion = (
            np.clip(base_mean_reversion, 0.0, 0.6)
            + (observation_reliability * reversion_adjustment)
        )
        player_mean_reversion = np.clip(player_mean_reversion, 0.02, 0.35)

        if initial_totals is not None:
            initial_totals = np.asarray(initial_totals, dtype=np.float64).reshape(-1)
            if initial_totals.shape[0] != len(players):
                initial_totals = None

        if round_fractions is not None:
            round_fractions = np.asarray(round_fractions, dtype=np.float64).reshape(-1)
            if round_fractions.size == 0:
                round_fractions = None

        if round_numbers is not None:
            round_numbers = np.asarray(round_numbers, dtype=np.int16).reshape(-1)
            if round_fractions is not None and round_numbers.size != round_fractions.size:
                round_numbers = None

        return SimulationInputs(
            player_ids=player_ids,
            player_names=player_names,
            mu_round=mu_round,
            sigma_round=sigma_round,
            mean_reversion=player_mean_reversion,
            initial_totals=initial_totals,
            round_fractions=round_fractions,
            round_numbers=round_numbers,
        )

    @staticmethod
    def _build_in_play_conditioning_context(
        players: list[_PlayerRecord],
        total_rounds: int = 4,
        enable: bool = True,
    ) -> _InPlayConditioningResult:
        n_players = len(players)
        zero_totals = np.zeros(n_players, dtype=np.float64)
        default_fractions = np.ones(total_rounds, dtype=np.float64)
        default_numbers = np.arange(1, total_rounds + 1, dtype=np.int16)

        if not enable or n_players == 0:
            return _InPlayConditioningResult(
                applied=False,
                note="In-play conditioning disabled.",
                initial_totals=zero_totals,
                round_fractions=default_fractions,
                round_numbers=default_numbers,
            )

        initial_totals = np.zeros(n_players, dtype=np.float64)
        have_live_scores = np.zeros(n_players, dtype=bool)
        nonzero_score = False
        has_round_data = False
        has_thru_data = False
        completed_round_estimates: list[int] = []
        thru_in_progress_holes: list[int] = []

        for idx, player in enumerate(players):
            score = player.current_score_to_par
            if score is not None and np.isfinite(score):
                initial_totals[idx] = float(score)
                have_live_scores[idx] = True
                if abs(float(score)) > 1e-8:
                    nonzero_score = True

            thru_holes = _thru_to_hole_count(player.current_thru)
            if thru_holes is not None:
                if thru_holes > 0:
                    has_thru_data = True
                if 1 <= thru_holes < 18:
                    thru_in_progress_holes.append(thru_holes)

            rounds_len = len(player.round_scores)
            if rounds_len > 0:
                has_round_data = True
                if thru_holes is not None and 1 <= thru_holes < 18:
                    completed_round_estimates.append(max(0, rounds_len - 1))
                elif thru_holes == 18:
                    completed_round_estimates.append(rounds_len)
                else:
                    completed_round_estimates.append(rounds_len)
            elif thru_holes == 18 and player.today_score_to_par is not None:
                # We have evidence the active round is complete even without explicit round list.
                completed_round_estimates.append(1)

        live_coverage = (
            float(have_live_scores.mean()) if have_live_scores.size > 0 else 0.0
        )
        started_signal = has_round_data or has_thru_data or nonzero_score
        if not started_signal:
            return _InPlayConditioningResult(
                applied=False,
                note="No reliable in-play signal; using full 4-round pre-event simulation.",
                initial_totals=zero_totals,
                round_fractions=default_fractions,
                round_numbers=default_numbers,
            )

        imputed_player_count = 0
        if have_live_scores.any():
            missing_scores = ~have_live_scores
            imputed_player_count = int(missing_scores.sum())
            if imputed_player_count > 0:
                observed_scores = initial_totals[have_live_scores]
                fill_value = float(np.median(observed_scores))
                initial_totals[missing_scores] = fill_value

        completed_rounds = (
            int(np.median(completed_round_estimates)) if completed_round_estimates else 0
        )
        completed_rounds = int(np.clip(completed_rounds, 0, total_rounds))

        round_fractions: list[float] = []
        round_numbers: list[int] = []
        in_progress = bool(thru_in_progress_holes) and completed_rounds < total_rounds

        if in_progress:
            current_round_number = min(total_rounds, completed_rounds + 1)
            median_thru = float(np.median(thru_in_progress_holes))
            fraction_remaining = np.clip((18.0 - median_thru) / 18.0, 0.0, 1.0)
            if fraction_remaining > 1e-3:
                round_fractions.append(float(fraction_remaining))
                round_numbers.append(current_round_number)
            next_round = current_round_number + 1
        else:
            next_round = completed_rounds + 1

        for round_number in range(next_round, total_rounds + 1):
            round_fractions.append(1.0)
            round_numbers.append(round_number)

        if completed_rounds >= total_rounds and not in_progress:
            note = (
                "Live event appears complete; probabilities are conditioned on current scores only."
            )
            return _InPlayConditioningResult(
                applied=True,
                note=note,
                initial_totals=initial_totals,
                round_fractions=np.zeros(0, dtype=np.float64),
                round_numbers=np.zeros(0, dtype=np.int16),
            )

        if not round_fractions:
            return _InPlayConditioningResult(
                applied=False,
                note="Unable to infer reliable rounds remaining from live feed.",
                initial_totals=zero_totals,
                round_fractions=default_fractions,
                round_numbers=default_numbers,
            )

        note = (
            "In-play conditioning applied from live leaderboard state: "
            f"coverage={live_coverage:.0%}, completed_rounds={completed_rounds}, "
            f"remaining_steps={len(round_fractions)}, imputed_players={imputed_player_count}."
        )
        return _InPlayConditioningResult(
            applied=True,
            note=note,
            initial_totals=initial_totals,
            round_fractions=np.asarray(round_fractions, dtype=np.float64),
            round_numbers=np.asarray(round_numbers, dtype=np.int16),
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


def _extract_live_score_snapshot(row: dict[str, Any]) -> _LiveScoreSnapshot:
    current_position = _normalize_position_value(_value_from_payload(row, _LIVE_POSITION_KEYS))
    current_score_to_par = _score_to_par_from_live_row(row)
    current_thru = _normalize_thru_value(_value_from_payload(row, _LIVE_THRU_KEYS))
    today_score_to_par = _score_to_par_from_payload(row, _LIVE_TODAY_KEYS)
    round_scores = _round_scores_from_live_row(row)
    hole_scores = _hole_scores_from_live_row(row)

    derived_total = _derive_total_to_par_from_round_data(
        round_scores=round_scores,
        today_score_to_par=today_score_to_par,
        current_thru=current_thru,
    )
    rounds_are_to_par = _round_scores_look_like_to_par(round_scores)
    if derived_total is not None and (
        current_score_to_par is None
        or (
            rounds_are_to_par
            and abs(float(current_score_to_par) - float(derived_total)) >= 0.75
        )
        or (
            abs(float(current_score_to_par)) <= 0.1
            and abs(float(derived_total)) >= 0.5
        )
    ):
        current_score_to_par = derived_total

    return _LiveScoreSnapshot(
        current_position=current_position,
        current_score_to_par=current_score_to_par,
        current_thru=current_thru,
        today_score_to_par=today_score_to_par,
        round_scores=round_scores,
        hole_scores=hole_scores,
    )


def _reconcile_merged_live_scores(records: list[_PlayerRecord]) -> None:
    inferred_round_par = _infer_round_par_from_records(records)
    for record in records:
        derived_total = _derive_total_to_par_from_round_data(
            round_scores=record.round_scores,
            today_score_to_par=record.today_score_to_par,
            current_thru=record.current_thru,
            inferred_round_par=inferred_round_par,
        )
        if derived_total is None:
            continue

        current_score = record.current_score_to_par
        if current_score is None:
            record.current_score_to_par = derived_total
            continue

        # Live feeds sometimes return stale "E" while the round is in progress.
        if abs(float(current_score)) <= 0.1 and abs(float(derived_total)) >= 0.5:
            record.current_score_to_par = derived_total
            continue

        thru_holes = _thru_to_hole_count(record.current_thru)
        if (
            thru_holes is not None
            and 1 <= thru_holes < 18
            and record.today_score_to_par is not None
            and abs(float(record.today_score_to_par)) >= 0.5
            and abs(float(current_score) - float(derived_total)) >= 0.75
        ):
            record.current_score_to_par = derived_total


def _infer_round_par_from_records(records: list[_PlayerRecord]) -> float | None:
    candidates: list[float] = []
    for record in records:
        current_score = record.current_score_to_par
        if current_score is None or not record.round_scores:
            continue
        if _round_scores_look_like_to_par(record.round_scores):
            continue

        round_values = np.asarray(record.round_scores, dtype=np.float64)
        if round_values.size == 0 or not np.isfinite(round_values).all():
            continue

        thru_holes = _thru_to_hole_count(record.current_thru)
        adjusted_total = float(current_score)
        if thru_holes is not None and 1 <= thru_holes < 18:
            today = record.today_score_to_par
            if today is None:
                continue
            # Ignore likely placeholder totals ("E") while in-progress.
            if abs(float(current_score)) <= 0.1 and abs(float(today)) >= 0.5:
                continue
            adjusted_total -= float(today)

        par_candidate = (float(np.sum(round_values)) - adjusted_total) / float(round_values.size)
        if 66.0 <= par_candidate <= 75.0:
            candidates.append(float(par_candidate))

    if not candidates:
        return None
    return float(np.median(np.asarray(candidates, dtype=np.float64)))


def _round_scores_from_live_row(row: dict[str, Any]) -> list[float]:
    raw_scores = _value_from_payload(row, _LIVE_ROUND_SCORE_KEYS)
    if raw_scores is not None:
        extracted = _sanitize_round_scores(_numeric_sequence_from_value(raw_scores))
        if 1 <= len(extracted) <= 6:
            return extracted

    flattened: list[float] = []
    for key in _ROUND_SCORE_FLAT_KEYS:
        value = _numeric_from_payload(row, (key,))
        if value is None:
            continue
        flattened.append(value)
    return _sanitize_round_scores(flattened[:6])


def _hole_scores_from_live_row(row: dict[str, Any]) -> list[int]:
    raw_scores = _value_from_payload(row, _LIVE_HOLE_SCORE_KEYS)
    if raw_scores is not None:
        extracted = _sanitize_hole_scores(_numeric_sequence_from_value(raw_scores))
        if extracted:
            return extracted

    flattened: list[float] = []
    for key in _HOLE_SCORE_FLAT_KEYS:
        value = _numeric_from_payload(row, (key,))
        if value is None:
            continue
        flattened.append(value)
    return _sanitize_hole_scores(flattened)


def _sanitize_round_scores(values: list[float]) -> list[float]:
    out: list[float] = []
    for value in values:
        if not np.isfinite(value):
            continue
        if abs(value) > 250:
            continue
        out.append(float(value))
    return out


def _sanitize_hole_scores(values: list[float]) -> list[int]:
    out: list[int] = []
    for value in values:
        if not np.isfinite(value):
            continue
        rounded = int(round(value))
        if abs(value - rounded) > 1e-6:
            continue
        if 1 <= rounded <= 15:
            out.append(rounded)
    if len(out) > 18:
        return out[:18]
    return out


def _score_to_par_from_live_row(row: dict[str, Any]) -> float | None:
    # Prioritize player-level total score fields; avoid recursive grabs of hole-level "to_par"/"tot".
    direct_value = _top_level_value_from_keys(row, _LIVE_SCORE_TO_PAR_KEYS)
    direct = _score_to_par_from_value(direct_value)
    if direct is not None:
        return direct

    fallback_value = _top_level_value_from_keys(row, _LIVE_SCORE_TO_PAR_FALLBACK_KEYS)
    fallback = _score_to_par_from_value(fallback_value)
    if fallback is not None:
        return fallback

    lowered = {str(k).lower(): v for k, v in row.items()}
    for container in _LIVE_SCORE_NESTED_CONTAINERS:
        nested = lowered.get(container)
        if nested is None:
            continue
        nested_value = _value_from_payload(nested, _LIVE_SCORE_TO_PAR_KEYS)
        nested_score = _score_to_par_from_value(nested_value)
        if nested_score is not None:
            return nested_score

    return None


def _top_level_value_from_keys(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    lowered = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        if key not in lowered:
            continue
        value = lowered[key]
        if _is_empty_payload_value(value):
            continue
        return value
    return None


def _round_scores_look_like_to_par(round_scores: list[float]) -> bool:
    if not round_scores:
        return False
    numeric = np.asarray(round_scores, dtype=np.float64)
    if not np.isfinite(numeric).all():
        return False
    # To-par round values are usually small magnitudes; gross scores are ~65-80.
    return bool(np.max(np.abs(numeric)) <= 20.0)


def _derive_total_to_par_from_round_data(
    round_scores: list[float],
    today_score_to_par: float | None,
    current_thru: str | None,
    inferred_round_par: float | None = None,
) -> float | None:
    is_to_par_rounds = _round_scores_look_like_to_par(round_scores)

    round_values = np.asarray(round_scores, dtype=np.float64)
    round_total = float(np.sum(round_values))
    thru_holes = _thru_to_hole_count(current_thru)

    if not is_to_par_rounds:
        # Some DataGolf payloads expose gross round scores (e.g. 74/68/66/65) plus today's to-par.
        # Infer round par either from cohort context or from a completed round.
        round_par = inferred_round_par
        if round_par is None and (
            today_score_to_par is not None
            and round_values.size > 0
            and thru_holes is not None
            and thru_holes >= 18
        ):
            round_par = float(round_values[-1] - float(today_score_to_par))

        if round_par is None or not (66.0 <= float(round_par) <= 75.0):
            return None

        completed_round_total = round_total - (float(round_par) * float(round_values.size))
        if thru_holes is None or thru_holes <= 0 or thru_holes >= 18:
            if abs(completed_round_total) <= 80.0:
                return float(completed_round_total)
            return None

        if today_score_to_par is None:
            return None
        in_progress_total = completed_round_total + float(today_score_to_par)
        if abs(in_progress_total) <= 80.0:
            return float(in_progress_total)
        return None

    if thru_holes is None:
        return round_total
    if thru_holes >= 18:
        return round_total
    if thru_holes <= 0:
        return round_total

    if today_score_to_par is None:
        return round_total

    # Many feeds report completed rounds separately; add current in-progress today score.
    # If the latest round score already equals today's value, avoid double counting.
    if round_scores and abs(float(round_scores[-1]) - float(today_score_to_par)) <= 0.25:
        return round_total
    return round_total + float(today_score_to_par)


def _score_to_par_from_payload(payload: Any, keys: tuple[str, ...]) -> float | None:
    value = _value_from_payload(payload, keys)
    return _score_to_par_from_value(value)


def _score_to_par_from_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return _validated_to_par(float(value))
    text = str(value).strip()
    if not text:
        return None

    normalized = text.upper().replace("−", "-")
    if normalized in {"E", "EVEN", "PAR"}:
        return 0.0

    direct = re.fullmatch(r"[+-]?\d+(?:\.\d+)?", normalized)
    if direct:
        return _validated_to_par(float(normalized))

    wrapped = re.fullmatch(r"\(([+-]?\d+(?:\.\d+)?)\)", normalized)
    if wrapped:
        return _validated_to_par(float(wrapped.group(1)))

    words = re.fullmatch(r"(\d+(?:\.\d+)?)\s+(UNDER|OVER)", normalized)
    if words:
        magnitude = float(words.group(1))
        signed = -magnitude if words.group(2) == "UNDER" else magnitude
        return _validated_to_par(signed)
    return None


def _validated_to_par(value: float) -> float | None:
    if abs(value) > 80:
        return None
    return value


def _normalize_position_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if float(value).is_integer():
            return str(int(value))
        return str(value)

    text = str(value).strip()
    if not text:
        return None

    normalized = text.upper().replace("−", "-")
    if normalized in {"WD", "MC", "DQ", "DNS", "DNF"}:
        return normalized

    tie_match = re.fullmatch(r"T\s*([+-]?\d+(?:\.\d+)?)", normalized)
    if tie_match:
        number = float(tie_match.group(1))
        number_text = str(int(number)) if number.is_integer() else str(number)
        return f"T{number_text}"

    straight_match = re.fullmatch(r"[+-]?\d+(?:\.\d+)?", normalized)
    if straight_match:
        number = float(normalized)
        if number.is_integer():
            return str(int(number))
        return str(number)
    return text


def _normalize_thru_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if float(value).is_integer():
            return str(int(value))
        return str(value)

    text = str(value).strip()
    if not text:
        return None

    normalized = text.upper()
    if normalized in {"F", "FIN", "FINAL", "COMPLETE", "COMPLETED"}:
        return "F"
    if normalized in {"WD", "MC", "DQ"}:
        return normalized

    prefix_match = re.match(r"THRU\s+(\d{1,2})$", normalized)
    if prefix_match:
        return prefix_match.group(1)

    digit_match = re.fullmatch(r"\d{1,2}", normalized)
    if digit_match:
        return digit_match.group(0)
    return text


def _thru_to_hole_count(value: Any) -> int | None:
    normalized = _normalize_thru_value(value)
    if normalized is None:
        return None
    text = normalized.strip().upper()
    if text in {"F", "FINAL"}:
        return 18
    if text.isdigit():
        holes = int(text)
        if 0 <= holes <= 18:
            return holes
    return None


def _numeric_from_payload(payload: Any, keys: tuple[str, ...]) -> float | None:
    value = _value_from_payload(payload, keys)
    return _to_float(value)


def _value_from_payload(payload: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(payload, dict):
        lowered = {str(k).lower(): v for k, v in payload.items()}
        for key in keys:
            if key not in lowered:
                continue
            value = lowered[key]
            if _is_empty_payload_value(value):
                continue
            return value
        for nested in lowered.values():
            nested_value = _value_from_payload(nested, keys)
            if nested_value is not None:
                return nested_value
    elif isinstance(payload, list):
        for item in payload:
            nested_value = _value_from_payload(item, keys)
            if nested_value is not None:
                return nested_value
    return None


def _is_empty_payload_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def _historical_event_descriptors(
    event_list_payload: Any,
    season: int,
    limit: int = 72,
) -> list[tuple[str, int]]:
    records = _historical_event_records(
        event_list_payload=event_list_payload,
        season=season,
        limit=limit,
    )
    return [(str(record["event_id"]), int(record["year"])) for record in records]


def _historical_event_records(
    event_list_payload: Any,
    season: int,
    limit: int = 72,
) -> list[dict[str, Any]]:
    rows = _extract_rows(event_list_payload, ("event", "list", "historical", "schedule"))
    if not rows:
        rows = _collect_dict_nodes(event_list_payload)

    records: list[dict[str, Any]] = []
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
        records.append(
            {
                "event_id": event_id,
                "year": int(year),
                "date": _string_from_keys(row, _DATE_KEYS),
            }
        )
        if len(records) >= limit:
            break

    records.sort(key=lambda record: _safe_date_sort_key(_to_str(record.get("date"))))
    total = len(records)
    for idx, record in enumerate(records):
        if total <= 1:
            phase = _season_phase_from_date(_to_str(record.get("date")))
            record["phase"] = 0.5 if phase is None else phase
        else:
            record["phase"] = float(idx / (total - 1))
    return records


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
                if number is None:
                    number = _score_to_par_from_value(item)
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
                    continue
                score_to_par = _score_to_par_from_value(
                    _string_from_keys(
                        item,
                        (
                            "to_par",
                            "score_to_par",
                            "round_to_par",
                            "today",
                        ),
                    )
                )
                if score_to_par is not None:
                    out.append(score_to_par)
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


def _to_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_player_event_features(
    payload: Any,
    event_phase: float,
) -> list[dict[str, Any]]:
    rows = _extract_rows(payload, ("event_stats", "event", "player"))
    if not rows:
        rows = _collect_dict_nodes(payload)

    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        canonical_key, player_id, player_name = _historical_player_identity(row)
        if not canonical_key:
            continue
        metric = _event_level_metric_from_row(row)
        if metric is None:
            metric = _round_metric_from_row(row)
        if metric is None:
            round_scores = _round_scores_from_row(row)
            if round_scores:
                metric = float(-np.mean(np.asarray(round_scores, dtype=np.float64)))
        if metric is None:
            continue

        finish_rank = _finish_rank_from_history_row(row)
        existing = deduped.get(canonical_key)
        if existing is None:
            deduped[canonical_key] = {
                "canonical_key": canonical_key,
                "player_id": player_id,
                "player_name": player_name,
                "metric": float(metric),
                "finish_rank": finish_rank,
                "won": bool(finish_rank == 1 if finish_rank is not None else False),
                "top10": bool(finish_rank is not None and finish_rank <= 10),
                "event_phase": float(event_phase),
                "n": 1,
            }
            continue

        # Some payload variants duplicate rows; collapse to one event-level datapoint.
        existing["metric"] = (
            (float(existing["metric"]) * int(existing["n"])) + float(metric)
        ) / float(int(existing["n"]) + 1)
        existing["n"] = int(existing["n"]) + 1
        if finish_rank is not None:
            prior_rank = _to_float(existing.get("finish_rank"))
            if prior_rank is None or finish_rank < prior_rank:
                existing["finish_rank"] = int(finish_rank)
            existing["won"] = bool(existing.get("won")) or finish_rank == 1
            existing["top10"] = bool(existing.get("top10")) or finish_rank <= 10

    out: list[dict[str, Any]] = []
    for feature in deduped.values():
        feature.pop("n", None)
        out.append(feature)
    return out


def _finish_rank_from_history_row(row: dict[str, Any]) -> int | None:
    fin_text = _string_from_keys(row, ("fin_text", "finish", "position", "pos", "rank"))
    if fin_text:
        parsed = _finish_text_to_rank(fin_text)
        if parsed is not None:
            return parsed
    rank = _numeric_from_keys(row, ("finish", "position", "pos", "rank"))
    if rank is None:
        return None
    rank_int = int(round(rank))
    if rank_int <= 0:
        return None
    return rank_int


def _finish_score_from_rank(rank_value: float | None) -> float | None:
    if rank_value is None:
        return None
    rank = int(round(rank_value))
    if rank <= 0:
        return None
    # Positive score for strong finishes, decays smoothly by rank.
    return float(1.0 / np.sqrt(rank))


def _ewma(values: list[float], alpha: float = 0.55) -> float | None:
    if not values:
        return None
    clipped_alpha = float(np.clip(alpha, 0.05, 0.95))
    estimate = float(values[0])
    for value in values[1:]:
        estimate = (clipped_alpha * float(value)) + ((1.0 - clipped_alpha) * estimate)
    return float(estimate)


def _phase_weighted_metric_from_profile(
    profile: dict[str, Any],
    target_phase: float,
    default_value: float | None = None,
) -> float | None:
    metrics_raw = profile.get("event_metrics")
    phases_raw = profile.get("event_phases")
    if not isinstance(metrics_raw, list) or not isinstance(phases_raw, list):
        return default_value
    if not metrics_raw or not phases_raw:
        return default_value

    metrics = np.asarray(metrics_raw, dtype=np.float64)
    phases = np.asarray(phases_raw, dtype=np.float64)
    if metrics.size == 0 or phases.size == 0 or metrics.size != phases.size:
        return default_value

    target = float(np.clip(target_phase, 0.0, 1.0))
    distance = np.abs(phases - target)
    # Blend local seasonality with a mild global floor so sparse histories stay stable.
    weights = np.exp(-0.5 * ((distance / 0.18) ** 2)) + 0.10
    weight_total = float(np.sum(weights))
    if weight_total <= 1e-8:
        return default_value
    return float(np.sum(weights * metrics) / weight_total)


def _hot_streak_score(win_phases: list[float]) -> float:
    if not win_phases:
        return 0.0
    phases = np.sort(np.asarray(win_phases, dtype=np.float64))
    if phases.size == 1:
        return 0.25
    phase_gaps = np.diff(phases)
    mean_gap = float(np.mean(phase_gaps)) if phase_gaps.size > 0 else 1.0
    cluster_score = 1.0 - float(np.clip(mean_gap / 0.35, 0.0, 1.0))
    win_count_boost = float(np.clip((phases.size - 1) / 4.0, 0.0, 1.0))
    return float((0.65 * cluster_score) + (0.35 * win_count_boost))


def _season_phase_from_date(date_value: str | None) -> float | None:
    if not date_value:
        return None
    text = str(date_value).strip()
    if not text:
        return None

    parsed: datetime | None = None
    parse_candidates = [text]
    if "T" in text:
        parse_candidates.append(text.split("T")[0])
    if " " in text:
        parse_candidates.append(text.split(" ")[0])
    parse_candidates.append(text[:10])

    formats = ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S")
    for candidate in parse_candidates:
        if not candidate:
            continue
        for fmt in formats:
            try:
                parsed = datetime.strptime(candidate, fmt)
                break
            except ValueError:
                continue
        if parsed is not None:
            break

    if parsed is None:
        safe = _safe_date_sort_key(text)
        if safe.year >= 9999:
            return None
        parsed = safe

    year = int(parsed.year)
    is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    days_in_year = 366 if is_leap else 365
    day_of_year = int(parsed.timetuple().tm_yday)
    if days_in_year <= 1:
        return 0.5
    return float(np.clip((day_of_year - 1) / (days_in_year - 1), 0.0, 1.0))
