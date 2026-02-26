from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class EventSummary(BaseModel):
    event_id: str
    event_name: str
    start_date: Optional[str] = None
    course: Optional[str] = None
    simulatable: bool = True
    unavailable_reason: Optional[str] = None


class SimulationRequest(BaseModel):
    tour: str = Field(default="pga")
    event_id: Optional[str] = Field(default=None)
    resolution_mode: str = Field(default="fixed_cap")
    simulations: int = Field(default=1_000_000, ge=500, le=2_000_000)
    min_simulations: int = Field(default=250_000, ge=500, le=250_000)
    simulation_batch_size: int = Field(default=1_000, ge=500, le=50_000)
    seed: Optional[int] = Field(default=None)
    cut_size: int = Field(default=70, ge=20, le=156)
    mean_reversion: float = Field(default=0.10, ge=0.0, le=0.4)
    shared_round_shock_sigma: float = Field(default=0.35, ge=0.0, le=2.0)
    enable_adaptive_simulation: bool = Field(default=True)
    ci_confidence: float = Field(default=0.975, ge=0.5, le=0.999)
    ci_half_width_target: float = Field(default=0.0015, ge=0.0001, le=0.05)
    ci_top_n: int = Field(default=15, ge=1, le=50)
    enable_in_play_conditioning: bool = Field(default=True)
    enable_seasonal_form: bool = Field(default=True)
    baseline_season: Optional[int] = Field(default=None, ge=1990, le=2100)
    current_season: Optional[int] = Field(default=None, ge=1990, le=2100)
    seasonal_form_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    current_season_weight: float = Field(default=0.85, ge=0.0, le=1.0)
    form_delta_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    snapshot_type: str = Field(default="manual")


class PlayerSimulationOutput(BaseModel):
    player_id: str
    player_name: str
    win_probability: float
    top_3_probability: float
    top_5_probability: float
    top_10_probability: float
    mean_finish: float
    mean_total_relative_to_field: float
    baseline_win_probability: Optional[float] = None
    baseline_top_3_probability: Optional[float] = None
    baseline_top_5_probability: Optional[float] = None
    baseline_top_10_probability: Optional[float] = None
    baseline_season_metric: Optional[float] = None
    current_season_metric: Optional[float] = None
    form_delta_metric: Optional[float] = None
    baseline_season_rounds: int = 0
    current_season_rounds: int = 0
    current_position: Optional[str] = None
    current_score_to_par: Optional[float] = None
    current_thru: Optional[str] = None
    today_score_to_par: Optional[float] = None
    win_delta_prev: Optional[float] = None
    win_delta_start: Optional[float] = None
    round_scores: list[float] = Field(default_factory=list)
    hole_scores: list[int] = Field(default_factory=list)


class SimulationResponse(BaseModel):
    generated_at: datetime
    tour: str
    event_id: Optional[str]
    event_name: Optional[str]
    simulation_version: int = 1
    simulations: int
    requested_simulations: Optional[int] = None
    adaptive_stopped_early: bool = False
    win_ci_half_width_top_n: Optional[float] = None
    ci_target_met: bool = False
    stop_reason: Optional[str] = None
    recommended_simulations: Optional[int] = None
    in_play_conditioning_applied: bool = False
    in_play_conditioning_note: Optional[str] = None
    baseline_season: Optional[int] = None
    current_season: Optional[int] = None
    form_adjustment_applied: bool = False
    form_adjustment_note: Optional[str] = None
    calibration_applied: bool = False
    calibration_version: int = 0
    calibration_note: Optional[str] = None
    players: list[PlayerSimulationOutput]


class LiveScoreRow(BaseModel):
    player_id: Optional[str] = None
    player_name: str
    current_position: Optional[str] = None
    current_score_to_par: Optional[float] = None
    current_thru: Optional[str] = None
    today_score_to_par: Optional[float] = None
    round_scores: list[float] = Field(default_factory=list)
    hole_scores: list[int] = Field(default_factory=list)


class LiveScoresResponse(BaseModel):
    generated_at: datetime
    tour: str
    event_id: Optional[str] = None
    event_name: Optional[str] = None
    event_state: str = "scheduled"
    source_note: Optional[str] = None
    players: list[LiveScoreRow] = Field(default_factory=list)


class CalibrationMarketStatus(BaseModel):
    market: str
    alpha: float = 0.0
    beta: float = 1.0
    samples: int = 0
    positives: int = 0
    brier_before: Optional[float] = None
    brier_after: Optional[float] = None
    logloss_before: Optional[float] = None
    logloss_after: Optional[float] = None


class LearningStatusResponse(BaseModel):
    tour: str
    predictions_logged: int = 0
    resolved_predictions: int = 0
    resolved_events: int = 0
    pending_events: int = 0
    calibration_version: int = 0
    calibration_updated_at: Optional[datetime] = None
    markets: list[CalibrationMarketStatus] = Field(default_factory=list)


class LearningSyncRequest(BaseModel):
    tour: str = Field(default="pga")
    max_events: int = Field(default=40, ge=1, le=200)


class LearningSyncResponse(LearningStatusResponse):
    outcomes_fetched: int = 0
    provisional_outcomes_fetched: int = 0
    events_processed: int = 0
    event_ids_processed: list[str] = Field(default_factory=list)
    provisional_event_ids: list[str] = Field(default_factory=list)
    awaiting_outcomes_count: int = 0
    awaiting_outcomes_event_ids: list[str] = Field(default_factory=list)
    retrain_executed: bool = False
    sync_note: Optional[str] = None


class LearningEventSnapshot(BaseModel):
    run_id: str
    created_at: datetime
    simulation_version: int = 1
    snapshot_type: str = "manual"
    simulations: Optional[int] = None
    in_play_applied: bool = False


class LearningEventTrendPoint(BaseModel):
    run_id: str
    created_at: datetime
    simulation_version: int = 1
    snapshot_type: str = "manual"
    win_probability: float
    top_3_probability: float
    top_5_probability: float
    top_10_probability: float


class LearningPlayerEventTrend(BaseModel):
    player_id: Optional[str] = None
    player_name: str
    latest_win_probability: float
    delta_win_since_first: Optional[float] = None
    delta_win_since_previous: Optional[float] = None
    latest_top_3_probability: float
    latest_top_5_probability: float
    latest_top_10_probability: float
    points: list[LearningEventTrendPoint] = Field(default_factory=list)


class LearningEventTrendsResponse(BaseModel):
    tour: str
    event_id: str
    event_year: int
    event_name: Optional[str] = None
    snapshot_count: int = 0
    latest_run_id: Optional[str] = None
    latest_simulation_version: Optional[int] = None
    snapshots: list[LearningEventSnapshot] = Field(default_factory=list)
    players: list[LearningPlayerEventTrend] = Field(default_factory=list)


class LifecycleEventStatus(BaseModel):
    tour: str
    event_id: str
    event_year: int
    event_name: Optional[str] = None
    event_date: Optional[str] = None
    state: str = "scheduled"
    pre_event_snapshot_version: Optional[int] = None
    outcomes_source: Optional[str] = None
    retrain_version: Optional[int] = None
    updated_at: Optional[datetime] = None
    last_note: Optional[str] = None


class LifecycleStatusResponse(BaseModel):
    generated_at: datetime
    tour: str
    automation_enabled: bool = True
    active_event_id: Optional[str] = None
    active_event_name: Optional[str] = None
    active_event_year: Optional[int] = None
    active_event_state: Optional[str] = None
    pre_event_snapshot_ready: bool = False
    pre_event_snapshot_version: Optional[int] = None
    pending_events: int = 0
    last_run_at: Optional[datetime] = None
    last_run_note: Optional[str] = None
    recent_events: list[LifecycleEventStatus] = Field(default_factory=list)


class PowerRankingEvent(BaseModel):
    event_id: str
    event_name: Optional[str] = None
    event_year: int
    event_date: Optional[str] = None
    source_snapshot_type: Optional[str] = None
    source_simulation_version: Optional[int] = None
    outcomes_available: bool = False


class PowerRankingPoint(BaseModel):
    event_id: str
    event_name: Optional[str] = None
    event_year: int
    event_date: Optional[str] = None
    rank: Optional[int] = None
    score: Optional[float] = None
    event_score: Optional[float] = None


class PowerRankingPlayerSeries(BaseModel):
    player_id: Optional[str] = None
    player_name: str
    latest_rank: Optional[int] = None
    latest_score: Optional[float] = None
    points: list[PowerRankingPoint] = Field(default_factory=list)


class PowerRankingsResponse(BaseModel):
    generated_at: datetime
    tour: str
    event_year: int
    lookback_events: int
    top_n: int
    events: list[PowerRankingEvent] = Field(default_factory=list)
    players: list[PowerRankingPlayerSeries] = Field(default_factory=list)
    note: Optional[str] = None


class PowerRankingWarmStartRequest(BaseModel):
    tours: list[str] = Field(default_factory=lambda: ["pga", "liv", "euro", "kft"])
    event_year: Optional[int] = Field(default=None, ge=1990, le=2100)
    simulations: int = Field(default=100_000, ge=500, le=2_000_000)
    force: bool = Field(default=False)


class PowerRankingWarmStartTourResult(BaseModel):
    tour: str
    status: str
    event_id: Optional[str] = None
    event_name: Optional[str] = None
    event_year: Optional[int] = None
    simulation_version: Optional[int] = None
    simulations: Optional[int] = None
    note: Optional[str] = None


class PowerRankingWarmStartResponse(BaseModel):
    generated_at: datetime
    event_year: int
    simulations: int
    force: bool = False
    results: list[PowerRankingWarmStartTourResult] = Field(default_factory=list)
    note: Optional[str] = None
