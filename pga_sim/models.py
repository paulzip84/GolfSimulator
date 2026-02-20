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
    resolution_mode: str = Field(default="auto_target")
    simulations: int = Field(default=10_000, ge=500, le=250_000)
    min_simulations: int = Field(default=5_000, ge=500, le=250_000)
    simulation_batch_size: int = Field(default=5_000, ge=500, le=50_000)
    seed: Optional[int] = Field(default=None)
    cut_size: int = Field(default=70, ge=20, le=156)
    mean_reversion: float = Field(default=0.10, ge=0.0, le=0.4)
    shared_round_shock_sigma: float = Field(default=0.35, ge=0.0, le=2.0)
    enable_adaptive_simulation: bool = Field(default=True)
    ci_confidence: float = Field(default=0.95, ge=0.5, le=0.999)
    ci_half_width_target: float = Field(default=0.0025, ge=0.0001, le=0.05)
    ci_top_n: int = Field(default=10, ge=1, le=50)
    enable_in_play_conditioning: bool = Field(default=True)
    enable_seasonal_form: bool = Field(default=True)
    baseline_season: Optional[int] = Field(default=None, ge=1990, le=2100)
    current_season: Optional[int] = Field(default=None, ge=1990, le=2100)
    seasonal_form_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    current_season_weight: float = Field(default=0.60, ge=0.0, le=1.0)
    form_delta_weight: float = Field(default=0.25, ge=0.0, le=1.0)


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
    round_scores: list[float] = Field(default_factory=list)
    hole_scores: list[int] = Field(default_factory=list)


class SimulationResponse(BaseModel):
    generated_at: datetime
    tour: str
    event_id: Optional[str]
    event_name: Optional[str]
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
    players: list[PlayerSimulationOutput]
