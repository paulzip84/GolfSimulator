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
    simulations: int = Field(default=10_000, ge=500, le=250_000)
    seed: Optional[int] = Field(default=None)
    cut_size: int = Field(default=70, ge=20, le=156)
    mean_reversion: float = Field(default=0.10, ge=0.0, le=0.4)
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


class SimulationResponse(BaseModel):
    generated_at: datetime
    tour: str
    event_id: Optional[str]
    event_name: Optional[str]
    simulations: int
    baseline_season: Optional[int] = None
    current_season: Optional[int] = None
    form_adjustment_applied: bool = False
    form_adjustment_note: Optional[str] = None
    players: list[PlayerSimulationOutput]
