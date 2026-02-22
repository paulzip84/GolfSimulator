from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from .config import get_settings
from .datagolf_client import DataGolfAPIError, DataGolfClient
from .learning import LearningStore
from .models import (
    EventSummary,
    LearningEventTrendsResponse,
    LearningStatusResponse,
    LearningSyncRequest,
    LearningSyncResponse,
    SimulationRequest,
    SimulationResponse,
)
from .service import SimulationService

_settings = get_settings()
_client = DataGolfClient(_settings)
_learning_store = LearningStore(_settings.learning_database_path)
_service = SimulationService(_client, learning_store=_learning_store)
_web_dir = Path(__file__).resolve().parent / "web"
_assets_dir = _web_dir / "assets"


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        yield
    finally:
        await _client.aclose()


app = FastAPI(
    title="PGA Markov Simulator",
    version="0.1.0",
    description="Local PGA tournament simulator with DataGolf and Markov + stochastic modeling.",
    lifespan=lifespan,
)

if _assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
@app.get("/ui", include_in_schema=False)
async def web_app() -> FileResponse:
    index_html = _web_dir / "index.html"
    if not index_html.exists():
        raise HTTPException(status_code=503, detail="Web UI assets not found.")
    return FileResponse(index_html)


@app.get("/events/upcoming", response_model=list[EventSummary])
async def events_upcoming(
    tour: str = Query(default="pga"),
    limit: int = Query(default=12, ge=1, le=40),
) -> list[EventSummary]:
    try:
        return await _service.list_upcoming_events(tour=tour, limit=limit)
    except DataGolfAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/simulate", response_model=SimulationResponse)
async def simulate_tournament(request: SimulationRequest) -> SimulationResponse:
    try:
        return await _service.simulate(request)
    except DataGolfAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/learning/status", response_model=LearningStatusResponse)
async def learning_status(tour: str = Query(default="pga")) -> LearningStatusResponse:
    try:
        return await _service.get_learning_status(tour=tour)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/learning/sync-train", response_model=LearningSyncResponse)
async def learning_sync_train(request: LearningSyncRequest) -> LearningSyncResponse:
    try:
        return await _service.sync_learning_and_retrain(
            tour=request.tour,
            max_events=request.max_events,
        )
    except DataGolfAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/learning/event-trends", response_model=LearningEventTrendsResponse)
async def learning_event_trends(
    tour: str = Query(default="pga"),
    event_id: str = Query(...),
    event_year: Optional[int] = Query(default=None, ge=1990, le=2100),
    max_snapshots: int = Query(default=80, ge=2, le=300),
    max_players: int = Query(default=40, ge=1, le=200),
) -> LearningEventTrendsResponse:
    try:
        return await _service.get_learning_event_trends(
            tour=tour,
            event_id=event_id,
            event_year=event_year,
            max_snapshots=max_snapshots,
            max_players=max_players,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def run() -> None:
    uvicorn.run("pga_sim.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    run()
