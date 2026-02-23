from __future__ import annotations

import asyncio
from contextlib import suppress
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from .auth import AuthConfigurationError, RequestAuthenticator
from .config import get_settings
from .datagolf_client import DataGolfAPIError, DataGolfClient
from .learning import LearningStore
from .models import (
    EventSummary,
    LifecycleStatusResponse,
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
_service = SimulationService(
    _client,
    learning_store=_learning_store,
    simulation_max_sync_simulations=_settings.simulation_max_sync_simulations,
    simulation_max_batch_size=_settings.simulation_max_batch_size,
    lifecycle_automation_enabled=_settings.lifecycle_automation_enabled,
    lifecycle_tour=_settings.lifecycle_tour,
    lifecycle_pre_event_simulations=_settings.lifecycle_pre_event_simulations,
    lifecycle_pre_event_seed=_settings.lifecycle_pre_event_seed,
    lifecycle_sync_max_events=_settings.lifecycle_sync_max_events,
    lifecycle_backfill_enabled=_settings.lifecycle_backfill_enabled,
    lifecycle_backfill_batch_size=_settings.lifecycle_backfill_batch_size,
    lifecycle_target_year=_settings.lifecycle_target_year,
)
_authenticator = RequestAuthenticator(_settings)
_web_dir = Path(__file__).resolve().parent / "web"
_assets_dir = _web_dir / "assets"
_lifecycle_stop_event: asyncio.Event | None = None
_lifecycle_task: asyncio.Task[None] | None = None


async def _lifecycle_worker(stop_event: asyncio.Event) -> None:
    interval_seconds = max(30, int(_settings.lifecycle_automation_interval_seconds))
    lifecycle_tour = (_settings.lifecycle_tour or "pga").strip().lower()
    while not stop_event.is_set():
        try:
            await _service.run_lifecycle_cycle(tour=lifecycle_tour)
        except Exception:
            # Keep automation resilient; status endpoint still reflects last run note.
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            continue


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _lifecycle_stop_event
    global _lifecycle_task
    _authenticator.validate_configuration()
    if _settings.lifecycle_automation_enabled:
        _lifecycle_stop_event = asyncio.Event()
        _lifecycle_task = asyncio.create_task(_lifecycle_worker(_lifecycle_stop_event))
    try:
        yield
    finally:
        if _lifecycle_stop_event is not None:
            _lifecycle_stop_event.set()
        if _lifecycle_task is not None:
            with suppress(Exception):
                await _lifecycle_task
        _lifecycle_stop_event = None
        _lifecycle_task = None
        await _client.aclose()


app = FastAPI(
    title="PGA Markov Simulator",
    version="0.1.0",
    description="Local PGA tournament simulator with DataGolf and Markov + stochastic modeling.",
    lifespan=lifespan,
)

if _assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    try:
        _authenticator.authenticate_request(request)
    except AuthConfigurationError as exc:
        return JSONResponse(status_code=500, content={"detail": str(exc)})
    except HTTPException as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=exc.headers or None,
        )
    return await call_next(request)


def require_learning_admin(request: Request) -> None:
    _authenticator.require_role(request, "admin")


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


@app.get("/auth/me")
async def auth_me(request: Request) -> dict[str, object]:
    user = _authenticator.current_user(request)
    return {
        "auth_mode": _authenticator.mode,
        "authenticated": user is not None,
        "subject": user.subject if user is not None else None,
        "email": user.email if user is not None else None,
        "roles": sorted(user.roles) if user is not None else [],
    }


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
async def learning_sync_train(
    request: LearningSyncRequest,
    _: None = Depends(require_learning_admin),
) -> LearningSyncResponse:
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


@app.get("/lifecycle/status", response_model=LifecycleStatusResponse)
async def lifecycle_status(
    tour: str = Query(default="pga"),
) -> LifecycleStatusResponse:
    try:
        return await _service.get_lifecycle_status(tour=tour)
    except DataGolfAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/lifecycle/run", response_model=LifecycleStatusResponse)
async def lifecycle_run(
    tour: Optional[str] = Query(default=None),
    _: None = Depends(require_learning_admin),
) -> LifecycleStatusResponse:
    try:
        return await _service.run_lifecycle_cycle(tour=tour)
    except DataGolfAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def run() -> None:
    uvicorn.run("pga_sim.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    run()
