from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from .config import get_settings
from .datagolf_client import DataGolfAPIError, DataGolfClient
from .models import EventSummary, SimulationRequest, SimulationResponse
from .service import SimulationService

app = FastAPI(
    title="PGA Markov Simulator",
    version="0.1.0",
    description="Local PGA tournament simulator with DataGolf and Markov + stochastic modeling.",
)

_settings = get_settings()
_client = DataGolfClient(_settings)
_service = SimulationService(_client)
_web_dir = Path(__file__).resolve().parent / "web"
_assets_dir = _web_dir / "assets"

if _assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await _client.aclose()


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


def run() -> None:
    uvicorn.run("pga_sim.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    run()
