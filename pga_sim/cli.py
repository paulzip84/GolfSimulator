from __future__ import annotations

import argparse
import asyncio

from .config import get_settings
from .datagolf_client import DataGolfAPIError, DataGolfClient
from .models import SimulationRequest
from .service import SimulationService


async def _run_events_command(tour: str, limit: int) -> None:
    settings = get_settings()
    async with DataGolfClient(settings) as client:
        service = SimulationService(client)
        events = await service.list_upcoming_events(tour=tour, limit=limit)

    if not events:
        print("No events returned.")
        return

    print(f"{'Event ID':<22} {'Start Date':<12} Event")
    print("-" * 70)
    for event in events:
        suffix = "" if event.simulatable else " [unavailable now]"
        print(
            f"{event.event_id:<22} "
            f"{(event.start_date or '-'): <12} "
            f"{event.event_name}{suffix}"
        )


async def _run_simulate_command(args: argparse.Namespace) -> None:
    settings = get_settings()
    request = SimulationRequest(
        tour=args.tour,
        event_id=args.event_id,
        simulations=args.simulations or settings.default_simulations,
        seed=args.seed,
        cut_size=args.cut_size,
        mean_reversion=args.mean_reversion,
        enable_seasonal_form=not args.disable_seasonal_form,
        baseline_season=args.baseline_season,
        current_season=args.current_season,
        seasonal_form_weight=args.seasonal_form_weight,
        current_season_weight=args.current_season_weight,
        form_delta_weight=args.form_delta_weight,
    )

    async with DataGolfClient(settings) as client:
        service = SimulationService(client)
        result = await service.simulate(request)

    top_n = min(args.top, len(result.players))
    print(
        f"\nEvent: {result.event_name or 'Unknown'} "
        f"(id={result.event_id or 'unknown'}) | sims={result.simulations}"
    )
    if result.baseline_season and result.current_season:
        print(
            f"Seasonal form blend: baseline={result.baseline_season}, "
            f"current={result.current_season}, applied={result.form_adjustment_applied}"
        )
    if result.form_adjustment_note:
        print(f"Note: {result.form_adjustment_note}")
    print(
        f"{'Rank':<4} {'Player':<26} {'Win%':>8} {'Top3%':>8} "
        f"{'Top5%':>8} {'Top10%':>8} {'AvgFin':>8}"
    )
    print("-" * 84)
    for rank, player in enumerate(result.players[:top_n], start=1):
        print(
            f"{rank:<4} "
            f"{player.player_name[:26]:<26} "
            f"{100.0 * player.win_probability:>7.2f}% "
            f"{100.0 * player.top_3_probability:>7.2f}% "
            f"{100.0 * player.top_5_probability:>7.2f}% "
            f"{100.0 * player.top_10_probability:>7.2f}% "
            f"{player.mean_finish:>8.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PGA Tour Markov simulation CLI backed by DataGolf."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    events_parser = sub.add_parser("events", help="List upcoming events")
    events_parser.add_argument("--tour", default="pga")
    events_parser.add_argument("--limit", type=int, default=12)

    sim_parser = sub.add_parser("simulate", help="Run tournament simulation")
    sim_parser.add_argument("--tour", default="pga")
    sim_parser.add_argument("--event-id", default=None)
    sim_parser.add_argument("--simulations", type=int, default=None)
    sim_parser.add_argument("--seed", type=int, default=None)
    sim_parser.add_argument("--cut-size", type=int, default=70)
    sim_parser.add_argument("--mean-reversion", type=float, default=0.10)
    sim_parser.add_argument("--disable-seasonal-form", action="store_true")
    sim_parser.add_argument("--baseline-season", type=int, default=None)
    sim_parser.add_argument("--current-season", type=int, default=None)
    sim_parser.add_argument("--seasonal-form-weight", type=float, default=0.35)
    sim_parser.add_argument("--current-season-weight", type=float, default=0.60)
    sim_parser.add_argument("--form-delta-weight", type=float, default=0.25)
    sim_parser.add_argument("--top", type=int, default=20)

    args = parser.parse_args()
    try:
        if args.command == "events":
            asyncio.run(_run_events_command(tour=args.tour, limit=args.limit))
            return
        if args.command == "simulate":
            asyncio.run(_run_simulate_command(args))
            return
        parser.error(f"Unsupported command: {args.command}")
    except DataGolfAPIError as exc:
        print(f"DataGolf API error: {exc}")


if __name__ == "__main__":
    main()
