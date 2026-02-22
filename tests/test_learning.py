from __future__ import annotations

import numpy as np

from pga_sim.learning import LearningStore


def _event_payload(event_name: str, winners: list[tuple[int, str, int]]) -> dict:
    return {
        "event_name": event_name,
        "event_completed": "2025-04-14",
        "event_stats": [
            {
                "dg_id": player_id,
                "player_name": player_name,
                "fin_text": str(rank),
                "dg_points": float(max(1, 35 - rank)),
            }
            for player_id, player_name, rank in winners
        ],
    }


def test_learning_store_records_predictions_and_outcomes(tmp_path) -> None:
    db_path = tmp_path / "learning.sqlite3"
    store = LearningStore(str(db_path))

    players = [
        {
            "player_id": str(1000 + idx),
            "player_name": f"Player {idx + 1}",
            "win_probability": 0.04 * (idx + 1),
            "top_3_probability": min(0.95, 0.08 * (idx + 1)),
            "top_5_probability": min(0.98, 0.11 * (idx + 1)),
            "top_10_probability": min(0.99, 0.14 * (idx + 1)),
        }
        for idx in range(8)
    ]

    store.record_prediction(
        tour="pga",
        event_id="14",
        event_name="Masters Tournament",
        event_date="2025-04-10",
        requested_simulations=10000,
        simulations=10000,
        enable_in_play=True,
        in_play_applied=False,
        players=players,
    )

    pending = store.list_pending_events(tour="pga", max_events=10)
    assert len(pending) == 1
    assert pending[0].event_id == "14"
    assert pending[0].event_year == 2025

    outcome_rows = [(1000 + idx, f"Player {idx + 1}", idx + 1) for idx in range(8)]
    inserted = store.record_outcome_payload(
        tour="pga",
        event_id="14",
        event_year=2025,
        payload=_event_payload("Masters Tournament", outcome_rows),
    )
    assert inserted == 8

    pending_after = store.list_pending_events(tour="pga", max_events=10)
    assert pending_after == []

    snapshot = store.retrain(tour="pga")
    assert snapshot.version == 1
    assert "win" in snapshot.markets

    status = store.status(tour="pga")
    assert status["predictions_logged"] == 1
    assert status["resolved_events"] == 1
    assert status["resolved_predictions"] == 8


def test_learning_retraining_updates_calibration_metrics(tmp_path) -> None:
    db_path = tmp_path / "learning_calibration.sqlite3"
    store = LearningStore(str(db_path))

    rng = np.random.default_rng(42)
    base_win = np.array([0.30, 0.20, 0.14, 0.11, 0.08, 0.06, 0.05, 0.03], dtype=np.float64)
    base_win = base_win / base_win.sum()
    # Sharper true distribution than model probabilities.
    true_weights = base_win ** 1.6
    true_win = true_weights / true_weights.sum()

    for event_index in range(30):
        event_id = f"E{event_index}"
        event_date = f"2025-03-{(event_index % 27) + 1:02d}"

        players = []
        for idx, win_prob in enumerate(base_win):
            players.append(
                {
                    "player_id": str(2000 + idx),
                    "player_name": f"Golfer {idx + 1}",
                    "win_probability": float(win_prob),
                    "top_3_probability": float(min(0.97, win_prob * 3.0)),
                    "top_5_probability": float(min(0.985, win_prob * 4.0)),
                    "top_10_probability": float(min(0.995, win_prob * 6.0)),
                }
            )

        winner_idx = int(rng.choice(len(base_win), p=true_win))
        remaining = [idx for idx in range(len(base_win)) if idx != winner_idx]
        rng.shuffle(remaining)
        rank_order = [winner_idx] + remaining

        event_stats: list[tuple[int, str, int]] = []
        for rank, idx in enumerate(rank_order, start=1):
            event_stats.append((2000 + idx, f"Golfer {idx + 1}", rank))

        store.record_prediction(
            tour="pga",
            event_id=event_id,
            event_name=f"Event {event_index}",
            event_date=event_date,
            requested_simulations=20000,
            simulations=20000,
            enable_in_play=True,
            in_play_applied=False,
            players=players,
        )
        store.record_outcome_payload(
            tour="pga",
            event_id=event_id,
            event_year=2025,
            payload=_event_payload(f"Event {event_index}", event_stats),
        )

    snapshot = store.retrain(tour="pga")
    win_market = snapshot.markets["win"]
    assert win_market.samples >= 240
    assert win_market.brier_before is not None
    assert win_market.brier_after is not None
    assert win_market.brier_after <= (win_market.brier_before + 0.01)
    assert win_market.beta > 0.0


def test_event_trends_returns_snapshot_deltas(tmp_path) -> None:
    db_path = tmp_path / "learning_trends.sqlite3"
    store = LearningStore(str(db_path))

    base_players = [
        {
            "player_id": "5001",
            "player_name": "Alpha",
            "win_probability": 0.20,
            "top_3_probability": 0.45,
            "top_5_probability": 0.62,
            "top_10_probability": 0.83,
        },
        {
            "player_id": "5002",
            "player_name": "Beta",
            "win_probability": 0.14,
            "top_3_probability": 0.36,
            "top_5_probability": 0.51,
            "top_10_probability": 0.74,
        },
    ]
    updated_players = [
        {
            "player_id": "5001",
            "player_name": "Alpha",
            "win_probability": 0.31,
            "top_3_probability": 0.56,
            "top_5_probability": 0.70,
            "top_10_probability": 0.88,
        },
        {
            "player_id": "5002",
            "player_name": "Beta",
            "win_probability": 0.09,
            "top_3_probability": 0.29,
            "top_5_probability": 0.45,
            "top_10_probability": 0.69,
        },
    ]

    store.record_prediction(
        tour="pga",
        event_id="14",
        event_name="Masters Tournament",
        event_date="2025-04-10",
        requested_simulations=12000,
        simulations=12000,
        enable_in_play=True,
        in_play_applied=False,
        players=base_players,
    )
    store.record_prediction(
        tour="pga",
        event_id="14",
        event_name="Masters Tournament",
        event_date="2025-04-10",
        requested_simulations=12000,
        simulations=12000,
        enable_in_play=True,
        in_play_applied=True,
        players=updated_players,
    )

    trends = store.event_trends(
        tour="pga",
        event_id="14",
        event_year=2025,
        max_snapshots=10,
        max_players=5,
    )
    assert trends["snapshot_count"] == 2
    assert trends["latest_run_id"] is not None
    assert len(trends["players"]) == 2

    alpha = next(player for player in trends["players"] if player["player_id"] == "5001")
    assert len(alpha["points"]) == 2
    assert alpha["delta_win_since_first"] > 0
    assert alpha["delta_win_since_previous"] > 0


def test_retrain_without_observations_keeps_version_stable(tmp_path) -> None:
    db_path = tmp_path / "learning_empty.sqlite3"
    store = LearningStore(str(db_path))

    snapshot_before = store.get_snapshot(tour="pga")
    snapshot_after = store.retrain(tour="pga")

    assert snapshot_before.version == 0
    assert snapshot_after.version == 0
