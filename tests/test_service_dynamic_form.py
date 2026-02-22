from pga_sim.service import (
    SimulationService,
    _PlayerRecord,
    _phase_weighted_metric_from_profile,
)


def test_phase_weighted_metric_prefers_requested_season_window() -> None:
    profile = {
        "event_metrics": [2.4, -1.2, 2.0],
        "event_phases": [0.1, 0.5, 0.9],
    }
    early = _phase_weighted_metric_from_profile(profile, target_phase=0.1, default_value=0.0)
    middle = _phase_weighted_metric_from_profile(profile, target_phase=0.5, default_value=0.0)
    late = _phase_weighted_metric_from_profile(profile, target_phase=0.9, default_value=0.0)

    assert early is not None and middle is not None and late is not None
    assert early > middle
    assert late > middle


def test_form_delta_uses_start_count_shrinkage() -> None:
    players = [_PlayerRecord(player_id="1001", player_name="Player 1")]
    baseline_metrics = {
        "1001": {
            "player_id": "1001",
            "player_name": "Player 1",
            "metric": 1.8,
            "rounds": 12,
            "starts": 12,
            "event_metrics": [1.8, 1.7, 1.9],
            "event_phases": [0.1, 0.5, 0.9],
        }
    }
    # Current-season metric exists but starts=0 (player has not actually teed it up yet).
    current_metrics = {
        "1001": {
            "player_id": "1001",
            "player_name": "Player 1",
            "metric": -2.2,
            "recent_metric": -2.2,
            "recent_finish_score": 0.2,
            "rounds": 1,
            "starts": 0,
        }
    }

    applied = SimulationService._apply_seasonal_form_metrics(
        players=players,
        baseline_metrics=baseline_metrics,
        current_metrics=current_metrics,
        season_phase=0.2,
    )

    assert applied == 1
    assert players[0].form_delta_metric is not None
    # With zero starts, posterior current form should shrink completely to baseline anchor.
    assert abs(players[0].form_delta_metric) < 1e-9


def test_dynamic_player_weight_uses_recent_finishes_and_starts() -> None:
    seasonality_player = _PlayerRecord(
        player_id="a",
        player_name="Seasonality Heavy",
        baseline_season_metric=0.4,
        baseline_phase_metric=0.3,
        current_season_metric=-0.8,
        current_recent_metric=-0.9,
        current_recent_finish_score=0.1,
        baseline_hot_streak_score=0.9,
        baseline_season_starts=18,
        current_season_starts=1,
    )
    recency_player = _PlayerRecord(
        player_id="b",
        player_name="Recency Heavy",
        baseline_season_metric=0.3,
        baseline_phase_metric=0.2,
        current_season_metric=1.2,
        current_recent_metric=1.5,
        current_recent_finish_score=0.9,
        baseline_hot_streak_score=0.1,
        baseline_season_starts=18,
        current_season_starts=8,
    )

    inputs = SimulationService._build_simulation_inputs(
        players=[seasonality_player, recency_player],
        seasonal_form_weight=1.0,
        current_season_weight=0.6,
        form_delta_weight=0.25,
    )

    assert seasonality_player.dynamic_current_weight is not None
    assert recency_player.dynamic_current_weight is not None
    assert recency_player.dynamic_current_weight > seasonality_player.dynamic_current_weight
    assert seasonality_player.dynamic_current_weight < 0.5
    assert recency_player.dynamic_current_weight > 0.5
    assert inputs.mu_round.shape[0] == 2
