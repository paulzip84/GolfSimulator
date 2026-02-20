import numpy as np

from pga_sim.simulation import HybridMarkovSimulator, MarkovSimulationConfig, SimulationInputs


def test_stronger_player_has_higher_win_probability() -> None:
    inputs = SimulationInputs(
        player_ids=["a", "b", "c"],
        player_names=["Player A", "Player B", "Player C"],
        mu_round=np.array([-1.2, -0.2, 0.9], dtype=np.float64),
        sigma_round=np.array([2.3, 2.5, 2.8], dtype=np.float64),
    )
    sim = HybridMarkovSimulator(MarkovSimulationConfig(mean_reversion=0.12))
    out = sim.simulate(inputs=inputs, n_simulations=12_000, seed=7, cut_size=3)

    assert out.win_probability[0] > out.win_probability[1] > out.win_probability[2]
    assert abs(float(out.win_probability.sum()) - 1.0) < 1e-9


def test_probability_monotonicity_and_mean_finish() -> None:
    inputs = SimulationInputs(
        player_ids=["x1", "x2", "x3", "x4"],
        player_names=["X1", "X2", "X3", "X4"],
        mu_round=np.array([-1.0, -0.5, 0.2, 0.9], dtype=np.float64),
        sigma_round=np.array([2.2, 2.4, 2.6, 2.9], dtype=np.float64),
    )
    sim = HybridMarkovSimulator()
    out = sim.simulate(inputs=inputs, n_simulations=10_000, seed=101, cut_size=4)

    assert np.all(out.top_10_probability >= out.top_5_probability)
    assert np.all(out.top_5_probability >= out.top_3_probability)
    assert np.all(out.top_3_probability >= out.win_probability)

    # Better player (lower mean round delta) should finish better on average.
    assert out.mean_finish[0] < out.mean_finish[1] < out.mean_finish[2] < out.mean_finish[3]


def test_adaptive_stopping_can_end_early_for_clear_favorites() -> None:
    inputs = SimulationInputs(
        player_ids=["fav", "longshot"],
        player_names=["Favorite", "Longshot"],
        mu_round=np.array([-4.0, 4.0], dtype=np.float64),
        sigma_round=np.array([1.8, 2.8], dtype=np.float64),
    )
    sim = HybridMarkovSimulator(MarkovSimulationConfig(mean_reversion=0.10, round_shock_sigma=0.25))
    out = sim.simulate(
        inputs=inputs,
        n_simulations=30_000,
        seed=123,
        cut_size=2,
        adaptive=True,
        min_simulations=2_000,
        batch_size=1_000,
        ci_half_width_target=0.005,
        ci_top_n=1,
    )

    assert out.simulations_run < 30_000
    assert out.adaptive_stopped_early is True
    assert out.win_ci_half_width_top_n is not None
    assert out.win_ci_half_width_top_n <= 0.005


def test_in_play_conditioning_with_no_rounds_left_is_deterministic() -> None:
    inputs = SimulationInputs(
        player_ids=["p1", "p2", "p3"],
        player_names=["P1", "P2", "P3"],
        mu_round=np.array([-0.2, -0.2, -0.2], dtype=np.float64),
        sigma_round=np.array([2.4, 2.4, 2.4], dtype=np.float64),
        initial_totals=np.array([-5.0, 0.0, 2.0], dtype=np.float64),
        round_fractions=np.array([], dtype=np.float64),
        round_numbers=np.array([], dtype=np.int16),
    )
    sim = HybridMarkovSimulator()
    out = sim.simulate(inputs=inputs, n_simulations=4_000, seed=9, cut_size=3)

    assert out.win_probability[0] == 1.0
    assert out.win_probability[1] == 0.0
    assert out.win_probability[2] == 0.0
    assert np.allclose(out.mean_total_relative_to_field, np.array([-5.0, 0.0, 2.0]))


def test_player_specific_mean_reversion_changes_expected_totals() -> None:
    inputs = SimulationInputs(
        player_ids=["high_mr", "low_mr"],
        player_names=["HighMR", "LowMR"],
        mu_round=np.array([0.0, 0.0], dtype=np.float64),
        sigma_round=np.array([2.4, 2.4], dtype=np.float64),
        mean_reversion=np.array([0.24, 0.02], dtype=np.float64),
        initial_totals=np.array([8.0, 8.0], dtype=np.float64),
    )
    sim = HybridMarkovSimulator(MarkovSimulationConfig(mean_reversion=0.10, round_shock_sigma=0.0))
    out = sim.simulate(inputs=inputs, n_simulations=12_000, seed=77, cut_size=2)

    # Higher reversion should pull positive states back toward field faster.
    assert out.mean_total_relative_to_field[0] < out.mean_total_relative_to_field[1]


def test_adaptive_recommends_higher_cap_when_target_not_met() -> None:
    inputs = SimulationInputs(
        player_ids=["a", "b", "c", "d"],
        player_names=["A", "B", "C", "D"],
        mu_round=np.array([-0.2, -0.1, 0.1, 0.2], dtype=np.float64),
        sigma_round=np.array([2.5, 2.5, 2.5, 2.5], dtype=np.float64),
    )
    sim = HybridMarkovSimulator(MarkovSimulationConfig(mean_reversion=0.10, round_shock_sigma=0.3))
    out = sim.simulate(
        inputs=inputs,
        n_simulations=4_000,
        seed=22,
        cut_size=4,
        adaptive=True,
        min_simulations=2_000,
        batch_size=1_000,
        ci_confidence=0.99,
        ci_half_width_target=0.001,
        ci_top_n=4,
    )

    assert out.ci_target_met is False
    assert out.stop_reason == "max_simulations_reached"
    assert out.recommended_simulations is not None
    assert out.recommended_simulations > 4_000
