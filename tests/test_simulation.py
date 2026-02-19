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

