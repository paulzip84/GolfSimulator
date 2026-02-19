from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MarkovSimulationConfig:
    rounds: int = 4
    cut_after_round: int = 2
    min_state: int = -40
    max_state: int = 40
    max_delta_per_round: int = 10
    mean_reversion: float = 0.10


@dataclass
class SimulationInputs:
    player_ids: list[str]
    player_names: list[str]
    mu_round: np.ndarray
    sigma_round: np.ndarray


@dataclass
class SimulationOutputs:
    win_probability: np.ndarray
    top_3_probability: np.ndarray
    top_5_probability: np.ndarray
    top_10_probability: np.ndarray
    mean_finish: np.ndarray
    mean_total_relative_to_field: np.ndarray
    made_cut_probability: np.ndarray


class HybridMarkovSimulator:
    def __init__(self, config: MarkovSimulationConfig | None = None):
        self._config = config or MarkovSimulationConfig()

    def simulate(
        self,
        inputs: SimulationInputs,
        n_simulations: int,
        seed: int | None = None,
        cut_size: int = 70,
    ) -> SimulationOutputs:
        if len(inputs.player_ids) == 0:
            raise ValueError("No players available for simulation.")
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive.")

        rng = np.random.default_rng(seed)
        n_players = len(inputs.player_ids)
        config = self._config

        state_values = np.arange(config.min_state, config.max_state + 1, dtype=np.float64)
        n_states = state_values.size
        delta_values = np.arange(
            -config.max_delta_per_round, config.max_delta_per_round + 1, dtype=np.int16
        )
        transition_cdfs = self._build_transition_cdfs(
            mu_round=inputs.mu_round.astype(np.float64),
            sigma_round=inputs.sigma_round.astype(np.float64),
            state_values=state_values,
            delta_values=delta_values,
            mean_reversion=config.mean_reversion,
        )

        totals = np.zeros((n_simulations, n_players), dtype=np.int16)
        states = np.zeros_like(totals)
        alive = np.ones((n_simulations, n_players), dtype=bool)

        player_idx = np.arange(n_players, dtype=np.int32)[None, :]
        for round_idx in range(config.rounds):
            state_idx = np.clip(states - config.min_state, 0, n_states - 1)
            cdf_rows = transition_cdfs[player_idx, state_idx]
            draws = rng.random((n_simulations, n_players))
            delta_idx = (draws[..., None] > cdf_rows).sum(axis=2)
            deltas = delta_values[delta_idx]
            deltas = np.where(alive, deltas, 0).astype(np.int16)

            totals += deltas
            states = np.clip(states + deltas, config.min_state, config.max_state)

            if (round_idx + 1) == config.cut_after_round and n_players > cut_size:
                sorted_totals = np.sort(totals, axis=1)
                cut_line = sorted_totals[:, cut_size - 1]
                alive = totals <= cut_line[:, None]

        win_probability = self._winner_probability(totals)
        top_3_probability = self._top_k_probability(totals, 3, rng)
        top_5_probability = self._top_k_probability(totals, 5, rng)
        top_10_probability = self._top_k_probability(totals, 10, rng)
        mean_finish = self._mean_finish(totals, rng)
        mean_total = totals.mean(axis=0)
        made_cut_probability = alive.mean(axis=0)

        return SimulationOutputs(
            win_probability=win_probability,
            top_3_probability=top_3_probability,
            top_5_probability=top_5_probability,
            top_10_probability=top_10_probability,
            mean_finish=mean_finish,
            mean_total_relative_to_field=mean_total,
            made_cut_probability=made_cut_probability,
        )

    @staticmethod
    def _build_transition_cdfs(
        mu_round: np.ndarray,
        sigma_round: np.ndarray,
        state_values: np.ndarray,
        delta_values: np.ndarray,
        mean_reversion: float,
    ) -> np.ndarray:
        player_count = mu_round.shape[0]
        sigma = np.clip(sigma_round.reshape(player_count, 1, 1), 0.8, 8.0)
        player_mu = mu_round.reshape(player_count, 1, 1)
        state_grid = state_values.reshape(1, state_values.size, 1)
        delta_grid = delta_values.reshape(1, 1, delta_values.size)

        shifted_mean = player_mu - (mean_reversion * state_grid)
        z = (delta_grid - shifted_mean) / sigma
        weights = np.exp(-0.5 * np.square(z))
        weights = np.clip(weights, 1e-12, None)
        pmf = weights / weights.sum(axis=2, keepdims=True)
        return np.cumsum(pmf, axis=2)

    @staticmethod
    def _winner_probability(totals: np.ndarray) -> np.ndarray:
        winner_scores = totals.min(axis=1, keepdims=True)
        winners = totals == winner_scores
        winner_count = winners.sum(axis=1, keepdims=True)
        return (winners / winner_count).mean(axis=0)

    @staticmethod
    def _top_k_probability(totals: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        n_simulations, n_players = totals.shape
        k = min(k, n_players)
        jitter = rng.uniform(0.0, 1e-6, size=totals.shape)
        order = np.argsort(totals + jitter, axis=1)
        rows = np.arange(n_simulations)[:, None]
        top_idx = order[:, :k]
        hits = np.zeros_like(totals, dtype=np.uint8)
        hits[rows, top_idx] = 1
        return hits.mean(axis=0)

    @staticmethod
    def _mean_finish(totals: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n_simulations, n_players = totals.shape
        jitter = rng.uniform(0.0, 1e-6, size=totals.shape)
        order = np.argsort(totals + jitter, axis=1)
        finish = np.empty_like(order, dtype=np.float64)
        rows = np.arange(n_simulations)[:, None]
        finish[rows, order] = np.arange(1, n_players + 1, dtype=np.float64)[None, :]
        return finish.mean(axis=0)
