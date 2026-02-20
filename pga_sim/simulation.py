from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


@dataclass
class MarkovSimulationConfig:
    rounds: int = 4
    cut_after_round: int = 2
    min_state: int = -40
    max_state: int = 40
    max_delta_per_round: int = 10
    mean_reversion: float = 0.10
    round_shock_sigma: float = 0.35


@dataclass
class SimulationInputs:
    player_ids: list[str]
    player_names: list[str]
    mu_round: np.ndarray
    sigma_round: np.ndarray
    mean_reversion: np.ndarray | None = None
    initial_totals: np.ndarray | None = None
    round_fractions: np.ndarray | None = None
    round_numbers: np.ndarray | None = None


@dataclass
class SimulationOutputs:
    win_probability: np.ndarray
    top_3_probability: np.ndarray
    top_5_probability: np.ndarray
    top_10_probability: np.ndarray
    mean_finish: np.ndarray
    mean_total_relative_to_field: np.ndarray
    made_cut_probability: np.ndarray
    simulations_run: int
    adaptive_stopped_early: bool = False
    win_ci_half_width_top_n: float | None = None
    ci_target_met: bool = False
    recommended_simulations: int | None = None
    stop_reason: str | None = None


@dataclass
class _ChunkMetrics:
    simulations: int
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
        adaptive: bool = False,
        min_simulations: int | None = None,
        batch_size: int = 5_000,
        ci_confidence: float = 0.95,
        ci_half_width_target: float = 0.0025,
        ci_top_n: int = 10,
    ) -> SimulationOutputs:
        if len(inputs.player_ids) == 0:
            raise ValueError("No players available for simulation.")
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive.")

        rng = np.random.default_rng(seed)
        n_players = len(inputs.player_ids)
        config = self._config

        max_simulations = int(n_simulations)
        min_sims = max(1, int(min_simulations if min_simulations is not None else max_simulations))
        min_sims = min(min_sims, max_simulations)
        batch = max(250, min(int(batch_size), max_simulations))
        adaptive_enabled = bool(adaptive and max_simulations > min_sims)
        ci_top_n = max(1, min(int(ci_top_n), n_players))
        ci_confidence = float(np.clip(ci_confidence, 0.5, 0.999))
        ci_half_width_target = float(np.clip(ci_half_width_target, 1e-6, 0.25))
        z_score = NormalDist().inv_cdf(0.5 + (0.5 * ci_confidence))

        mu_round = np.asarray(inputs.mu_round, dtype=np.float64)
        sigma_round = np.asarray(inputs.sigma_round, dtype=np.float64)
        if mu_round.shape[0] != n_players or sigma_round.shape[0] != n_players:
            raise ValueError("Simulation input arrays do not match number of players.")

        round_fractions, round_numbers = self._resolve_round_plan(
            inputs=inputs,
            default_rounds=config.rounds,
        )
        initial_totals = self._resolve_initial_totals(inputs=inputs, n_players=n_players)
        mean_reversion = self._resolve_mean_reversion(
            inputs=inputs,
            n_players=n_players,
            default_value=config.mean_reversion,
        )

        state_values = np.arange(config.min_state, config.max_state + 1, dtype=np.float64)
        n_states = state_values.size
        delta_values = np.arange(
            -config.max_delta_per_round, config.max_delta_per_round + 1, dtype=np.int16
        )

        transition_cdfs: dict[float, np.ndarray] = {}
        for fraction in np.unique(round_fractions):
            if fraction <= 1e-8:
                continue
            key = self._fraction_key(float(fraction))
            transition_cdfs[key] = self._build_transition_cdfs(
                mu_round=mu_round * fraction,
                sigma_round=sigma_round * np.sqrt(fraction),
                state_values=state_values,
                delta_values=delta_values,
                mean_reversion=mean_reversion * fraction,
            )

        sum_win = np.zeros(n_players, dtype=np.float64)
        sum_top_3 = np.zeros(n_players, dtype=np.float64)
        sum_top_5 = np.zeros(n_players, dtype=np.float64)
        sum_top_10 = np.zeros(n_players, dtype=np.float64)
        sum_finish = np.zeros(n_players, dtype=np.float64)
        sum_total = np.zeros(n_players, dtype=np.float64)
        sum_made_cut = np.zeros(n_players, dtype=np.float64)

        simulations_run = 0
        adaptive_stopped_early = False
        win_ci_half_width_top_n: float | None = None
        ci_target_met = False

        while simulations_run < max_simulations:
            chunk_size = min(batch, max_simulations - simulations_run)
            chunk = self._simulate_chunk(
                rng=rng,
                chunk_size=chunk_size,
                cut_size=cut_size,
                n_players=n_players,
                n_states=n_states,
                config=config,
                initial_totals=initial_totals,
                round_fractions=round_fractions,
                round_numbers=round_numbers,
                transition_cdfs=transition_cdfs,
                delta_values=delta_values,
            )

            sum_win += chunk.win_probability * chunk.simulations
            sum_top_3 += chunk.top_3_probability * chunk.simulations
            sum_top_5 += chunk.top_5_probability * chunk.simulations
            sum_top_10 += chunk.top_10_probability * chunk.simulations
            sum_finish += chunk.mean_finish * chunk.simulations
            sum_total += chunk.mean_total_relative_to_field * chunk.simulations
            sum_made_cut += chunk.made_cut_probability * chunk.simulations
            simulations_run += chunk.simulations

            if adaptive_enabled and simulations_run >= min_sims:
                current_win = sum_win / float(simulations_run)
                stderr = np.sqrt(
                    np.clip(current_win * (1.0 - current_win), 0.0, 0.25)
                    / float(simulations_run)
                )
                half_width = z_score * stderr
                top_idx = np.argsort(current_win)[::-1][:ci_top_n]
                win_ci_half_width_top_n = (
                    float(np.max(half_width[top_idx])) if top_idx.size > 0 else 0.0
                )
                if win_ci_half_width_top_n <= ci_half_width_target:
                    adaptive_stopped_early = simulations_run < max_simulations
                    ci_target_met = True
                    break

        if adaptive_enabled and win_ci_half_width_top_n is not None and not ci_target_met:
            ci_target_met = win_ci_half_width_top_n <= ci_half_width_target

        divisor = float(simulations_run)
        recommended_simulations: int | None = None
        stop_reason = "max_simulations_reached"
        if adaptive_enabled:
            if ci_target_met:
                stop_reason = "ci_target_met"
            elif win_ci_half_width_top_n is not None and win_ci_half_width_top_n > 0:
                projected = float(simulations_run) * (
                    (win_ci_half_width_top_n / ci_half_width_target) ** 2
                )
                recommended_simulations = int(np.ceil(projected / 500.0) * 500)
                recommended_simulations = max(
                    max_simulations + 500,
                    min(recommended_simulations, 1_000_000),
                )
        else:
            stop_reason = "fixed_cap_mode"

        return SimulationOutputs(
            win_probability=sum_win / divisor,
            top_3_probability=sum_top_3 / divisor,
            top_5_probability=sum_top_5 / divisor,
            top_10_probability=sum_top_10 / divisor,
            mean_finish=sum_finish / divisor,
            mean_total_relative_to_field=sum_total / divisor,
            made_cut_probability=sum_made_cut / divisor,
            simulations_run=simulations_run,
            adaptive_stopped_early=adaptive_stopped_early,
            win_ci_half_width_top_n=win_ci_half_width_top_n,
            ci_target_met=ci_target_met,
            recommended_simulations=recommended_simulations,
            stop_reason=stop_reason,
        )

    def _simulate_chunk(
        self,
        rng: np.random.Generator,
        chunk_size: int,
        cut_size: int,
        n_players: int,
        n_states: int,
        config: MarkovSimulationConfig,
        initial_totals: np.ndarray,
        round_fractions: np.ndarray,
        round_numbers: np.ndarray,
        transition_cdfs: dict[float, np.ndarray],
        delta_values: np.ndarray,
    ) -> _ChunkMetrics:
        totals = np.repeat(initial_totals.reshape(1, n_players), repeats=chunk_size, axis=0)
        states = np.clip(np.rint(totals), config.min_state, config.max_state).astype(np.int16)
        alive = np.ones((chunk_size, n_players), dtype=bool)
        player_idx = np.arange(n_players, dtype=np.int32)[None, :]

        for step_idx, fraction in enumerate(round_fractions):
            if fraction <= 1e-8:
                continue
            key = self._fraction_key(float(fraction))
            cdfs = transition_cdfs[key]
            state_idx = np.clip(states - config.min_state, 0, n_states - 1)
            cdf_rows = cdfs[player_idx, state_idx]
            draws = rng.random((chunk_size, n_players))
            delta_idx = (draws[..., None] > cdf_rows).sum(axis=2)
            deltas = delta_values[delta_idx].astype(np.float64)

            if config.round_shock_sigma > 1e-8:
                shared_round_shock = rng.normal(
                    loc=0.0,
                    scale=config.round_shock_sigma * np.sqrt(fraction),
                    size=(chunk_size, 1),
                )
                deltas += shared_round_shock

            deltas = np.where(alive, deltas, 0.0)
            totals += deltas
            states = np.clip(
                np.rint(states.astype(np.float64) + deltas),
                config.min_state,
                config.max_state,
            ).astype(np.int16)

            round_number = int(round_numbers[step_idx])
            if round_number == config.cut_after_round and n_players > cut_size:
                sorted_totals = np.sort(totals, axis=1)
                cut_line = sorted_totals[:, cut_size - 1]
                alive = totals <= cut_line[:, None]

        return _ChunkMetrics(
            simulations=chunk_size,
            win_probability=self._winner_probability(totals),
            top_3_probability=self._top_k_probability(totals, 3, rng),
            top_5_probability=self._top_k_probability(totals, 5, rng),
            top_10_probability=self._top_k_probability(totals, 10, rng),
            mean_finish=self._mean_finish(totals, rng),
            mean_total_relative_to_field=totals.mean(axis=0),
            made_cut_probability=alive.mean(axis=0),
        )

    @staticmethod
    def _fraction_key(value: float) -> float:
        return float(np.round(value, 6))

    @staticmethod
    def _resolve_initial_totals(inputs: SimulationInputs, n_players: int) -> np.ndarray:
        if inputs.initial_totals is None:
            return np.zeros(n_players, dtype=np.float64)
        arr = np.asarray(inputs.initial_totals, dtype=np.float64).reshape(-1)
        if arr.shape[0] != n_players:
            return np.zeros(n_players, dtype=np.float64)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        return np.clip(arr, -200.0, 200.0)

    @staticmethod
    def _resolve_mean_reversion(
        inputs: SimulationInputs,
        n_players: int,
        default_value: float,
    ) -> np.ndarray:
        if inputs.mean_reversion is None:
            return np.full(n_players, default_value, dtype=np.float64)
        arr = np.asarray(inputs.mean_reversion, dtype=np.float64).reshape(-1)
        if arr.shape[0] != n_players:
            return np.full(n_players, default_value, dtype=np.float64)
        arr = np.where(np.isfinite(arr), arr, default_value)
        return np.clip(arr, 0.0, 0.8)

    @staticmethod
    def _resolve_round_plan(
        inputs: SimulationInputs,
        default_rounds: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if inputs.round_fractions is not None:
            fractions = np.asarray(inputs.round_fractions, dtype=np.float64).reshape(-1)
            fractions = np.where(np.isfinite(fractions), fractions, 0.0)
            fractions = np.clip(fractions, 0.0, 1.0)
        else:
            fractions = np.ones(default_rounds, dtype=np.float64)

        if inputs.round_numbers is not None:
            numbers = np.asarray(inputs.round_numbers, dtype=np.int16).reshape(-1)
            if numbers.shape[0] != fractions.shape[0]:
                numbers = np.arange(1, fractions.shape[0] + 1, dtype=np.int16)
        else:
            numbers = np.arange(1, fractions.shape[0] + 1, dtype=np.int16)
        return fractions, numbers

    @staticmethod
    def _build_transition_cdfs(
        mu_round: np.ndarray,
        sigma_round: np.ndarray,
        state_values: np.ndarray,
        delta_values: np.ndarray,
        mean_reversion: np.ndarray | float,
    ) -> np.ndarray:
        player_count = mu_round.shape[0]
        sigma = np.clip(sigma_round.reshape(player_count, 1, 1), 0.8, 8.0)
        player_mu = mu_round.reshape(player_count, 1, 1)
        state_grid = state_values.reshape(1, state_values.size, 1)
        delta_grid = delta_values.reshape(1, 1, delta_values.size)

        mean_reversion_arr = np.asarray(mean_reversion, dtype=np.float64).reshape(-1)
        if mean_reversion_arr.size == 1:
            reversion_grid = np.full((player_count, 1, 1), mean_reversion_arr.item())
        elif mean_reversion_arr.size == player_count:
            reversion_grid = mean_reversion_arr.reshape(player_count, 1, 1)
        else:
            raise ValueError("mean_reversion must be scalar or length equal to player count.")

        shifted_mean = player_mu - (reversion_grid * state_grid)
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
