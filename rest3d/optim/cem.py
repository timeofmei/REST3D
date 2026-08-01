"""Backend-neutral cross-entropy optimizer for rigid-body pose deltas."""

from __future__ import annotations

import numpy as np


class CEMOptimizer:
    """Optimize one or more 6-DoF pose deltas from scalar rewards.

    The configuration object intentionally follows ``StableSceneCfg`` so the
    Isaac Gym backend keeps its existing call sites while Isaac Lab can import
    the optimizer without importing either simulator runtime.
    """

    def __init__(self, args, n_objects=1, seed=None, rng=None):
        if n_objects < 1:
            raise ValueError("n_objects must be positive")
        action_dimension = int(args.act_dim)
        if action_dimension != 6:
            raise ValueError(
                "CEM pose actions require 6 values per object "
                "(translation xyz and axis-angle xyz)"
            )
        self.args = args
        self.n_objects = int(n_objects)
        self.act_dim = self.n_objects * action_dimension
        self.pop_size = int(args.cem_pop_size)
        if self.pop_size < 1:
            raise ValueError("cem_pop_size must be positive")
        elite_fraction = float(args.cem_elite_frac)
        if not 0.0 < elite_fraction <= 1.0:
            raise ValueError("cem_elite_frac must be in (0, 1]")
        self.n_elite = max(1, int(self.pop_size * elite_fraction))
        self.cem_iters_joint = int(args.cem_iters_joint)
        self.cem_iters_subtree = int(args.cem_iters_subtree)
        self.reward_threshold = float(getattr(args, "reward_threshold", -0.01))
        self.keep_best = bool(getattr(args, "keep_best", False))
        self.update_use_only_best = bool(
            getattr(args, "update_use_only_best", False)
        )
        self.std_update_mode = getattr(args, "std_update_mode", "topk_std")
        self.decay_std_rate = float(getattr(args, "decay_std_rate", 0.95))
        self.cem_final_best = bool(getattr(args, "cem_final_best", False))
        self._rng = rng
        if self._rng is None:
            configured_seed = getattr(args, "cem_seed", None) if seed is None else seed
            self._rng = (
                np.random.default_rng(configured_seed)
                if configured_seed is not None
                else np.random
            )

        init_mean_6 = np.array(
            [
                getattr(args, "init_trans_x_mean", 0.0),
                getattr(args, "init_trans_y_mean", 0.0),
                getattr(args, "init_trans_z_mean", 0.0),
                getattr(args, "init_rot_roll_mean", 0.0),
                getattr(args, "init_rot_pitch_mean", 0.0),
                getattr(args, "init_rot_yaw_mean", 0.0),
            ],
            dtype=np.float32,
        )
        init_std_6 = np.array(
            [
                getattr(args, "init_trans_x_std", 0.5),
                getattr(args, "init_trans_y_std", 0.1),
                getattr(args, "init_trans_z_std", 0.5),
                getattr(args, "init_rot_roll_std", 0.5),
                getattr(args, "init_rot_pitch_std", 0.5),
                getattr(args, "init_rot_yaw_std", 0.5),
            ],
            dtype=np.float32,
        )
        if not np.isfinite(init_mean_6).all():
            raise ValueError("initial CEM mean must be finite")
        if not np.isfinite(init_std_6).all() or np.any(init_std_6 <= 0.0):
            raise ValueError("initial CEM standard deviations must be finite and positive")
        self._init_mean = np.tile(init_mean_6, self.n_objects)
        self._init_std = np.tile(init_std_6, self.n_objects)
        self._best_action = None
        self._best_reward = -np.inf
        self.reset()

    def reset(self):
        """Reset the sampling distribution while preserving the all-time best."""

        self.mean = self._init_mean.copy()
        self.std = self._init_std.copy()
        self._last_elites = None
        self._prev_elites = None

    def warm_start(self, mode, alpha=0.8):
        prev_mean = self.mean.copy()
        prev_std = self.std.copy()
        prev_elites = self._last_elites
        self._last_elites = None
        if mode == "prev_mean":
            self.mean = prev_mean
            self.std = self._init_std.copy()
            self._prev_elites = None
        elif mode == "prev_mean_std":
            self.mean = prev_mean
            self.std = prev_std
            self._prev_elites = None
        elif mode == "momentum":
            if not 0.0 <= alpha <= 1.0:
                raise ValueError("warm-start momentum alpha must be in [0, 1]")
            self.mean = alpha * prev_mean + (1.0 - alpha) * self._init_mean
            self.std = self._init_std.copy()
            self._prev_elites = None
        elif mode == "icem":
            self.mean = prev_mean
            self.std = prev_std
            self._prev_elites = prev_elites
        elif mode == "warm_start_best":
            self.mean = (
                self._best_action.copy()
                if self._best_action is not None
                else prev_mean
            )
            self.std = self._init_std.copy()
            self._prev_elites = None
        else:
            raise ValueError(f"unsupported CEM warm-start mode: {mode}")

    def sample(self, n=None):
        n = self.pop_size if n is None else int(n)
        if n < 1:
            raise ValueError("sample count must be positive")
        samples = self._rng.standard_normal((n, self.act_dim)) * self.std + self.mean
        prev = self._prev_elites
        if prev is not None:
            n_carry = min(len(prev), n)
            samples[-n_carry:] = prev[-n_carry:]
            self._prev_elites = None
        return samples

    @staticmethod
    def _nes_rank_utilities(n):
        return (np.arange(n, dtype=np.float32) - (n - 1) * 0.5) / n

    def update(self, samples, rewards):
        samples = np.asarray(samples)
        rewards = np.asarray(rewards)
        if samples.ndim != 2 or samples.shape[1] != self.act_dim:
            raise ValueError(
                f"samples must have shape [population, {self.act_dim}], got {samples.shape}"
            )
        if rewards.ndim != 1 or rewards.shape[0] != samples.shape[0]:
            raise ValueError("rewards must contain one scalar per sample")
        if samples.shape[0] < self.n_elite:
            raise ValueError(
                f"update needs at least {self.n_elite} samples, got {samples.shape[0]}"
            )
        if not np.isfinite(samples).all() or not np.isfinite(rewards).all():
            raise ValueError("CEM samples and rewards must be finite")

        idx_best = int(np.argmax(rewards))
        if rewards[idx_best] > self._best_reward:
            self._best_reward = float(rewards[idx_best])
            self._best_action = samples[idx_best].copy()

        update_mode = getattr(self.args, "cem_update_mode", "cem")
        if update_mode == "nes":
            self._update_nes(samples, rewards)
        elif update_mode == "cem":
            self._update_cem(samples, rewards, idx_best)
        else:
            raise ValueError(f"unsupported CEM update mode: {update_mode}")

    def _update_nes(self, samples, rewards):
        if self.keep_best and self._best_action is not None:
            augmented_samples = np.vstack([samples, self._best_action[None]])
            augmented_rewards = np.append(rewards, self._best_reward)
        else:
            augmented_samples = samples
            augmented_rewards = rewards
        sort_idx = np.argsort(augmented_rewards)
        utilities = self._nes_rank_utilities(len(augmented_samples))
        ranked_utilities = np.empty(len(augmented_samples), dtype=np.float32)
        ranked_utilities[sort_idx] = utilities
        dimension = self.act_dim
        eta_mu = float(getattr(self.args, "nes_lr_mu", 1.0))
        eta_sigma = getattr(self.args, "nes_lr_sigma", None)
        if eta_sigma is None:
            eta_sigma = (3.0 + np.log(dimension)) / (5.0 * np.sqrt(dimension))
        epsilon = (augmented_samples - self.mean) / (self.std + 1.0e-8)
        grad_mu = (ranked_utilities[:, None] * epsilon).mean(axis=0)
        grad_sigma = (
            ranked_utilities[:, None] * (epsilon**2 - 1.0)
        ).mean(axis=0)
        self.mean = self.mean + eta_mu * self.std * grad_mu
        self.std = np.maximum(
            self.std * np.exp(float(eta_sigma) * 0.5 * grad_sigma), 1.0e-5
        )
        elite_idx = np.argpartition(rewards, -self.n_elite)[-self.n_elite :]
        self._last_elites = samples[elite_idx].copy()

    def _update_cem(self, samples, rewards, idx_best):
        elite_idx = np.argpartition(rewards, -self.n_elite)[-self.n_elite :]
        elites = samples[elite_idx].copy()
        if self.keep_best and self._best_action is not None:
            worst = int(np.argmin(rewards[elite_idx]))
            elites[worst] = self._best_action
        if self.update_use_only_best:
            self.mean = samples[idx_best].copy()
            if self.std_update_mode == "same_std":
                pass
            elif self.std_update_mode == "decay_std":
                self.std = np.maximum(self.std * self.decay_std_rate, 1.0e-5)
            else:
                self.std = elites.std(axis=0) + 1.0e-5
        else:
            self.mean = elites.mean(axis=0)
            self.std = elites.std(axis=0) + 1.0e-5
        self._last_elites = elites.copy()

    def converged(self, rewards):
        rewards = np.asarray(rewards)
        if rewards.size == 0 or not np.isfinite(rewards).all():
            raise ValueError("convergence rewards must be non-empty and finite")
        return float(np.max(rewards)) > self.reward_threshold

    def get_final_action(self):
        if self._best_action is not None:
            return self._best_action.copy(), self.mean.copy()
        return None, self.mean.copy()
