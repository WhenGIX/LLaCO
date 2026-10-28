# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Q_log semantic scheduler with an optional gamma-uniform training mixture."""

from dataclasses import dataclass
import math

import torch

from ...nemotron_labs_diffusion_spherical_protocol import (
    GAMMA_UNIFORM_FRACTION,
    T_UNIFORM_ENDPOINT_INCLUSIVE,
    T_UNIFORM_FRACTION,
    T_UNIFORM_GRID_DENOMINATOR,
    T_UNIFORM_GRID_SIZE,
    T_UNIFORM_GRID_STEP,
)


LOGISTIC_K = 63.7732
LOGISTIC_GAMMA_C = 0.966621
EPSILON_UNIFORM = 0.1
INVERSE_GRID_SIZE = 10001
GAMMA_UNIFORM_PROBABILITY = GAMMA_UNIFORM_FRACTION
T_UNIFORM_PROBABILITY = T_UNIFORM_FRACTION


def _logistic_survival(gamma: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(-LOGISTIC_K * (gamma - LOGISTIC_GAMMA_C))


def q_fit(gamma: torch.Tensor | float) -> torch.Tensor:
    """Endpoint-normalized Logistic survival quality from the Q_log fit."""
    value = torch.as_tensor(gamma)
    work_dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
    bounded = value.to(dtype=work_dtype).clamp(0.0, 1.0)
    s0 = _logistic_survival(torch.zeros((), dtype=work_dtype, device=bounded.device))
    s1 = _logistic_survival(torch.ones((), dtype=work_dtype, device=bounded.device))
    result = ((_logistic_survival(bounded) - s1) / (s0 - s1)).clamp(0.0, 1.0)
    result = torch.where(bounded == 0.0, torch.ones_like(result), result)
    result = torch.where(bounded == 1.0, torch.zeros_like(result), result)
    return result.to(dtype=value.dtype) if value.is_floating_point() else result


def t_from_gamma(gamma: torch.Tensor | float) -> torch.Tensor:
    """Map spherical progress gamma to semantic time t."""
    value = torch.as_tensor(gamma)
    work_dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
    bounded = value.to(dtype=work_dtype).clamp(0.0, 1.0)
    result = (1.0 - EPSILON_UNIFORM) * (1.0 - q_fit(bounded).to(work_dtype)) + EPSILON_UNIFORM * bounded
    result = torch.where(bounded == 0.0, torch.zeros_like(result), result)
    result = torch.where(bounded == 1.0, torch.ones_like(result), result)
    return result.to(dtype=value.dtype) if value.is_floating_point() else result


def _inverse_table(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    gamma_grid = torch.linspace(0.0, 1.0, INVERSE_GRID_SIZE, device=device, dtype=dtype)
    return t_from_gamma(gamma_grid).to(dtype=dtype), gamma_grid


def gamma_from_t(t: torch.Tensor | float) -> torch.Tensor:
    """Invert ``t(gamma)`` using the shared monotone piecewise-linear table.

    This is the inverse convention used by both training and inference.
    It intentionally does not use ``epsilon*t + (1-epsilon)*gamma_Q(t)``.
    """
    value = torch.as_tensor(t)
    work_dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
    bounded = value.to(dtype=work_dtype).clamp(0.0, 1.0)
    t_grid, gamma_grid = _inverse_table(bounded.device, work_dtype)
    flat = bounded.reshape(-1)
    indices = torch.searchsorted(t_grid, flat, right=True).clamp(1, INVERSE_GRID_SIZE - 1)
    t0 = t_grid[indices - 1]
    t1 = t_grid[indices]
    g0 = gamma_grid[indices - 1]
    g1 = gamma_grid[indices]
    fraction = (flat - t0) / (t1 - t0).clamp_min(torch.finfo(work_dtype).eps)
    result = (g0 + fraction * (g1 - g0)).reshape(bounded.shape)
    result = torch.where(bounded == 0.0, torch.zeros_like(result), result)
    result = torch.where(bounded == 1.0, torch.ones_like(result), result)
    return result.to(dtype=value.dtype) if value.is_floating_point() else result


@dataclass(frozen=True)
class TimeSample:
    """Consistent t/gamma block coordinates; the model selects its configured input."""

    gamma: torch.Tensor
    t: torch.Tensor
    mode: str
    gamma_uniform: torch.Tensor
    decode: torch.Tensor | None = None


@dataclass(frozen=True)
class SphericalScheduler:
    """Stateless t-grid sampler, optionally mixed with continuous uniform gamma."""

    t_min: float = 0.0
    t_max: float = 1.0
    t_grid_size: int = T_UNIFORM_GRID_SIZE
    t_step: float = T_UNIFORM_GRID_STEP
    endpoint_inclusive: bool = T_UNIFORM_ENDPOINT_INCLUSIVE
    gamma_min: float = 0.0
    gamma_max: float = 1.0
    gamma_uniform_probability: float = GAMMA_UNIFORM_PROBABILITY
    decode_enabled: bool = False
    decode_probability: float = 0.2
    decode_gamma_min: float = 0.85
    decode_gamma_max: float = 0.95

    def __post_init__(self) -> None:
        if self.decode_enabled:
            from .decode_protocol import validate_decode_config

            validate_decode_config({
                "spherical_decode_training": True,
                "spherical_decode_probability": self.decode_probability,
                "spherical_decode_gamma_min": self.decode_gamma_min,
                "spherical_decode_gamma_max": self.decode_gamma_max,
                "spherical_gamma_uniform_probability": self.gamma_uniform_probability,
            })
        if not 0 <= self.t_min < self.t_max <= 1 or not 0 <= self.gamma_min < self.gamma_max <= 1:
            raise ValueError("scheduler ranges must be ordered within [0, 1]")
        if (self.t_min, self.t_max, self.t_grid_size, self.t_step, self.endpoint_inclusive) != (
            0.0, 1.0, T_UNIFORM_GRID_SIZE, T_UNIFORM_GRID_STEP, True
        ):
            raise ValueError("t-uniform sampling requires the inclusive 1001-point grid index/1000")
        if not math.isfinite(self.gamma_uniform_probability) or not 0 <= self.gamma_uniform_probability <= 1:
            raise ValueError("gamma_uniform_probability must be finite and in [0, 1]")

    def sample(self, shape: tuple[int, ...], *, device=None, generator=None) -> TimeSample:
        """Sample block coordinates, preserving the legacy RNG path at probability0."""
        if self.decode_enabled:
            return self._sample_decode(shape, device=device, generator=generator)
        t_index = torch.randint(0, self.t_grid_size, shape, device=device, generator=generator)
        t_uniform = t_index.to(torch.float32) / T_UNIFORM_GRID_DENOMINATOR
        gamma = gamma_from_t(t_uniform)
        gamma_branch = torch.zeros(shape, dtype=torch.bool, device=device)
        if self.gamma_uniform_probability:
            gamma_branch = torch.rand(shape, device=device, generator=generator) < self.gamma_uniform_probability
            gamma_uniform = self.gamma_min + (self.gamma_max - self.gamma_min) * torch.rand(
                shape, device=device, generator=generator
            )
            gamma = torch.where(gamma_branch, gamma_uniform, gamma)
            t_uniform = torch.where(gamma_branch, t_from_gamma(gamma_uniform), t_uniform)
        return TimeSample(
            gamma=gamma,
            t=t_uniform,
            mode="mixed" if self.gamma_uniform_probability else "t_uniform",
            gamma_uniform=gamma_branch,
        )


    def _sample_decode(self, shape: tuple[int, ...], *, device=None, generator=None) -> TimeSample:
        """Keep physical corruption gamma independent of the t=0 decoder label."""
        t = torch.randint(1, self.t_grid_size, shape, device=device, generator=generator).float()
        t = t / T_UNIFORM_GRID_DENOMINATOR
        gamma = gamma_from_t(t)
        selected = torch.rand(shape, device=device, generator=generator) < self.decode_probability
        decoder_gamma = self.decode_gamma_min + (self.decode_gamma_max - self.decode_gamma_min) * torch.rand(
            shape, device=device, generator=generator
        )
        return TimeSample(
            gamma=torch.where(selected, decoder_gamma, gamma),
            t=torch.where(selected, torch.zeros_like(t), t),
            mode="decode_mixture",
            gamma_uniform=torch.zeros_like(selected),
            decode=selected,
        )


def validate_scheduler() -> None:
    """Validate endpoints, monotonicity, and the shared inverse convention."""
    gamma = torch.linspace(0.0, 1.0, 1001)
    times = t_from_gamma(gamma)
    if not torch.allclose(times[[0, -1]], torch.tensor([0.0, 1.0]), atol=1e-6):
        raise AssertionError("t(gamma) endpoints must be 0 and 1")
    if not torch.all(torch.diff(times) > 0):
        raise AssertionError("t(gamma) must be strictly increasing")
    recovered = gamma_from_t(times)
    if not torch.allclose(recovered, gamma, atol=2e-4):
        raise AssertionError("gamma(t(gamma)) inverse error exceeds table tolerance")


GammaSample = TimeSample


__all__ = [
    "EPSILON_UNIFORM",
    "GAMMA_UNIFORM_PROBABILITY",
    "GammaSample",
    "INVERSE_GRID_SIZE",
    "LOGISTIC_GAMMA_C",
    "LOGISTIC_K",
    "SphericalScheduler",
    "T_UNIFORM_PROBABILITY",
    "TimeSample",
    "gamma_from_t",
    "q_fit",
    "t_from_gamma",
    "validate_scheduler",
]
