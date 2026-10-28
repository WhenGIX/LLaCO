# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Identity-calibrated semantic-time schedulers for RMS linear paths."""

from __future__ import annotations

from dataclasses import dataclass

import torch


SCHEDULER_COEFFICIENTS = {
    "scale-matched": (7.163168719611431, 1.4689579228072418),
    "unit-noise": (7.05364890067688, 1.4554665034447531),
    "equal-mask-rms": (7.571661005577266, 1.5657146138043683),
}
A_EXPONENT, T_EXPONENT = SCHEDULER_COEFFICIENTS["scale-matched"]


@dataclass(frozen=True)
class IdentityScheduler:
    """Map identity-calibrated semantic time to the linear-path coordinate."""

    terminal_mode: str
    a: float
    b: float

    @classmethod
    def for_terminal(cls, terminal_mode: str) -> "IdentityScheduler":
        if terminal_mode not in SCHEDULER_COEFFICIENTS:
            raise ValueError(f"no scheduler is registered for terminal mode {terminal_mode!r}")
        a, b = SCHEDULER_COEFFICIENTS[terminal_mode]
        return cls(terminal_mode, a, b)

    def gamma_from_t(self, t):
        """Convert semantic time ``t`` to path progress ``gamma``."""
        t = torch.as_tensor(t)
        one_minus_t = (1.0 - t).clamp(0.0, 1.0)
        return (1.0 - one_minus_t.pow(1.0 / self.b)).clamp(0.0, 1.0).pow(1.0 / self.a)

    def t_from_gamma(self, gamma):
        """Convert path progress ``gamma`` to semantic time ``t``."""
        gamma = torch.as_tensor(gamma).clamp(0.0, 1.0)
        return 1.0 - (1.0 - gamma.pow(self.a)).clamp(0.0, 1.0).pow(self.b)

    def metadata(self) -> dict:
        return {
            "family": "kumaraswamy_identity_survival",
            "terminal_mode": self.terminal_mode,
            "a": self.a,
            "b": self.b,
            "condition": "gamma",
        }


def gamma_from_t(t, terminal_mode: str = "scale-matched"):
    """Convert semantic time using the scheduler for ``terminal_mode``."""
    return IdentityScheduler.for_terminal(terminal_mode).gamma_from_t(t)


def t_from_gamma(gamma, terminal_mode: str = "scale-matched"):
    """Convert path progress using the scheduler for ``terminal_mode``."""
    return IdentityScheduler.for_terminal(terminal_mode).t_from_gamma(gamma)


def validate_scheduler(terminal_mode: str = "scale-matched", atol: float = 3e-6) -> None:
    """Validate scheduler endpoints, monotonicity, and inverse accuracy."""
    scheduler = IdentityScheduler.for_terminal(terminal_mode)
    grid = torch.linspace(0.0, 1.0, 10001)
    mapped = scheduler.gamma_from_t(grid)
    if not torch.allclose(mapped[0], torch.tensor(0.0), atol=atol):
        raise AssertionError("gamma(0) must be zero")
    if not torch.allclose(mapped[-1], torch.tensor(1.0), atol=atol):
        raise AssertionError("gamma(1) must be one")
    if not torch.all(torch.diff(mapped) >= -1e-7):
        raise AssertionError("scheduler is not monotonic")
    values = torch.linspace(0.0, 1.0, 1001)
    if not torch.allclose(scheduler.t_from_gamma(scheduler.gamma_from_t(values)), values, atol=atol, rtol=atol):
        raise AssertionError("t/gamma inverse mismatch")
