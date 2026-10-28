"""Fixed identity-calibrated gamma <-> semantic-time scheduler."""

import torch


A_EXPONENT = 6.1621
T_EXPONENT = 1.5617


def gamma_from_t(t):
    """Map semantic time to spherical path progress."""
    t = torch.as_tensor(t)
    work_dtype = torch.float64 if t.dtype == torch.float64 else torch.float32
    value = t.to(dtype=work_dtype).clamp(0.0, 1.0)
    inner = -torch.expm1(torch.log1p(-value) / T_EXPONENT)
    result = inner.clamp(0.0, 1.0).pow(1.0 / A_EXPONENT)
    result = torch.where(
        value == 0.0,
        torch.zeros_like(result),
        torch.where(value == 1.0, torch.ones_like(result), result),
    )
    return result.to(dtype=t.dtype) if t.is_floating_point() else result


def t_from_gamma(gamma):
    """Map spherical path progress to semantic time."""
    gamma = torch.as_tensor(gamma)
    work_dtype = torch.float64 if gamma.dtype == torch.float64 else torch.float32
    value = gamma.to(dtype=work_dtype).clamp(0.0, 1.0)
    result = -torch.expm1(T_EXPONENT * torch.log1p(-value.pow(A_EXPONENT).clamp(0.0, 1.0)))
    result = torch.where(
        value == 0.0,
        torch.zeros_like(result),
        torch.where(value == 1.0, torch.ones_like(result), result),
    )
    return result.to(dtype=gamma.dtype) if gamma.is_floating_point() else result


def validate_scheduler() -> None:
    """Validate the scheduler endpoints and monotonicity."""
    grid = torch.linspace(0, 1, 1001)
    if not torch.allclose(gamma_from_t(torch.tensor(0.0)), torch.tensor(0.0), atol=1e-6):
        raise AssertionError("gamma(0) must be zero")
    if not torch.allclose(gamma_from_t(torch.tensor(1.0)), torch.tensor(1.0), atol=1e-6):
        raise AssertionError("gamma(1) must be one")
    if not torch.all(torch.diff(gamma_from_t(grid)) >= -1e-7):
        raise AssertionError("scheduler is not monotonic")
