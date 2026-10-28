"""Numerically stable spherical geometry used by continuous Nemotron flow."""

import torch


def normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize vectors and fail loudly for degenerate rows."""
    if not bool(torch.isfinite(x).all()):
        raise ValueError("cannot normalize a non-finite spherical vector")
    norm_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    norm = x.to(dtype=norm_dtype).norm(dim=-1, keepdim=True)
    if not bool(torch.isfinite(norm).all()):
        raise ValueError("cannot normalize a vector with a non-finite norm")
    if bool((norm <= eps).any()):
        raise ValueError("cannot normalize a degenerate spherical vector")
    return x / norm.to(dtype=x.dtype)


def terminal_noise(
    random_direction: torch.Tensor, mask_direction: torch.Tensor, eps: float = 1e-6, generator=None
) -> torch.Tensor:
    """Construct z=normalize(n+m), resampling degenerate rows in-place."""
    if not bool(torch.isfinite(random_direction).all()) or not bool(torch.isfinite(mask_direction).all()):
        raise ValueError("terminal noise directions must be finite")
    z = random_direction + mask_direction
    norm_dtype = torch.float64 if z.dtype == torch.float64 else torch.float32
    norm = z.to(dtype=norm_dtype).norm(dim=-1, keepdim=True)
    if not bool(torch.isfinite(norm).all()):
        raise ValueError("terminal noise has a non-finite norm")
    bad = norm.squeeze(-1) <= eps
    while bool(bad.any()):
        replacement = torch.randn(
            (int(bad.sum()), random_direction.shape[-1]),
            device=random_direction.device,
            dtype=norm_dtype,
            generator=generator,
        )
        replacement = replacement / replacement.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        z = z.clone()
        z[bad] = replacement.to(z.dtype) + mask_direction
        norm = z.to(dtype=norm_dtype).norm(dim=-1, keepdim=True)
        if not bool(torch.isfinite(norm).all()):
            raise ValueError("resampled terminal noise has a non-finite norm")
        bad = norm.squeeze(-1) <= eps
    return z / norm.clamp_min(eps).to(dtype=z.dtype)


def slerp(x0: torch.Tensor, x1: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Spherical linear interpolation with a normalized-lerp small-angle limit."""
    output_dtype = x0.dtype
    work_dtype = torch.float64 if x0.dtype == torch.float64 or x1.dtype == torch.float64 else torch.float32
    x0_work = normalize(x0).to(dtype=work_dtype)
    x1_work = normalize(x1).to(device=x0.device, dtype=work_dtype)
    x0_work = x0_work / x0_work.norm(dim=-1, keepdim=True).clamp_min(eps)
    x1_work = x1_work / x1_work.norm(dim=-1, keepdim=True).clamp_min(eps)

    g = torch.as_tensor(gamma, device=x0.device, dtype=work_dtype)
    if not bool(torch.isfinite(g).all()):
        raise ValueError("slerp gamma must be finite")
    while g.ndim < x0_work.ndim:
        g = g.unsqueeze(-1)

    dot = (x0_work * x1_work).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    projected = x1_work - dot * x0_work
    projected_norm = projected.norm(dim=-1, keepdim=True)
    projected_tangent = projected / projected_norm.clamp_min(eps)

    # An exactly antipodal pair has no endpoint-defined tangent. Choose one
    # deterministically, but retain the projected x1 tangent whenever it exists.
    basis_idx = x0_work.abs().argmin(dim=-1)
    basis = torch.zeros_like(x0_work).scatter(-1, basis_idx.unsqueeze(-1), 1.0)
    fallback_tangent = basis - (basis * x0_work).sum(dim=-1, keepdim=True) * x0_work
    fallback_tangent = fallback_tangent / fallback_tangent.norm(dim=-1, keepdim=True).clamp_min(eps)
    antipodal_degenerate = (dot < 0.0) & (projected_norm <= eps)
    tangent = torch.where(antipodal_degenerate, fallback_tangent, projected_tangent)
    theta = torch.atan2(projected_norm, dot)
    theta = torch.where(antipodal_degenerate, torch.full_like(theta, torch.pi), theta)
    curved = torch.cos(g * theta) * x0_work + torch.sin(g * theta) * tangent

    linear_raw = (1.0 - g) * x0_work + g * x1_work
    linear = linear_raw / linear_raw.norm(dim=-1, keepdim=True).clamp_min(eps)
    same_direction_degenerate = (dot >= 0.0) & (projected_norm <= eps)
    result = torch.where(same_direction_degenerate, linear, curved)
    result = torch.where(g == 0.0, x0_work, torch.where(g == 1.0, x1_work, result))
    result = result / result.norm(dim=-1, keepdim=True).clamp_min(eps)
    return result.to(dtype=output_dtype)


def random_sphere(shape: tuple[int, ...], device=None, dtype=torch.bfloat16, generator=None) -> torch.Tensor:
    """Sample uniform directions on a sphere."""
    x = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    return (x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)).to(dtype)
