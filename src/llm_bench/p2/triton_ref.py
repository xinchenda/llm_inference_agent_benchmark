from __future__ import annotations

import torch


def vector_add_reference(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left + right


def rmsnorm_reference(
    values: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    normalized = values * torch.rsqrt(values.pow(2).mean(dim=-1, keepdim=True) + eps)
    return normalized * weight
