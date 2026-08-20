"""Probability and communication helpers used by QS and RelayTree."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def lattice_based_quantization_torch(probabilities: torch.Tensor, q_level: int) -> torch.Tensor:
    """Project a probability vector onto the paper's integer simplex lattice."""

    probabilities = torch.as_tensor(
        probabilities,
        dtype=torch.float32,
        device=getattr(probabilities, "device", None),
    )
    probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
    probabilities = torch.clamp(probabilities, min=0.0)
    total = probabilities.sum()
    if not torch.isfinite(total) or total <= 0:
        probabilities = torch.full_like(probabilities, 1.0 / probabilities.numel())
    else:
        probabilities = probabilities / total

    lattice = torch.floor(float(q_level) * probabilities + 0.5).to(torch.int64)
    difference = int(q_level) - int(lattice.sum().item())
    if difference:
        rounding_error = lattice.float() - float(q_level) * probabilities
        count = min(abs(difference), rounding_error.numel())
        if difference > 0:
            indices = torch.topk(rounding_error, count, largest=False).indices
            lattice[indices] += 1
        else:
            indices = torch.topk(rounding_error, count, largest=True).indices
            lattice[indices] = torch.clamp(lattice[indices] - 1, min=0)

    quantized = torch.clamp(lattice.float() / float(q_level), min=0.0)
    return quantized / quantized.sum()


def calculate_quantization_bits(q_level: int, vocab_size: int) -> int:
    """Bits needed to encode one lattice-quantized probability vector."""

    if q_level <= 0:
        return 0
    n = q_level + vocab_size - 1
    k = min(q_level, n - q_level)
    log2_combinations = (
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    ) / math.log(2)
    return math.ceil(log2_combinations)


def norm_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Convert logits to a full next-token distribution."""

    if temperature <= 0:
        indices = logits.argmax(dim=-1, keepdim=True)
        return torch.zeros_like(logits).scatter_(1, indices, 1.0)
    return F.softmax(logits / temperature, dim=-1)


def quantized_logits(
    logits: torch.Tensor,
    q_level: int,
    temperature: float,
) -> torch.Tensor:
    probabilities = norm_logits(logits, temperature)
    if q_level <= 0:
        return probabilities
    return torch.stack(
        [
            lattice_based_quantization_torch(row, q_level)
            for row in probabilities
        ],
        dim=0,
    )


def sample(probabilities: torch.Tensor) -> torch.Tensor:
    probabilities = torch.clamp(probabilities, min=1e-9)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    return torch.multinomial(probabilities, num_samples=1)


def max_fn(values: torch.Tensor) -> torch.Tensor:
    """Normalize the positive part of a residual distribution."""

    positive = torch.clamp(values, min=0.0)
    totals = positive.sum(dim=-1, keepdim=True)
    fallback = torch.full_like(positive, 1.0 / positive.size(-1))
    return torch.where(totals > 0, positive / totals, fallback)
