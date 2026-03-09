"""
Sampling strategies for token generation.

Implements temperature scaling, top-k filtering, and top-p (nucleus) sampling,
inspired by vLLM's sampling approach.
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class SamplingParams:
    """Parameters controlling the sampling behavior."""
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(f"top_k must be -1 (disabled) or a positive integer, got {self.top_k}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0


def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by temperature.

    A lower temperature sharpens the distribution (more deterministic),
    while a higher temperature flattens it (more random).
    """
    if temperature == 0.0 or temperature == 1.0:
        return logits
    return logits / temperature


def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Zero out logits outside the top-k highest values.

    Following vLLM's approach: sort by descending logit value, keep only the
    top_k entries, and mask the rest to -inf.
    """
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits

    # Remove tokens with logits below the top-k threshold
    top_k_values, _ = torch.topk(logits, top_k, dim=-1)
    min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
    logits = logits.masked_fill(logits < min_top_k_value, float("-inf"))
    return logits


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply nucleus (top-p) sampling.

    Following vLLM's approach:
    1. Sort logits in descending order.
    2. Compute cumulative probabilities from the sorted softmax.
    3. Mask tokens whose cumulative probability exceeds top_p.
    4. Scatter the mask back to the original logit positions.
    """
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Create mask: keep tokens where cumulative prob <= top_p
    # Shift right by 1 so the first token exceeding top_p is still included
    sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p

    # Set masked logits to -inf in sorted order
    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

    # Scatter back to original ordering
    logits = torch.zeros_like(logits).scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return logits


def sample(
    logits: torch.Tensor,
    sampling_params: Optional[SamplingParams] = None,
) -> torch.Tensor:
    """Sample the next token from logits using the given sampling parameters.

    Processing order (following vLLM convention):
    1. Temperature scaling
    2. Top-k filtering
    3. Top-p (nucleus) filtering
    4. Multinomial sampling (or argmax if greedy)

    Args:
        logits: Raw logits of shape (batch_size, vocab_size).
        sampling_params: Sampling configuration. If None, uses greedy decoding.

    Returns:
        Token ids of shape (batch_size, 1).
    """
    if sampling_params is None or sampling_params.is_greedy:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 1. Temperature scaling
    logits = _apply_temperature(logits, sampling_params.temperature)

    # 2. Top-k filtering
    if sampling_params.top_k > 0:
        logits = _apply_top_k(logits, sampling_params.top_k)

    # 3. Top-p (nucleus) filtering
    if sampling_params.top_p < 1.0:
        logits = _apply_top_p(logits, sampling_params.top_p)

    # 4. Convert to probabilities and sample
    probs = torch.softmax(logits, dim=-1)
    next_token_id = torch.multinomial(probs, num_samples=1)

    return next_token_id
