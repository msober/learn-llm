


import torch
import torch.nn as nn


class Qwen3Model(nn.Module):
    """Qwen3 language model with RoPE positional encoding and grouped-query attention."""

    def __init__(self, config):
        super().__init__()

        self.token_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"], dtype=config["dtype"])

        # Use ModuleList instead of Sequential because each block requires multiple inputs (hidden_states, mask, cos, sin)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config["n_layers"])]
        )

        self.final_norm = RMSNorm(config["emb_dim"])
        self.output_projection = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False, dtype=config["dtype"])

        head_dim = config["head_dim"] if config["head_dim"] is not None else config["emb_dim"] // config["n_heads"]
        rope_cos, rope_sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=config["rope_base"],
            context_length=config["context_length"]
        )
        self.register_buffer("cos", rope_cos, persistent=False)
        self.register_buffer("sin", rope_sin, persistent=False)
        self.config = config

    def forward(self, input_token_ids, start_position=0):
        """
        Args:
            input_token_ids: Token indices of shape (batch_size, sequence_length).
            start_position: Position offset for KV cache.
                            0 = prefill (no cache or first pass); >0 = decode with cache.
        Returns:
            Logits of shape (batch_size, sequence_length, vocab_size).
        """
        hidden_states = self.token_embedding(input_token_ids)

        query_length = hidden_states.shape[1]
        total_length = start_position + query_length

        # Build causal mask of shape (query_length, total_length):
        # each query position can attend to all KV positions up to and including itself.
        causal_mask = torch.triu(
            torch.ones(query_length, total_length, device=hidden_states.device, dtype=torch.bool),
            diagonal=start_position + 1
        )

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, causal_mask, self.cos, self.sin, start_position=start_position)

        hidden_states = self.final_norm(hidden_states)
        logits = self.output_projection(hidden_states.to(self.output_projection.weight.dtype))
        return logits

    def clear_kv_cache(self):
        """Clear KV cache in all transformer blocks (call before each new generation)."""
        for block in self.transformer_blocks:
            block.attention.clear_cache()

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with grouped-query attention and SwiGLU feed-forward."""

    def __init__(self, config):
        super().__init__()
        self.attention = GroupedQueryAttention(
            embedding_dim=config["emb_dim"],
            num_heads=config["n_heads"],
            head_dim=config["head_dim"],
            num_kv_groups=config["n_kv_groups"],
            qk_norm=config["qk_norm"],
            dtype=config["dtype"]
        )
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config["emb_dim"], eps=1e-6)
        self.feed_forward_norm = RMSNorm(config["emb_dim"], eps=1e-6)

    def forward(self, hidden_states, mask, cos, sin, start_position=0):
        # Pre-norm attention with residual connection
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        hidden_states = self.attention(hidden_states, mask, cos, sin, start_position=start_position)
        hidden_states = hidden_states + residual

        # Pre-norm feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.feed_forward_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states

class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) that shares key/value heads across query head groups,
    reducing KV-cache memory while preserving model quality.

    Supports an optional KV cache for efficient autoregressive generation:
    - Prefill stage (start_position=0): processes the full prompt and populates the cache.
    - Decode stage (start_position>0): processes only the new token(s) and appends to the cache.
    """

    def __init__(
        self, embedding_dim, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.heads_per_kv_group = num_heads // num_kv_groups

        if head_dim is None:
            assert embedding_dim % num_heads == 0, "`embedding_dim` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = embedding_dim // num_heads

        self.head_dim = head_dim
        self.total_head_dim = num_heads * head_dim

        self.query_projection = nn.Linear(embedding_dim, self.total_head_dim, bias=False, dtype=dtype)
        self.key_projection = nn.Linear(embedding_dim, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.value_projection = nn.Linear(embedding_dim, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.output_projection = nn.Linear(self.total_head_dim, embedding_dim, bias=False, dtype=dtype)

        if qk_norm:
            self.query_norm = RMSNorm(head_dim, eps=1e-6)
            self.key_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.query_norm = self.key_norm = None

        # KV cache buffers (populated during generation)
        self.cached_keys = None
        self.cached_values = None

    def clear_cache(self):
        """Reset the KV cache (call before each new generation sequence)."""
        self.cached_keys = None
        self.cached_values = None

    def forward(self, hidden_states, mask, cos, sin, start_position=0):
        """
        Args:
            hidden_states: (batch_size, sequence_length, embedding_dim)
            mask: Boolean causal mask.
            cos, sin: RoPE tables.
            start_position: Position offset for RoPE when using KV cache.
                            0 means no cache / prefill; >0 means decode with cache.
        """
        batch_size, sequence_length, _ = hidden_states.shape

        queries = self.query_projection(hidden_states)
        keys = self.key_projection(hidden_states)
        values = self.value_projection(hidden_states)

        # Reshape to (batch_size, num_heads_or_kv_groups, sequence_length, head_dim)
        queries = queries.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, sequence_length, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, sequence_length, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.query_norm:
            queries = self.query_norm(queries)
        if self.key_norm:
            keys = self.key_norm(keys)

        # Apply RoPE with offset for KV cache
        queries = apply_rope(queries, cos, sin, offset=start_position)
        keys = apply_rope(keys, cos, sin, offset=start_position)

        # KV cache: concatenate new keys/values with cached ones
        if self.cached_keys is not None:
            keys = torch.cat([self.cached_keys, keys], dim=2)
            values = torch.cat([self.cached_values, values], dim=2)
        self.cached_keys = keys
        self.cached_values = values

        # Repeat KV heads to match the number of query heads for GQA (zero-copy via expand)
        keys = repeat_kv(keys, self.heads_per_kv_group)
        values = repeat_kv(values, self.heads_per_kv_group)

        attention_scores = queries @ keys.transpose(2, 3)
        attention_scores = attention_scores.masked_fill(mask, -torch.inf)
        attention_weights = torch.softmax(attention_scores / self.head_dim**0.5, dim=-1)

        context_vectors = (attention_weights @ values).transpose(1, 2).reshape(batch_size, sequence_length, self.total_head_dim)
        return self.output_projection(context_vectors)


def repeat_kv(hidden_states, num_repeats):
    """
    Expand KV heads to match the number of query heads for GQA, without copying memory.

    Uses torch.expand (stride=0 broadcast) instead of repeat_interleave to avoid
    allocating a full-size copy of the KV tensor.

    Args:
        hidden_states: Tensor of shape (batch_size, num_kv_groups, sequence_length, head_dim).
        num_repeats: Number of times each KV head should be repeated (num_heads // num_kv_groups).

    Returns:
        Tensor of shape (batch_size, num_kv_groups * num_repeats, sequence_length, head_dim).
    """
    if num_repeats == 1:
        return hidden_states

    batch_size, num_kv_groups, sequence_length, head_dim = hidden_states.shape
    # Insert a new dim and broadcast without copying: (batch, kv_groups, 1, seq, head_dim)
    #   → expand to (batch, kv_groups, num_repeats, seq, head_dim)
    #   → reshape to (batch, kv_groups * num_repeats, seq, head_dim)
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size, num_kv_groups, num_repeats, sequence_length, head_dim
    ).reshape(batch_size, num_kv_groups * num_repeats, sequence_length, head_dim)
    return hidden_states


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    """Precompute cosine and sine tables for Rotary Position Embedding (RoPE)."""
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    inverse_frequencies = 1.0 / (
        theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim)
    )

    position_indices = torch.arange(context_length, dtype=dtype)

    # Outer product: each position × each frequency → (context_length, head_dim // 2)
    angles = position_indices.unsqueeze(1) * inverse_frequencies.unsqueeze(0)

    # Duplicate to cover full head_dim → (context_length, head_dim)
    angles = torch.cat([angles, angles], dim=1)

    return torch.cos(angles), torch.sin(angles)


def apply_rope(hidden_states, cos, sin, offset=0):
    """
    Apply rotary position embedding to hidden states.

    Args:
        hidden_states: Tensor of shape (batch_size, num_heads, sequence_length, head_dim).
        cos: Precomputed cosine table of shape (max_sequence_length, head_dim).
        sin: Precomputed sine table of shape (max_sequence_length, head_dim).
        offset: Position offset for KV cache (0 during prefill, >0 during decode).
    """
    batch_size, num_heads, sequence_length, head_dim = hidden_states.shape
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    first_half = hidden_states[..., : head_dim // 2]
    second_half = hidden_states[..., head_dim // 2:]

    # Slice cos/sin with offset and broadcast to (1, 1, sequence_length, head_dim)
    cos = cos[offset : offset + sequence_length, :].unsqueeze(0).unsqueeze(0)
    sin = sin[offset : offset + sequence_length, :].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat((-second_half, first_half), dim=-1)
    rotated_hidden_states = (hidden_states * cos) + (rotated * sin)

    return rotated_hidden_states.to(dtype=hidden_states.dtype)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (without mean-centering)."""

    def __init__(self, embedding_dim, eps=1e-6, bias=False, upcast_to_float32=True):
        super().__init__()
        self.eps = eps
        self.upcast_to_float32 = upcast_to_float32
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim)) if bias else None

    def forward(self, hidden_states):
        original_dtype = hidden_states.dtype

        # Upcast to float32 for numerical stability during normalization
        if self.upcast_to_float32:
            hidden_states = hidden_states.to(torch.float32)

        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps)
        normalized = normalized * self.scale

        if self.shift is not None:
            normalized = normalized + self.shift

        return normalized.to(original_dtype)

    
class FeedForward(nn.Module):
    """SwiGLU feed-forward network: uses gated linear units with SiLU activation."""

    def __init__(self, config):
        super().__init__()
        self.gate_projection = nn.Linear(config["emb_dim"], config["hidden_dim"], dtype=config["dtype"], bias=False)
        self.up_projection = nn.Linear(config["emb_dim"], config["hidden_dim"], dtype=config["dtype"], bias=False)
        self.down_projection = nn.Linear(config["hidden_dim"], config["emb_dim"], dtype=config["dtype"], bias=False)

    def forward(self, hidden_states):
        gated = nn.functional.silu(self.gate_projection(hidden_states))
        up = self.up_projection(hidden_states)
        return self.down_projection(gated * up)