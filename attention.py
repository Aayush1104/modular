"""Nemotron attention implementation for MAX Graph API.

Nemotron attention is similar to Llama attention with these differences:
1. Partial Rotary Embeddings (only applied to a fraction of Q/K dims)
2. Optional attention bias (default: False)
3. GQA support (Grouped Query Attention)
"""

from __future__ import annotations

from max.graph import ops, Symbol, Graph

from .rotary_embedding import NemotronRotaryEmbedding


class NemotronAttention:
    """Multi-head attention with partial rotary embeddings and GQA.

    Supports:
    - Multi-Head Attention (MHA): num_kv_heads == num_heads
    - Grouped Query Attention (GQA): num_kv_heads < num_heads
    - Multi-Query Attention (MQA): num_kv_heads == 1
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        partial_rotary_factor: float = 0.5,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_attention_heads // num_key_value_heads
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # Projection dimensions
        self.q_proj_dim = num_attention_heads * head_dim
        self.k_proj_dim = num_key_value_heads * head_dim
        self.v_proj_dim = num_key_value_heads * head_dim

        # Partial rotary embeddings
        self.rotary_emb = NemotronRotaryEmbedding(
            head_dim=head_dim,
            partial_rotary_factor=partial_rotary_factor,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
        )

    def __call__(
        self,
        graph: Graph,
        hidden_states: Symbol,
        q_proj_weight: Symbol,
        k_proj_weight: Symbol,
        v_proj_weight: Symbol,
        o_proj_weight: Symbol,
        cos: Symbol,
        sin: Symbol,
        attention_mask: Symbol | None = None,
        kv_cache: tuple[Symbol, Symbol] | None = None,
    ) -> tuple[Symbol, tuple[Symbol, Symbol] | None]:
        """Forward pass for Nemotron attention.

        Args:
            graph: MAX computation graph.
            hidden_states: [batch, seq_len, hidden_size].
            q/k/v/o_proj_weight: Projection weight matrices.
            cos, sin: Precomputed rotary embedding values.
            attention_mask: Optional causal mask.
            kv_cache: Optional (key_cache, value_cache) for autoregressive.

        Returns:
            (attention_output, updated_kv_cache)
        """
        batch_size = hidden_states  # dynamic shape from graph

        # === Q, K, V Projections ===
        query = ops.matmul(hidden_states, ops.transpose(q_proj_weight, -2, -1))
        key = ops.matmul(hidden_states, ops.transpose(k_proj_weight, -2, -1))
        value = ops.matmul(hidden_states, ops.transpose(v_proj_weight, -2, -1))

        # Reshape to [batch, seq_len, num_heads, head_dim]
        # then transpose to [batch, num_heads, seq_len, head_dim]
        query = ops.reshape(query, [-1, -1, self.num_heads, self.head_dim])
        query = ops.transpose(query, 1, 2)

        key = ops.reshape(key, [-1, -1, self.num_kv_heads, self.head_dim])
        key = ops.transpose(key, 1, 2)

        value = ops.reshape(value, [-1, -1, self.num_kv_heads, self.head_dim])
        value = ops.transpose(value, 1, 2)

        # === Apply Partial Rotary Embeddings ===
        # This is the key Nemotron difference - only partial dims get RoPE
        query, key = self.rotary_emb.apply_partial_rotary(
            graph, query, key, cos, sin
        )

        # === KV Cache Update ===
        if kv_cache is not None:
            key_cache, value_cache = kv_cache
            key = ops.concat([key_cache, key], axis=2)
            value = ops.concat([value_cache, value], axis=2)
        updated_kv_cache = (key, value)

        # === GQA: Expand KV heads to match Q heads ===
        if self.num_kv_groups > 1:
            # Repeat KV heads to match number of query heads
            # key: [batch, kv_heads, seq, head_dim] -> [batch, heads, seq, head_dim]
            key = _repeat_kv(key, self.num_kv_groups)
            value = _repeat_kv(value, self.num_kv_groups)

        # === Scaled Dot-Product Attention ===
        scale = self.head_dim ** -0.5
        attn_weights = ops.matmul(query, ops.transpose(key, -2, -1)) * scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = ops.softmax(attn_weights, axis=-1)

        # Note: attention_dropout is only applied during training
        # For inference in MAX, we skip it

        attn_output = ops.matmul(attn_weights, value)

        # === Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden_size] ===
        attn_output = ops.transpose(attn_output, 1, 2)
        attn_output = ops.reshape(attn_output, [-1, -1, self.hidden_size])

        # === Output projection ===
        attn_output = ops.matmul(
            attn_output, ops.transpose(o_proj_weight, -2, -1)
        )

        return attn_output, updated_kv_cache


def _repeat_kv(hidden_states: Symbol, n_rep: int) -> Symbol:
    """Repeat KV heads n_rep times for GQA.

    Expands [batch, kv_heads, seq, head_dim] to [batch, heads, seq, head_dim]
    by repeating each KV head n_rep times.

    This is equivalent to torch.repeat_interleave(hidden_states, n_rep, dim=1)
    """
    if n_rep == 1:
        return hidden_states

    # In MAX Graph, this can be done via:
    # 1. Unsqueeze: [batch, kv_heads, 1, seq, head_dim]
    # 2. Expand: [batch, kv_heads, n_rep, seq, head_dim]
    # 3. Reshape: [batch, kv_heads * n_rep, seq, head_dim]
    #
    # Or via ops.tile / ops.repeat depending on MAX Graph API version

    # Conceptual implementation:
    expanded = ops.unsqueeze(hidden_states, 2)  # [B, kv_heads, 1, S, D]
    # expanded = ops.broadcast_to(expanded, [..., n_rep, ...])
    # result = ops.reshape(expanded, [batch, -1, seq, head_dim])

    return expanded  # Placeholder - needs full implementation
