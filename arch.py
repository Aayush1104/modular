"""Nemotron-4 model graph construction for MAX Engine.

Builds the full computation graph for Nemotron inference using the
MAX Graph API. The graph is compiled and optimized by MAX Engine's
advanced graph compiler for efficient execution on GPU/CPU.

The graph follows this structure:
    Input IDs -> Embedding -> N x DecoderLayer -> LayerNorm -> LM Head -> Logits

Each DecoderLayer:
    x -> LayerNorm -> Attention (+ partial RoPE) -> + residual
      -> LayerNorm -> MLP (squared ReLU) -> + residual
"""

from __future__ import annotations

from typing import Optional

from max.graph import ops, Symbol, Graph, TensorType, Type
from max.graph.weights import Weights

from .config import NemotronConfig
from .layers import (
    NemotronAttention,
    NemotronMLP,
    NemotronLayerNorm,
    NemotronRotaryEmbedding,
    build_layernorm,
    squared_relu,
)


class NemotronGraphBuilder:
    """Builds the MAX Graph for Nemotron-4 inference.

    This constructs the full computation graph that MAX Engine compiles
    into an optimized inference pipeline.
    """

    def __init__(self, config: NemotronConfig, weights: Weights):
        self.config = config
        self.weights = weights

    def build_graph(self, name: str = "nemotron") -> Graph:
        """Build the complete Nemotron inference graph.

        Args:
            name: Name for the graph.

        Returns:
            A MAX Graph ready for compilation.
        """
        config = self.config

        # Define graph inputs
        # input_ids: [batch_size, seq_len]
        # position_ids: [batch_size, seq_len]
        # attention_mask: [batch_size, 1, seq_len, total_seq_len]
        input_ids_type = TensorType(shape=[-1, -1], dtype="int64")
        position_ids_type = TensorType(shape=[-1, -1], dtype="int64")
        attention_mask_type = TensorType(
            shape=[-1, 1, -1, -1], dtype="float32"
        )

        graph = Graph(
            name=name,
            input_types=[input_ids_type, position_ids_type, attention_mask_type],
        )

        input_ids, position_ids, attention_mask = graph.inputs

        # === Token Embedding ===
        embed_tokens = self.weights["model.embed_tokens.weight"]
        hidden_states = ops.gather(embed_tokens, input_ids, axis=0)

        # === Precompute RoPE cos/sin ===
        rope = NemotronRotaryEmbedding(
            head_dim=config.head_dim,
            partial_rotary_factor=config.partial_rotary_factor,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
        )
        cos, sin = rope.compute_cos_sin(graph, position_ids)

        # === Decoder Layers ===
        for layer_idx in range(config.num_hidden_layers):
            hidden_states = self._build_decoder_layer(
                graph=graph,
                hidden_states=hidden_states,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
                layer_idx=layer_idx,
            )

        # === Final LayerNorm ===
        final_norm_weight = self.weights["model.norm.weight"]
        hidden_states = build_layernorm(
            graph, hidden_states, final_norm_weight, eps=config.norm_eps
        )

        # === LM Head ===
        lm_head_weight = self.weights["lm_head.weight"]
        logits = ops.matmul(
            hidden_states, ops.transpose(lm_head_weight, -2, -1)
        )

        graph.output(logits)
        return graph

    def _build_decoder_layer(
        self,
        graph: Graph,
        hidden_states: Symbol,
        cos: Symbol,
        sin: Symbol,
        attention_mask: Symbol,
        layer_idx: int,
    ) -> Symbol:
        """Build a single Nemotron decoder layer.

        Structure:
            residual = x
            x = LayerNorm(x)  # input_layernorm
            x = Attention(x)  # self_attn with partial RoPE
            x = x + residual
            residual = x
            x = LayerNorm(x)  # post_attention_layernorm
            x = MLP(x)        # mlp with squared ReLU
            x = x + residual
        """
        config = self.config
        prefix = f"model.layers.{layer_idx}"

        # === Self-Attention Block ===
        residual = hidden_states

        # Input LayerNorm (Nemotron uses LayerNorm, not RMSNorm)
        input_norm_weight = self.weights[f"{prefix}.input_layernorm.weight"]
        hidden_states = build_layernorm(
            graph, hidden_states, input_norm_weight, eps=config.norm_eps
        )

        # Attention
        attn = NemotronAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            partial_rotary_factor=config.partial_rotary_factor,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            attention_bias=config.attention_bias,
        )

        q_proj = self.weights[f"{prefix}.self_attn.q_proj.weight"]
        k_proj = self.weights[f"{prefix}.self_attn.k_proj.weight"]
        v_proj = self.weights[f"{prefix}.self_attn.v_proj.weight"]
        o_proj = self.weights[f"{prefix}.self_attn.o_proj.weight"]

        hidden_states, _ = attn(
            graph=graph,
            hidden_states=hidden_states,
            q_proj_weight=q_proj,
            k_proj_weight=k_proj,
            v_proj_weight=v_proj,
            o_proj_weight=o_proj,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
        )

        hidden_states = hidden_states + residual

        # === MLP Block ===
        residual = hidden_states

        # Post-attention LayerNorm
        post_attn_norm_weight = self.weights[
            f"{prefix}.post_attention_layernorm.weight"
        ]
        hidden_states = build_layernorm(
            graph, hidden_states, post_attn_norm_weight, eps=config.norm_eps
        )

        # MLP with Squared ReLU
        mlp = NemotronMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )

        up_proj = self.weights[f"{prefix}.mlp.up_proj.weight"]
        down_proj = self.weights[f"{prefix}.mlp.down_proj.weight"]

        hidden_states = mlp(
            graph=graph,
            hidden_states=hidden_states,
            up_proj_weight=up_proj,
            down_proj_weight=down_proj,
        )

        hidden_states = hidden_states + residual

        return hidden_states
