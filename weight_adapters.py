"""Weight adapters for Nemotron model.

Handles loading and transforming weights from HuggingFace SafeTensors
format into the format expected by the MAX Graph.

Nemotron weight names follow the HuggingFace convention:
    model.embed_tokens.weight
    model.layers.{i}.input_layernorm.weight
    model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    model.layers.{i}.post_attention_layernorm.weight
    model.layers.{i}.mlp.up_proj.weight
    model.layers.{i}.mlp.down_proj.weight
    model.norm.weight
    lm_head.weight

Note: Nemotron does NOT have:
    - model.layers.{i}.mlp.gate_proj.weight (no gated activation)
    - Any bias terms (bias=False for all projections and norms)
"""

from __future__ import annotations

from typing import Optional
from pathlib import Path

from max.graph.weights import Weights, WeightsAdapter


class NemotronWeightAdapter(WeightsAdapter):
    """Adapts HuggingFace Nemotron weights for MAX.

    The Nemotron architecture uses the same weight naming convention
    as the HuggingFace transformers implementation, so minimal
    transformation is needed for SafeTensors format.

    Key differences from Llama weight loading:
    - No gate_proj weights (2 MLP projections instead of 3)
    - LayerNorm weights instead of RMSNorm
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_weight_mapping(num_layers: int) -> dict[str, str]:
        """Get the mapping from MAX weight names to HuggingFace names.

        For Nemotron, these are 1:1 since we use HF naming in our graph.

        Returns:
            Dictionary mapping MAX weight keys to HF weight keys.
        """
        mapping = {
            "model.embed_tokens.weight": "model.embed_tokens.weight",
            "model.norm.weight": "model.norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

        for i in range(num_layers):
            prefix = f"model.layers.{i}"
            layer_weights = [
                "input_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "post_attention_layernorm.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ]
            for w in layer_weights:
                key = f"{prefix}.{w}"
                mapping[key] = key

        return mapping

    @staticmethod
    def expected_weight_shapes(config) -> dict[str, tuple]:
        """Get expected shapes for validation.

        Useful for verifying loaded weights match the model config.
        """
        shapes = {
            "model.embed_tokens.weight": (config.vocab_size, config.hidden_size),
            "model.norm.weight": (config.hidden_size,),
            "lm_head.weight": (config.vocab_size, config.hidden_size),
        }

        q_dim = config.num_attention_heads * config.head_dim
        kv_dim = config.num_key_value_heads * config.head_dim

        for i in range(config.num_hidden_layers):
            prefix = f"model.layers.{i}"
            shapes.update({
                f"{prefix}.input_layernorm.weight": (config.hidden_size,),
                f"{prefix}.self_attn.q_proj.weight": (q_dim, config.hidden_size),
                f"{prefix}.self_attn.k_proj.weight": (kv_dim, config.hidden_size),
                f"{prefix}.self_attn.v_proj.weight": (kv_dim, config.hidden_size),
                f"{prefix}.self_attn.o_proj.weight": (config.hidden_size, q_dim),
                f"{prefix}.post_attention_layernorm.weight": (config.hidden_size,),
                f"{prefix}.mlp.up_proj.weight": (
                    config.intermediate_size,
                    config.hidden_size,
                ),
                f"{prefix}.mlp.down_proj.weight": (
                    config.hidden_size,
                    config.intermediate_size,
                ),
            })

        return shapes

    @staticmethod
    def list_all_weight_names(num_layers: int) -> list[str]:
        """List all weight tensor names for the model.

        Useful for debugging and verifying checkpoint completeness.
        """
        names = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]

        for i in range(num_layers):
            prefix = f"model.layers.{i}"
            names.extend([
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            ])

        return names
