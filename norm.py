"""Nemotron LayerNorm implementation for MAX Graph API.

Nemotron uses standard LayerNorm (without bias) instead of RMSNorm.
This is one of the key architectural differences from Llama.

In MAX Graph API, we build this as a graph operation that can be
compiled and optimized by the MAX Engine compiler.
"""

from __future__ import annotations

from max.graph import ops, Symbol, TensorType, Graph
from max.graph.weights import Weights


class NemotronLayerNorm:
    """LayerNorm without bias, as used in Nemotron-4.

    Unlike RMSNorm (used in Llama), LayerNorm computes:
        y = (x - mean(x)) / sqrt(var(x) + eps) * weight

    The Nemotron variant has elementwise_affine=True but bias=False.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        weight_name: str = "weight",
    ):
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight_name = weight_name

    def __call__(
        self,
        graph: Graph,
        hidden_states: Symbol,
        weight: Symbol,
    ) -> Symbol:
        """Apply LayerNorm to hidden_states.

        Args:
            graph: The MAX computation graph.
            hidden_states: Input tensor of shape [..., hidden_size].
            weight: Learnable scale parameter of shape [hidden_size].

        Returns:
            Normalized tensor of the same shape as hidden_states.
        """
        # Compute mean along last dimension
        mean = ops.mean(hidden_states, axis=-1, keepdims=True)

        # Center the input
        centered = hidden_states - mean

        # Compute variance
        variance = ops.mean(centered * centered, axis=-1, keepdims=True)

        # Normalize
        inv_std = ops.rsqrt(variance + self.eps)
        normalized = centered * inv_std

        # Scale (no bias in Nemotron)
        return normalized * weight


def build_layernorm(
    graph: Graph,
    hidden_states: Symbol,
    weight: Symbol,
    eps: float = 1e-5,
) -> Symbol:
    """Functional interface for Nemotron LayerNorm.

    Convenience function that applies LayerNorm without
    instantiating the class.
    """
    mean = ops.mean(hidden_states, axis=-1, keepdims=True)
    centered = hidden_states - mean
    variance = ops.mean(centered * centered, axis=-1, keepdims=True)
    inv_std = ops.rsqrt(variance + eps)
    normalized = centered * inv_std
    return normalized * weight
