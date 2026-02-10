"""Nemotron MLP implementation for MAX Graph API.

Nemotron uses a Squared ReLU (ReLU²) activation instead of SiLU/SwiGLU.
This is one of the key architectural differences from Llama.

Architecture:
    up_proj:   hidden_size -> intermediate_size
    down_proj:  intermediate_size -> hidden_size
    Activation: relu(x)² (Squared ReLU)

Note: Unlike Llama/Mistral which use a gated architecture with gate_proj,
Nemotron uses a simpler two-projection MLP with squared ReLU.
"""

from __future__ import annotations

from max.graph import ops, Symbol, Graph


class NemotronMLP:
    """Nemotron MLP block with Squared ReLU activation.

    The Nemotron MLP differs from Llama's SwiGLU MLP:

    Llama MLP:
        gate = silu(gate_proj(x))
        up = up_proj(x)
        output = down_proj(gate * up)

    Nemotron MLP:
        up = relu(up_proj(x))²
        output = down_proj(up)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

    def __call__(
        self,
        graph: Graph,
        hidden_states: Symbol,
        up_proj_weight: Symbol,
        down_proj_weight: Symbol,
    ) -> Symbol:
        """Apply Nemotron MLP.

        Args:
            graph: The MAX computation graph.
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size].
            up_proj_weight: Weight for up projection [intermediate_size, hidden_size].
            down_proj_weight: Weight for down projection [hidden_size, intermediate_size].

        Returns:
            Output tensor of shape [batch, seq_len, hidden_size].
        """
        # Up projection: hidden_size -> intermediate_size
        up = ops.matmul(hidden_states, ops.transpose(up_proj_weight, -2, -1))

        # Squared ReLU activation: relu(x)²
        activated = squared_relu(up)

        # Down projection: intermediate_size -> hidden_size
        output = ops.matmul(activated, ops.transpose(down_proj_weight, -2, -1))

        return output


def squared_relu(x: Symbol) -> Symbol:
    """Squared ReLU activation: relu(x)²

    This is the activation function used in Nemotron-4.
    It provides stronger sparsity than standard ReLU while
    maintaining good gradient flow properties.

    Note: This can be efficiently fused into a single custom kernel
    on GPU. See mojo_ops/squared_relu.mojo for a Mojo implementation.
    """
    relu_x = ops.relu(x)
    return relu_x * relu_x
