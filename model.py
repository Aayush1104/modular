"""Nemotron-4 model for MAX Serve pipeline.

This module provides two integration paths:

1. **Quick path**: Extend Llama3Model (recommended for rapid integration)
   Since Nemotron-4 is architecturally very similar to Llama, you can
   extend the existing Llama3 implementation and override the differences.

2. **Full path**: Build from scratch using MAX Graph API
   For maximum control and optimization, build the full computation
   graph using NemotronGraphBuilder.

This file implements Path 1 (extending Llama3) as the primary approach,
with hooks for the full custom graph when needed.
"""

from __future__ import annotations

from typing import Optional

from max.driver import Device
from max.engine import InferenceSession
from max.graph.weights import Weights, WeightsAdapter
from max.nn import ReturnLogits
from max.pipelines.lib import KVCacheConfig, PipelineConfig, SupportedEncoding

from .config import NemotronConfig
from .weight_adapters import NemotronWeightAdapter


# =============================================================================
# Path 1: Extend Llama3Model (recommended for quick MAX integration)
# =============================================================================

try:
    from max.pipelines.architectures.llama3.model import Llama3Model

    class NemotronModel(Llama3Model):
        """Nemotron-4 model extending Llama3 for MAX Serve.

        Overrides the components where Nemotron differs from Llama:
        - Normalization: LayerNorm instead of RMSNorm
        - Activation: Squared ReLU instead of SwiGLU
        - Position Embeddings: Partial RoPE

        This approach leverages MAX's existing Llama3 infrastructure
        (KV cache, attention backends, SDPA, etc.) while customizing
        the Nemotron-specific components.
        """

        # Nemotron does NOT use attention bias by default
        attention_bias: bool = False

        def __init__(
            self,
            pipeline_config: PipelineConfig,
            session: InferenceSession,
            huggingface_config,
            encoding: SupportedEncoding,
            devices: list[Device],
            kv_cache_config: KVCacheConfig,
            weights: Weights,
            adapter: Optional[WeightsAdapter] = None,
            return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        ) -> None:
            # Parse Nemotron-specific config
            self.nemotron_config = NemotronConfig.from_huggingface(
                huggingface_config
            )

            # Use our custom weight adapter if none provided
            if adapter is None:
                adapter = NemotronWeightAdapter()

            super().__init__(
                pipeline_config=pipeline_config,
                session=session,
                huggingface_config=huggingface_config,
                encoding=encoding,
                devices=devices,
                kv_cache_config=kv_cache_config,
                weights=weights,
                adapter=adapter,
                return_logits=return_logits,
            )

        def _get_norm_type(self) -> str:
            """Override: Use LayerNorm instead of RMSNorm."""
            return "layer_norm"

        def _get_activation(self) -> str:
            """Override: Use Squared ReLU instead of SiLU."""
            return "relu2"

        def _get_partial_rotary_factor(self) -> float:
            """Override: Apply RoPE to only partial dimensions."""
            return self.nemotron_config.partial_rotary_factor

        def _get_mlp_has_gate(self) -> bool:
            """Override: Nemotron MLP has no gate_proj (not gated)."""
            return False

except ImportError:
    # Llama3Model not available - fall back to standalone implementation
    pass


# =============================================================================
# Path 2: Standalone model using full custom graph (for advanced use)
# =============================================================================

class NemotronStandaloneModel:
    """Standalone Nemotron model using full custom MAX Graph.

    Use this when you need complete control over the graph construction,
    or when the Llama3 base class doesn't expose the right override points.

    This builds the entire computation graph from scratch using
    NemotronGraphBuilder.
    """

    def __init__(
        self,
        config: NemotronConfig,
        weights: Weights,
        session: InferenceSession,
        devices: list[Device],
    ):
        from .arch import NemotronGraphBuilder

        self.config = config
        self.weights = weights
        self.session = session
        self.devices = devices

        # Build the computation graph
        self.graph_builder = NemotronGraphBuilder(config, weights)
        self.graph = self.graph_builder.build_graph("nemotron")

    def compile(self):
        """Compile the graph with MAX Engine."""
        from max.engine import compile_graph

        execution_device = self.devices[0] if self.devices else None
        self.compiled_model = compile_graph(
            self.graph,
            execution_device,
        )
        return self.compiled_model

    def generate_token(
        self,
        input_ids,
        position_ids,
        attention_mask,
    ):
        """Generate next token logits.

        Args:
            input_ids: Token IDs [batch, seq_len].
            position_ids: Position indices [batch, seq_len].
            attention_mask: Causal mask [batch, 1, seq_len, total_seq_len].

        Returns:
            Logits tensor [batch, seq_len, vocab_size].
        """
        if not hasattr(self, "compiled_model"):
            self.compile()

        outputs = self.compiled_model.execute(
            input_ids, position_ids, attention_mask
        )
        return outputs[0]  # logits
