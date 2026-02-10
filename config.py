"""Nemotron-4 configuration for MAX pipelines.

Adapts the HuggingFace NemotronConfig to MAX's pipeline configuration system.
Nemotron-4 is architecturally similar to Llama but with key differences:
- LayerNorm instead of RMSNorm
- Squared ReLU instead of SiLU/SwiGLU
- Partial Rotary Position Embeddings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NemotronConfig:
    """Configuration for the Nemotron-4 model.

    Maps to HuggingFace's NemotronConfig with defaults matching
    the Nemotron-8B architecture (nvidia/nemotron-3-8b-base-4k-hf).
    """

    # Model dimensions
    vocab_size: int = 256000
    hidden_size: int = 6144
    intermediate_size: int = 24576
    num_hidden_layers: int = 32
    num_attention_heads: int = 48
    num_key_value_heads: Optional[int] = None  # defaults to num_attention_heads
    head_dim: Optional[int] = None  # defaults to hidden_size // num_attention_heads

    # Normalization
    norm_eps: float = 1e-5

    # Activation
    hidden_act: str = "relu2"  # Squared ReLU

    # Position embeddings
    max_position_embeddings: int = 4096
    partial_rotary_factor: float = 0.5  # Key Nemotron difference
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None

    # Attention
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # Training config
    initializer_range: float = 0.0134
    use_cache: bool = True
    tie_word_embeddings: bool = False

    # Misc
    pad_token_id: Optional[int] = None
    bos_token_id: int = 2
    eos_token_id: int = 3

    model_type: str = "nemotron"

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @property
    def rotary_dim(self) -> int:
        """Number of dimensions that get rotary embeddings applied."""
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def num_kv_groups(self) -> int:
        """Number of query heads per KV head (for GQA)."""
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_huggingface(cls, hf_config) -> "NemotronConfig":
        """Create from a HuggingFace AutoConfig object."""
        rope_scaling = getattr(hf_config, "rope_scaling", None)
        rope_theta = hf_config.rope_theta if hasattr(hf_config, "rope_theta") else 10000.0

        # Handle rope_scaling dict which may contain rope_theta
        if rope_scaling and "rope_theta" in rope_scaling:
            rope_theta = rope_scaling["rope_theta"]

        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=getattr(
                hf_config, "num_key_value_heads", hf_config.num_attention_heads
            ),
            head_dim=getattr(hf_config, "head_dim", None),
            norm_eps=hf_config.norm_eps,
            hidden_act=getattr(hf_config, "hidden_act", "relu2"),
            max_position_embeddings=hf_config.max_position_embeddings,
            partial_rotary_factor=getattr(
                hf_config, "partial_rotary_factor", 0.5
            ),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=getattr(hf_config, "attention_bias", False),
            attention_dropout=getattr(hf_config, "attention_dropout", 0.0),
            initializer_range=getattr(hf_config, "initializer_range", 0.0134),
            use_cache=getattr(hf_config, "use_cache", True),
            tie_word_embeddings=getattr(
                hf_config, "tie_word_embeddings", False
            ),
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "norm_eps": self.norm_eps,
            "hidden_act": self.hidden_act,
            "max_position_embeddings": self.max_position_embeddings,
            "partial_rotary_factor": self.partial_rotary_factor,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "attention_bias": self.attention_bias,
            "attention_dropout": self.attention_dropout,
            "initializer_range": self.initializer_range,
            "use_cache": self.use_cache,
            "tie_word_embeddings": self.tie_word_embeddings,
        }
