# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
<<<<<<< HEAD
# https://www.llvm.org/LICENSE.txt
=======
# https://llvm.org/LICENSE.txt
>>>>>>> upstream/main
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

<<<<<<< HEAD
"""Standalone configuration for Nemotron (no MoE, LayerNorm1P, partial RoPE)."""

=======
>>>>>>> upstream/main
from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef
<<<<<<< HEAD
from max.graph.weights import WeightData, WeightsFormat, weights_format
=======
from max.graph.weights import WeightData
>>>>>>> upstream/main
from max.nn.legacy.kv_cache import KVCacheParams
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from max.pipelines.lib.interfaces.arch_config import ArchConfigWithKVCache
from transformers import AutoConfig
from typing_extensions import Self, override


<<<<<<< HEAD
def _get_partial_rotary_factor(huggingface_config: AutoConfig) -> float:
    rope_params = getattr(huggingface_config, "rope_parameters", None)
    if isinstance(rope_params, dict):
        return float(rope_params.get("partial_rotary_factor", 0.5))
    return float(getattr(huggingface_config, "partial_rotary_factor", 0.5))


def _get_head_dim(huggingface_config: AutoConfig) -> int:
    if hasattr(huggingface_config, "kv_channels") and getattr(
        huggingface_config, "kv_channels", None
    ):
        return huggingface_config.kv_channels
    if hasattr(huggingface_config, "head_dim"):
        return huggingface_config.head_dim
    return (
        huggingface_config.hidden_size
        // huggingface_config.num_attention_heads
    )


@dataclass(kw_only=True)
class NemotronConfig(ArchConfigWithKVCache):
    """Configuration for Nemotron (dense decoder, LayerNorm1P, ReLU² MLP, partial RoPE)."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    norm_eps: float
    rope_theta: float
    attention_bias: bool
    mlp_bias: bool
    partial_rotary_factor: float
    dtype: DType
    devices: list[DeviceRef]
    kv_params: KVCacheParams
    tie_word_embeddings: bool = False
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
=======
@dataclass(kw_only=True)
class NemotronConfig(ArchConfigWithKVCache):
    """Configuration for Nemotron models."""

    vocab_size: int
    """Vocabulary size of the Nemotron model."""

    hidden_size: int
    """Dimension of the hidden representations."""

    intermediate_size: int
    """Dimension of the MLP representations."""

    num_hidden_layers: int
    """Number of hidden layers in the Transformer decoder."""

    num_attention_heads: int
    """Number of attention heads for each attention layer."""

    num_key_value_heads: int
    """Number of key_value heads for Grouped Query Attention."""

    head_dim: int
    """The attention head dimension."""

    max_position_embeddings: int
    """The maximum sequence length that this model might ever be used with."""

    norm_eps: float
    """The epsilon used by the LayerNorm1P normalization layers."""

    rope_theta: float
    """The base period of the RoPE embeddings."""

    partial_rotary_factor: float
    """Fraction of head_dim to apply rotary embeddings to."""

    attention_bias: bool
    """Whether to use a bias in attention projection layers."""

    mlp_bias: bool
    """Whether to use a bias in MLP projection layers."""

    # MAX-specific config parameters.
    dtype: DType
    """DType of the model weights and input."""

    devices: list[DeviceRef]
    """Devices to run the model with."""

    kv_params: KVCacheParams
    """KV cache parameters."""

    tie_word_embeddings: bool = False
    """Whether to tie weight embeddings."""

    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    """Whether to return the last token, all logits, or a variable number of logits."""
>>>>>>> upstream/main

    def get_kv_params(self) -> KVCacheParams:
        return self.kv_params

    def get_max_seq_len(self) -> int:
        return self.max_position_embeddings

    @staticmethod
<<<<<<< HEAD
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return huggingface_config.num_hidden_layers

    @staticmethod
=======
>>>>>>> upstream/main
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return KVCacheParams(
            dtype=cache_dtype,
            num_layers=NemotronConfig.get_num_layers(huggingface_config),
            n_kv_heads=huggingface_config.num_key_value_heads,
<<<<<<< HEAD
            head_dim=_get_head_dim(huggingface_config),
=======
            head_dim=huggingface_config.head_dim,
>>>>>>> upstream/main
            page_size=kv_cache_config.kv_cache_page_size,
            cache_strategy=kv_cache_config.cache_strategy,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            enable_kvcache_swapping_to_host=kv_cache_config.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=kv_cache_config.host_kvcache_swap_space_gb,
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
        )

    @staticmethod
<<<<<<< HEAD
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        if pipeline_config.max_length:
            return pipeline_config.max_length
=======
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return huggingface_config.num_hidden_layers

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        max_seq_len = pipeline_config.max_length
        if max_seq_len:
            return max_seq_len
>>>>>>> upstream/main
        return huggingface_config.max_position_embeddings

    @override
    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        huggingface_config = pipeline_config.model.huggingface_config
        if huggingface_config is None:
            raise ValueError(
<<<<<<< HEAD
                f"HuggingFace config required for '{pipeline_config.model.model_path}'"
=======
                f"HuggingFace config is required for '{pipeline_config.model.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
>>>>>>> upstream/main
            )
        kv_cache_config = pipeline_config.model.kv_cache
        quantization_encoding = pipeline_config.model.quantization_encoding
        if quantization_encoding is None:
            raise ValueError("quantization_encoding must not be None")
        dtype = quantization_encoding.dtype
        cache_dtype = pipeline_config.model.kv_cache.cache_dtype
<<<<<<< HEAD
=======

>>>>>>> upstream/main
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model.device_specs
        ]

<<<<<<< HEAD
=======
        # Nemotron uses norm_eps (not rms_norm_eps)
        norm_eps = getattr(huggingface_config, "norm_eps", 1e-5)

        # partial_rotary_factor can be at top level or inside rope_parameters
        partial_rotary_factor = getattr(
            huggingface_config, "partial_rotary_factor", None
        )
        if partial_rotary_factor is None:
            rope_params = getattr(huggingface_config, "rope_parameters", None)
            if rope_params and isinstance(rope_params, dict):
                partial_rotary_factor = rope_params.get(
                    "partial_rotary_factor", 0.5
                )
            else:
                partial_rotary_factor = 0.5

        # MLP bias
        mlp_bias = getattr(huggingface_config, "mlp_bias", False)

        # Attention bias
        attention_bias = getattr(huggingface_config, "attention_bias", False)

        # RoPE theta (default 10000)
        rope_theta = getattr(huggingface_config, "rope_theta", 10000.0)

>>>>>>> upstream/main
        kv_params = cls.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

        return cls(
            vocab_size=huggingface_config.vocab_size,
            hidden_size=huggingface_config.hidden_size,
            intermediate_size=huggingface_config.intermediate_size,
            num_hidden_layers=huggingface_config.num_hidden_layers,
            num_attention_heads=huggingface_config.num_attention_heads,
            num_key_value_heads=huggingface_config.num_key_value_heads,
<<<<<<< HEAD
            head_dim=_get_head_dim(huggingface_config),
            max_position_embeddings=huggingface_config.max_position_embeddings,
            norm_eps=getattr(huggingface_config, "norm_eps", 1e-5),
            rope_theta=getattr(huggingface_config, "rope_theta", 10000.0),
            attention_bias=getattr(huggingface_config, "attention_bias", False),
            mlp_bias=getattr(huggingface_config, "mlp_bias", True),
            partial_rotary_factor=_get_partial_rotary_factor(huggingface_config),
=======
            head_dim=huggingface_config.head_dim,
            max_position_embeddings=huggingface_config.max_position_embeddings,
            norm_eps=norm_eps,
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            attention_bias=attention_bias,
            mlp_bias=mlp_bias,
>>>>>>> upstream/main
            dtype=dtype,
            devices=device_refs,
            kv_params=kv_params,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
    ) -> None:
        tie_word_embeddings = (
            getattr(huggingface_config, "tie_word_embeddings", False)
            or "language_model.lm_head.weight" not in state_dict
        )
<<<<<<< HEAD
=======

>>>>>>> upstream/main
        self.tie_word_embeddings = tie_word_embeddings
        self.return_logits = return_logits
