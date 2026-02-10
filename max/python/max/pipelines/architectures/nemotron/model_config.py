# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://www.llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Standalone configuration for Nemotron (no MoE, LayerNorm1P, partial RoPE)."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData, WeightsFormat, weights_format
from max.nn.legacy.kv_cache import KVCacheParams
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from max.pipelines.lib.interfaces.arch_config import ArchConfigWithKVCache
from transformers import AutoConfig
from typing_extensions import Self, override


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

    def get_kv_params(self) -> KVCacheParams:
        return self.kv_params

    def get_max_seq_len(self) -> int:
        return self.max_position_embeddings

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return huggingface_config.num_hidden_layers

    @staticmethod
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
            head_dim=_get_head_dim(huggingface_config),
            page_size=kv_cache_config.kv_cache_page_size,
            cache_strategy=kv_cache_config.cache_strategy,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            enable_kvcache_swapping_to_host=kv_cache_config.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=kv_cache_config.host_kvcache_swap_space_gb,
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
        )

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        if pipeline_config.max_length:
            return pipeline_config.max_length
        return huggingface_config.max_position_embeddings

    @override
    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        huggingface_config = pipeline_config.model.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config required for '{pipeline_config.model.model_path}'"
            )
        kv_cache_config = pipeline_config.model.kv_cache
        quantization_encoding = pipeline_config.model.quantization_encoding
        if quantization_encoding is None:
            raise ValueError("quantization_encoding must not be None")
        dtype = quantization_encoding.dtype
        cache_dtype = pipeline_config.model.kv_cache.cache_dtype
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model.device_specs
        ]

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
            head_dim=_get_head_dim(huggingface_config),
            max_position_embeddings=huggingface_config.max_position_embeddings,
            norm_eps=getattr(huggingface_config, "norm_eps", 1e-5),
            rope_theta=getattr(huggingface_config, "rope_theta", 10000.0),
            attention_bias=getattr(huggingface_config, "attention_bias", False),
            mlp_bias=getattr(huggingface_config, "mlp_bias", True),
            partial_rotary_factor=_get_partial_rotary_factor(huggingface_config),
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
        self.tie_word_embeddings = tie_word_embeddings
        self.return_logits = return_logits
