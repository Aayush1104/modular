# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Implements the Nemotron model."""

from __future__ import annotations

import functools
from collections.abc import Sequence

from max import functional as F
from max.dtype import DType
from max.graph import BufferValue, TensorValue
from max.kv_cache import PagedKVCacheManager
from max.nn import Module
from max.nn.embedding import Embedding
from max.nn.legacy.kv_cache import PagedCacheValues
from max.nn.linear import Linear
from max.nn.sequential import ModuleList
from max.tensor import Tensor

from .layers.attention import NemotronAttention, PartialRotaryEmbedding
from .layers.layer_norm_1p import NemotronLayerNorm1P
from .layers.mlp import NemotronMLP
from .layers.transformer_block import NemotronTransformerBlock
from .model_config import NemotronConfig


class NemotronTextModel(
    Module[[Tensor, PagedCacheValues, Tensor, Tensor], tuple[Tensor, ...]]
):
    """The Nemotron language model.

    Decoder-only Transformer with dense MLP (ReLU-squared),
    LayerNorm1P normalization, and partial rotary embeddings.
    """

    def __init__(self, config: NemotronConfig) -> None:
        super().__init__()
        self.devices = config.devices

        # Partial RoPE
        rope = PartialRotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            device=config.devices[0].to_device(),
            head_dim=config.head_dim,
            interleaved=False,
            partial_rotary_factor=config.partial_rotary_factor,
        )

        self.embed_tokens = Embedding(
            config.vocab_size,
            dim=config.hidden_size,
        )

        self.norm = NemotronLayerNorm1P(
            config.hidden_size,
            eps=config.norm_eps,
        )

        self.lm_head = Linear(
            in_dim=config.hidden_size,
            out_dim=config.vocab_size,
            bias=False,
        )

        create_norm = functools.partial(
            NemotronLayerNorm1P,
            config.hidden_size,
            eps=config.norm_eps,
        )

        layers = []
        for i in range(config.num_hidden_layers):
            layers.append(
                NemotronTransformerBlock(
                    attention=NemotronAttention(
                        rope=rope,
                        num_attention_heads=config.num_attention_heads,
                        num_key_value_heads=config.num_key_value_heads,
                        hidden_size=config.hidden_size,
                        kv_params=config.kv_params,
                        layer_idx=i,
                        has_bias=config.attention_bias,
                    ),
                    mlp=NemotronMLP(
                        hidden_size=config.hidden_size,
                        intermediate_size=config.intermediate_size,
                        bias=config.mlp_bias,
                    ),
                    input_layernorm=create_norm(),
                    post_attention_layernorm=create_norm(),
                )
            )

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.layers = ModuleList(layers)
        self.kv_params = config.kv_params
        self.return_logits = config.return_logits

    def forward(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
    ) -> tuple[Tensor, ...]:
        h = self.embed_tokens(tokens)
        # Run through transformer layers
        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=h.device)
            h = layer(
                layer_idx_tensor,
                h,
                kv_collection,
                input_row_offsets=input_row_offsets,
            )

        # Get last token logits only
        last_token_indices = input_row_offsets[1:] - 1
        last_token_h = F.gather(h, last_token_indices, axis=0)
        last_logits = F.cast(
            self.lm_head(self.norm(last_token_h)),
            DType.float32,
        )

        return (last_logits,)


class Nemotron(Module[..., tuple[Tensor, ...]]):
    """The Nemotron model."""

    def __init__(
        self,
        config: NemotronConfig,
        kv_manager: PagedKVCacheManager,
    ) -> None:
        super().__init__()
        self.language_model = NemotronTextModel(config)
        self.config = config
        self.kv_manager = kv_manager

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *variadic_args,
    ) -> tuple[Tensor, ...]:
        kv_collection = _unflatten_kv_inputs(
            self.config, self.kv_manager, variadic_args
        )
        return self.language_model(
            tokens, kv_collection[0], return_n_logits, input_row_offsets
        )


def _unflatten_kv_inputs(
    config: NemotronConfig,
    kv_manager: PagedKVCacheManager,
    kv_inputs_flat: Sequence[Tensor],
) -> list[PagedCacheValues]:
    kv_params = config.kv_params
    n_devices = kv_params.n_devices
    fetch_types = kv_manager.params.get_symbolic_inputs()[0]
    len_of_kv_tuple_per_dev = len(list(fetch_types))
    kv_caches_per_dev: list[PagedCacheValues] = []
    for i in range(n_devices):
        start_idx = i * len_of_kv_tuple_per_dev

        kv_block = kv_inputs_flat[start_idx]
        cache_lengths = kv_inputs_flat[start_idx + 1]
        lookup_table = kv_inputs_flat[start_idx + 2]
        max_lengths = kv_inputs_flat[start_idx + 3]

        kv_caches_per_dev.append(
            PagedCacheValues(
                kv_blocks=BufferValue(kv_block),
                cache_lengths=TensorValue(cache_lengths),
                lookup_table=TensorValue(lookup_table),
                max_lengths=TensorValue(max_lengths),
            )
        )
    return kv_caches_per_dev
