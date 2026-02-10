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

"""Nemotron decoder layer: LayerNorm1P → Attention → + residual → LayerNorm1P → MLP → + residual."""

from __future__ import annotations

from max.nn import Module
from max.nn.legacy.kv_cache import PagedCacheValues
from max.tensor import Tensor

from .attention import NemotronAttention
from .layer_norm_1p import NemotronLayerNorm1P
from .mlp import NemotronMLP


class NemotronTransformerBlock(Module[..., Tensor]):
    """Pre-norm residual block with LayerNorm1P, causal attention, and dense MLP."""

    def __init__(
        self,
        attention: NemotronAttention,
        mlp: NemotronMLP,
        input_layernorm: NemotronLayerNorm1P,
        post_attention_layernorm: NemotronLayerNorm1P,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm

    def forward(
        self,
        layer_idx: Tensor,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        **kwargs,
    ) -> Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = residual + self.self_attn(x, kv_collection=kv_collection, **kwargs)

        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)
        return x
