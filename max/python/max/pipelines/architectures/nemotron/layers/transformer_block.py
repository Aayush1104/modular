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

"""Implements the Nemotron transformer block."""

from __future__ import annotations

from max.nn import Module
from max.nn.legacy.kv_cache import PagedCacheValues
from max.tensor import Tensor

from .attention import NemotronAttention
from .layer_norm_1p import NemotronLayerNorm1P
from .mlp import NemotronMLP


class NemotronTransformerBlock(Module[..., Tensor]):
    """Pre-norm residual transformer block for Nemotron.

    Uses LayerNorm1P for normalization and a dense MLP with ReLU-squared.
    """

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
        # Pre-norm + attention + residual
        residual = x
        norm_xs = self.input_layernorm(x)
        attn_out = self.self_attn(
            norm_xs,
            kv_collection,
            input_row_offsets=input_row_offsets,
            **kwargs,
        )
        hidden_states = residual + attn_out

        # Pre-norm + MLP + residual
        residual = hidden_states
        norm_xs = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(norm_xs)
        hidden_states = residual + mlp_out

        return hidden_states
