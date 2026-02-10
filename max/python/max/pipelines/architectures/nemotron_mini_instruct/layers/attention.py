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

"""Nemotron Attention Layer."""

from __future__ import annotations

import math

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn import Linear, Module
from max.nn.legacy.attention import MHAMaskVariant
from max.nn.legacy.kv_cache import (
    KVCacheParams,
    PagedCacheValues,
)
from max.tensor import Tensor

from ...common_layers.rotary_embedding import RotaryEmbedding
from .functional_kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
)


class NemotronAttention(Module[..., Tensor]):
    """Attention layer for Nemotron models.

    Uses standard causal attention with partial rotary embeddings.
    """

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        scale: float | None = None,
        has_bias: bool = False,
    ) -> None:
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.kv_params = kv_params
        self.has_bias = has_bias
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )

        if not self.kv_params.cache_strategy.uses_opaque():
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy, not supported"
                " in Attention layer."
            )

        self.q_weight_dim = self.kv_params.head_dim * num_attention_heads
        self.kv_weight_dim = self.kv_params.head_dim * num_key_value_heads

        self.q_proj = Linear(
            in_dim=hidden_size,
            out_dim=self.q_weight_dim,
            bias=self.has_bias,
        )
        self.k_proj = Linear(
            in_dim=hidden_size,
            out_dim=self.kv_weight_dim,
            bias=self.has_bias,
        )
        self.v_proj = Linear(
            in_dim=hidden_size,
            out_dim=self.kv_weight_dim,
            bias=self.has_bias,
        )
        self.o_proj = Linear(
            in_dim=self.q_weight_dim,
            out_dim=hidden_size,
            bias=self.has_bias,
        )

    @property
    def wqkv(self) -> Tensor:
        """The concatenation of q, k, and v weight vectors."""
        wq: Tensor = self.q_proj.weight
        wk: Tensor = self.k_proj.weight
        wv: Tensor = self.v_proj.weight
        return F.concat([wq, wk, wv], axis=0)

    @property
    def wqkv_bias(self) -> Tensor | None:
        """The concatenation of q, k, and v bias weight vectors."""
        if not self.has_bias:
            return None

        if (
            self.q_proj.bias is None
            or self.k_proj.bias is None
            or self.v_proj.bias is None
        ):
            raise ValueError(
                "Projection bias is None, but has_bias=True was specified."
            )

        return F.concat(
            [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], axis=0
        )

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        **kwargs,
    ) -> Tensor:
        total_seq_len = x.shape[0]

        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())
        # Fused QKV projection + KV cache write.
        wqkv = self.wqkv
        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=wqkv,
            bias=self.wqkv_bias,
            input_row_offsets=kwargs["input_row_offsets"],
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )
        # Reshape for RoPE application.
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        # Apply partial rotary embedding via freqs_cis (sized to rotary_dim).
        freqs_cis = F.cast(self.rope.freqs_cis, xq.dtype).to(xq.device)
        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            kwargs["input_row_offsets"],
            kv_collection,
            freqs_cis,
            layer_idx,
            interleaved=self.rope.interleaved,
        )

        # Flash attention with standard causal mask.
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )
        attn_out = F.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)
