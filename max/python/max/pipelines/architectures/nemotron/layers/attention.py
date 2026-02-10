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
"""Standard causal attention with partial RoPE for Nemotron."""
=======
"""Nemotron Attention Layer with Partial RoPE."""
>>>>>>> upstream/main

from __future__ import annotations

import math
<<<<<<< HEAD

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn import Linear, Module
from max.nn.legacy.attention import MHAMaskVariant
from max.nn.legacy.kv_cache import KVCacheParams, PagedCacheValues
from max.tensor import Tensor

from ...common_layers.rotary_embedding import PartialRotaryEmbedding
=======
from collections.abc import Iterable
from functools import cached_property

from max import functional as F
from max.driver import CPU, Device
from max.dtype import DType
from max.nn import Linear, Module
from max.nn.legacy.attention import MHAMaskVariant
from max.nn.legacy.kv_cache import (
    KVCacheParams,
    PagedCacheValues,
)
from max.tensor import Tensor

>>>>>>> upstream/main
from .functional_kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
)


<<<<<<< HEAD
class NemotronAttention(Module[..., Tensor]):
    """Causal attention with partial RoPE. No sinks, no sliding window."""
=======
class PartialRotaryEmbedding(Module[..., Tensor]):
    """Rotary embedding applied to only a fraction of the head dimension.

    For Nemotron, only `partial_rotary_factor` (e.g. 0.5) of the head_dim
    gets rotary embeddings. The remaining dimensions are left unchanged.

    This is a self-contained implementation that produces freqs_cis in the
    flat format expected by the fused attention kernels:
    shape (max_seq_len * 2, head_dim).

    We pad the non-rotated dimensions with identity rotation values
    (cos=1, sin=0) so the fused kernel works unchanged.
    """

    dim: int
    n_heads: int
    theta: float
    max_seq_len: int
    head_dim: int
    device: Device
    interleaved: bool
    partial_rotary_factor: float
    _freqs_cis: Tensor | None

    def __init__(
        self,
        dim: int,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        device: Device,
        head_dim: int | None = None,
        interleaved: bool = True,
        partial_rotary_factor: float = 0.5,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim if head_dim is not None else dim // n_heads
        self.interleaved = interleaved
        self.device = device
        self.partial_rotary_factor = partial_rotary_factor
        self._freqs_cis = None

    @property
    def local_parameters(self) -> Iterable[tuple[str, Tensor]]:
        """Override to avoid freqs_cis being included in parameters."""
        return []

    def _compute_inv_freqs(self) -> Tensor:
        """Computes inv_freqs for the rotated portion of head_dim only."""
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)

        iota = F.arange(
            0, rotary_dim, step=2, dtype=DType.float64, device=self.device
        )
        inv_freq = F.cast(
            1.0 / (self.theta ** (iota / rotary_dim)), DType.float32
        )
        return inv_freq

    @cached_property
    def freqs_cis(self) -> Tensor:
        """Computes freqs_cis with identity padding for non-rotated dims.

        Returns a tensor of shape (max_seq_len * 2, head_dim) where the
        first `rotary_dim` entries have real rotation values and the remaining
        entries have cos=1, sin=0 (identity rotation), interleaved as
        [cos, sin, cos, sin, ...].
        """
        inv_freqs = self._compute_inv_freqs()

        t = F.arange(
            0, self.max_seq_len * 2, device=self.device, dtype=DType.float32
        )

        freqs = F.outer(t, inv_freqs)  # (max_seq_len*2, rotary_dim//2)
        cos_freqs = F.cos(freqs)
        sin_freqs = F.sin(freqs)

        # Stack to get (max_seq_len*2, rotary_dim//2, 2)
        rotary_pairs = F.stack([cos_freqs, sin_freqs], axis=-1)

        # Pad with identity rotation for non-rotated dimensions
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        pass_through_dim = (self.head_dim - rotary_dim) // 2

        if pass_through_dim > 0:
            seq_dim = self.max_seq_len * 2
            # Identity rotation: cos=1, sin=0
            ones = F.broadcast_to(
                F.constant(1.0, dtype=DType.float32, device=self.device),
                shape=(seq_dim, pass_through_dim, 1),
            )
            zeros = F.broadcast_to(
                F.constant(0.0, dtype=DType.float32, device=self.device),
                shape=(seq_dim, pass_through_dim, 1),
            )
            identity_pairs = F.concat(
                [ones, zeros], axis=-1
            )  # (seq_dim, pass_through_dim, 2)
            combined = F.concat(
                [rotary_pairs, identity_pairs], axis=1
            )  # (seq_dim, head_dim//2, 2)
        else:
            combined = rotary_pairs

        # Flatten to (max_seq_len * 2, head_dim)
        d1, d2, d3 = combined.shape
        return F.reshape(combined, [d1, d2 * d3])


class NemotronAttention(Module[..., Tensor]):
    """Standard causal attention for Nemotron.

    Unlike GPT OSS, this has no sink tokens, no sliding window,
    and always uses CAUSAL_MASK.
    """
>>>>>>> upstream/main

    def __init__(
        self,
        *,
        rope: PartialRotaryEmbedding,
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
<<<<<<< HEAD
                f"{self.kv_params.cache_strategy} cache strategy not supported"
=======
                f"{self.kv_params.cache_strategy} cache strategy, not supported"
                " in Attention layer."
>>>>>>> upstream/main
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
<<<<<<< HEAD
=======

>>>>>>> upstream/main
        self.o_proj = Linear(
            in_dim=self.q_weight_dim,
            out_dim=hidden_size,
            bias=self.has_bias,
        )

    @property
    def wqkv(self) -> Tensor:
<<<<<<< HEAD
        return F.concat(
            [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight],
            axis=0,
        )

    @property
    def wqkv_bias(self) -> Tensor | None:
        if not self.has_bias:
            return None
=======
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

>>>>>>> upstream/main
        if (
            self.q_proj.bias is None
            or self.k_proj.bias is None
            or self.v_proj.bias is None
        ):
<<<<<<< HEAD
            raise ValueError("has_bias=True but projection bias is None")
=======
            raise ValueError(
                "Projection bias is None, but has_bias=True was specified."
            )

>>>>>>> upstream/main
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
<<<<<<< HEAD
        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())

        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=self.wqkv,
=======

        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())
        # Fused QKV matmul
        wqkv = self.wqkv
        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=wqkv,
>>>>>>> upstream/main
            bias=self.wqkv_bias,
            input_row_offsets=kwargs["input_row_offsets"],
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )
<<<<<<< HEAD
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        freqs_cis = F.cast(self.rope.freqs_cis, xq.dtype).to(xq.device)
=======
        # Reshape for RoPE
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        # Apply rotary embedding (partial RoPE handled via padded freqs_cis)
        rope = self.rope
        freqs_cis = F.cast(rope.freqs_cis, xq.dtype).to(xq.device)
>>>>>>> upstream/main
        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            kwargs["input_row_offsets"],
            kv_collection,
            freqs_cis,
            layer_idx,
<<<<<<< HEAD
            interleaved=self.rope.interleaved,
        )

=======
            interleaved=rope.interleaved,
        )

        # Flash attention (standard causal, no sinks)
>>>>>>> upstream/main
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
<<<<<<< HEAD
            local_window_size=-1,
            sink_weights=None,
=======
>>>>>>> upstream/main
        )
        attn_out = F.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)
