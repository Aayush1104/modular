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
"""Nemotron model with partial RoPE, LayerNorm, and squared ReLU MLP.

Uses the new ``max.nn.Module`` API (not the legacy transformer).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterable
from functools import cached_property

from max import functional as F
from max.driver import CPU, Device
from max.dtype import DType
from max.graph import BufferValue, TensorValue
from max.kv_cache import PagedKVCacheManager
from max.nn import Embedding, Linear, Module
from max.nn.legacy.attention import MHAMaskVariant
from max.nn.legacy.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.norm import LayerNorm
from max.nn.sequential import ModuleList
from max.tensor import Tensor

from .functional_kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
)
from .model_config import NemotronConfig


# ---------------------------------------------------------------------------
# Partial Rotary Embedding
# ---------------------------------------------------------------------------


class PartialRotaryEmbedding(Module[..., Tensor]):
    """Rotary embedding that applies RoPE to only a fraction of head dims.

    Nemotron uses ``partial_rotary_factor=0.5``, meaning only the first half
    of each head's dimensions receive rotary positional encodings.  The
    remaining dimensions pass through unchanged (identity rotation: cos=1,
    sin=0).
    """

    dim: int
    n_heads: int
    theta: float
    max_seq_len: int
    head_dim: int
    device: Device
    partial_rotary_factor: float
    rotary_dim: int
    _freqs_cis: Tensor | None = None
    interleaved: bool

    def __init__(
        self,
        dim: int,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        device: Device,
        partial_rotary_factor: float = 0.5,
        head_dim: int | None = None,
        interleaved: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device
        self.head_dim = head_dim if head_dim is not None else dim // n_heads
        self.interleaved = interleaved
        self.partial_rotary_factor = partial_rotary_factor
        self.rotary_dim = int(self.head_dim * partial_rotary_factor)

    def _compute_inv_freqs(self) -> Tensor:
        """Computes inverse frequencies for the rotary dimensions only."""
        n = self.rotary_dim
        iota = F.arange(0, n, step=2, dtype=DType.float64, device=self.device)
        inv_freq = F.cast(1.0 / (self.theta ** (iota / n)), DType.float32)
        return inv_freq

    def freqs_cis_base(self) -> Tensor:
        """Computes frequency tensor padded with identity for non-rotary dims.

        Returns:
            Tensor of shape ``(max_seq_len * 2, head_dim // 2, 2)`` where
            the first ``rotary_dim // 2`` pairs carry real rotation angles
            and the rest are ``(cos=1, sin=0)`` (identity).
        """
        if self._freqs_cis is None:
            inv_freqs = self._compute_inv_freqs()  # [rotary_dim // 2]

            t = F.arange(
                0,
                self.max_seq_len * 2,
                device=self.device,
                dtype=DType.float32,
            )
            freqs = F.outer(t, inv_freqs)  # [max_seq_len*2, rotary_dim//2]

            cos_freqs = F.cos(freqs)
            sin_freqs = F.sin(freqs)

            # Pad non-rotary dims with identity rotation (cos=1, sin=0).
            non_rotary_pairs = (self.head_dim - self.rotary_dim) // 2
            if non_rotary_pairs > 0:
                seq_len_dim = cos_freqs.shape[0]
                ones_pad = F.broadcast_to(
                    F.constant(1.0, DType.float32, self.device),
                    shape=(seq_len_dim, non_rotary_pairs),
                )
                zeros_pad = F.broadcast_to(
                    F.constant(0.0, DType.float32, self.device),
                    shape=(seq_len_dim, non_rotary_pairs),
                )
                cos_freqs = F.concat(
                    [cos_freqs, ones_pad], axis=-1
                )  # [seq*2, head_dim//2]
                sin_freqs = F.concat(
                    [sin_freqs, zeros_pad], axis=-1
                )  # [seq*2, head_dim//2]

            self._freqs_cis = F.stack(
                [cos_freqs, sin_freqs], axis=-1
            )  # [max_seq_len*2, head_dim//2, 2]
        assert isinstance(self._freqs_cis, Tensor)
        return self._freqs_cis

    @cached_property
    def freqs_cis(self) -> Tensor:
        freqs = self.freqs_cis_base()
        d1, d2, d3 = freqs.shape  # (max_seq_len * 2, head_dim // 2, 2)
        self._freqs_cis = F.reshape(freqs, [d1, d2 * d3])
        assert isinstance(self._freqs_cis, Tensor)
        return self._freqs_cis

    def compute_scale(self, user_scale: float | None = None) -> float:
        return user_scale if user_scale else math.sqrt(1.0 / self.head_dim)

    @property
    def local_parameters(self) -> Iterable[tuple[str, Tensor]]:
        """Exclude precomputed freqs_cis from the model parameter set."""
        return []

    def forward(self, x: Tensor) -> Tensor:
        """Not used directly — freqs_cis is consumed by fused RoPE kernel."""
        raise NotImplementedError(
            "PartialRotaryEmbedding.forward is not used; "
            "use .freqs_cis with fused_qk_ragged_rope instead."
        )


# ---------------------------------------------------------------------------
# Nemotron MLP (non-gated, squared ReLU)
# ---------------------------------------------------------------------------


class NemotronMLP(Module[[Tensor], Tensor]):
    """Non-gated MLP with squared ReLU activation.

    Unlike the LLaMA-style gated MLP, Nemotron uses::

        output = down_proj(squared_relu(up_proj(x)))

    where ``squared_relu(x) = relu(x) ** 2``.
    """

    def __init__(
        self,
        hidden_dim: int,
        feed_forward_length: int,
    ) -> None:
        super().__init__()
        self.up_proj = Linear(
            in_dim=hidden_dim,
            out_dim=feed_forward_length,
            bias=False,
        )
        self.down_proj = Linear(
            in_dim=feed_forward_length,
            out_dim=hidden_dim,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: up_proj -> squared ReLU -> down_proj."""
        h = self.up_proj(x)
        h = F.relu(h)
        h = h * h  # squared ReLU
        return self.down_proj(h)


# ---------------------------------------------------------------------------
# Nemotron Attention (with partial RoPE)
# ---------------------------------------------------------------------------


class NemotronAttention(Module[..., Tensor]):
    """Grouped-query attention with partial rotary positional embeddings.

    Uses the fused ragged QKV matmul and flash-attention kernels.
    """

    def __init__(
        self,
        *,
        rope: PartialRotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        has_bias: bool = False,
        scale: float | None = None,
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

        self.q_weight_dim = kv_params.head_dim * num_attention_heads
        self.kv_weight_dim = kv_params.head_dim * num_key_value_heads

        self.q_proj = Linear(
            in_dim=hidden_size, out_dim=self.q_weight_dim, bias=has_bias
        )
        self.k_proj = Linear(
            in_dim=hidden_size, out_dim=self.kv_weight_dim, bias=has_bias
        )
        self.v_proj = Linear(
            in_dim=hidden_size, out_dim=self.kv_weight_dim, bias=has_bias
        )
        self.o_proj = Linear(
            in_dim=self.q_weight_dim, out_dim=hidden_size, bias=False
        )

    @property
    def wqkv(self) -> Tensor:
        """Concatenated Q/K/V weight matrices."""
        return F.concat(
            [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight],
            axis=0,
        )

    @property
    def wqkv_bias(self) -> Tensor | None:
        """Concatenated Q/K/V bias vectors (or None)."""
        if not self.has_bias:
            return None
        assert isinstance(self.q_proj.bias, Tensor)
        assert isinstance(self.k_proj.bias, Tensor)
        assert isinstance(self.v_proj.bias, Tensor)
        return F.concat(
            [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], axis=0
        )

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        **kwargs,
    ) -> Tensor:
        total_seq_len = x.shape[0]
        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())

        # Fused QKV matmul + KV cache write.
        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=self.wqkv,
            bias=self.wqkv_bias,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )

        # Reshape Q for RoPE: (total_seq_len, n_heads, head_dim)
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        # Fused Q/K RoPE application (uses padded freqs_cis for partial RoPE).
        freqs_cis = F.cast(self.rope.freqs_cis, xq.dtype).to(xq.device)
        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            input_row_offsets,
            kv_collection,
            freqs_cis,
            layer_idx,
            interleaved=self.rope.interleaved,
        )

        # Flash attention.
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )
        attn_out = F.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)


# ---------------------------------------------------------------------------
# Nemotron Transformer Block
# ---------------------------------------------------------------------------


class NemotronTransformerBlock(Module[..., Tensor]):
    """Pre-norm transformer block with LayerNorm, attention, and MLP."""

    def __init__(
        self,
        attention: NemotronAttention,
        mlp: NemotronMLP,
        input_layernorm: LayerNorm,
        post_attention_layernorm: LayerNorm,
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
        # Attention with pre-norm and residual.
        residual = x
        h = self.input_layernorm(x)
        attn_out = self.self_attn(
            h,
            kv_collection,
            input_row_offsets=input_row_offsets,
        )
        h = residual + attn_out

        # MLP with pre-norm and residual.
        residual = h
        h = self.post_attention_layernorm(h)
        mlp_out = self.mlp(h)
        return residual + mlp_out


# ---------------------------------------------------------------------------
# Nemotron Text Model (full decoder)
# ---------------------------------------------------------------------------


class NemotronTextModel(
    Module[[Tensor, PagedCacheValues, Tensor, Tensor], tuple[Tensor, ...]]
):
    """Nemotron decoder-only language model.

    Composed of:
    - Token embedding
    - N transformer blocks (LayerNorm + partial-RoPE attention + squared-ReLU MLP)
    - Final LayerNorm
    - Linear LM head
    """

    def __init__(self, config: NemotronConfig) -> None:
        super().__init__()
        self.devices = config.devices

        # Partial Rotary Embedding.
        rope = PartialRotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            device=config.devices[0].to_device(),
            partial_rotary_factor=config.partial_rotary_factor,
            head_dim=config.head_dim,
            interleaved=config.interleaved_rope_weights,
        )

        # Embedding.
        self.embed_tokens = Embedding(config.vocab_size, dim=config.hidden_size)

        # Final LayerNorm.
        self.norm = LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            keep_dtype=False,
        )

        # LM head.
        self.lm_head = Linear(
            in_dim=config.hidden_size,
            out_dim=config.vocab_size,
            bias=False,
        )

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # LayerNorm factory for each block.
        create_norm = functools.partial(
            LayerNorm,
            config.hidden_size,
            eps=config.norm_eps,
            keep_dtype=False,
        )

        # Attention scale.
        attention_scale = config.attention_multiplier

        # Transformer blocks.
        layers = [
            NemotronTransformerBlock(
                attention=NemotronAttention(
                    rope=rope,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    layer_idx=i,
                    has_bias=config.attention_bias,
                    scale=attention_scale,
                ),
                mlp=NemotronMLP(
                    hidden_dim=config.hidden_size,
                    feed_forward_length=config.intermediate_size,
                ),
                input_layernorm=create_norm(),
                post_attention_layernorm=create_norm(),
            )
            for i in range(config.num_hidden_layers)
        ]

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

        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=h.device)
            h = layer(
                layer_idx_tensor,
                h,
                kv_collection,
                input_row_offsets=input_row_offsets,
            )

        # Logits for the last token of each sequence in the batch.
        last_token_indices = input_row_offsets[1:] - 1
        last_h = F.gather(h, last_token_indices, axis=0)
        last_logits = F.cast(self.lm_head(self.norm(last_h)), DType.float32)

        return (last_logits,)


# ---------------------------------------------------------------------------
# Top-level wrapper (handles KV cache unflattening)
# ---------------------------------------------------------------------------


class Nemotron(Module[..., tuple[Tensor, ...]]):
    """Top-level Nemotron model that unflattens KV cache inputs."""

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
    kv_inputs_flat: tuple[Tensor, ...],
) -> list[PagedCacheValues]:
    """Converts flat KV cache inputs into structured PagedCacheValues."""
    kv_params = config.kv_params
    n_devices = kv_params.n_devices
    fetch_types = kv_manager.params.get_symbolic_inputs()[0]
    len_of_kv_tuple_per_dev = len(list(fetch_types))
    kv_caches_per_dev: list[PagedCacheValues] = []
    for i in range(n_devices):
        start_idx = i * len_of_kv_tuple_per_dev
        kv_caches_per_dev.append(
            PagedCacheValues(
                kv_blocks=BufferValue(kv_inputs_flat[start_idx]),
                cache_lengths=TensorValue(kv_inputs_flat[start_idx + 1]),
                lookup_table=TensorValue(kv_inputs_flat[start_idx + 2]),
                max_lengths=TensorValue(kv_inputs_flat[start_idx + 3]),
            )
        )
    return kv_caches_per_dev
