"""Partial Rotary Position Embedding GPU kernel in Mojo for Nemotron.

Nemotron applies RoPE to only the first `rotary_dim` dimensions of Q and K,
leaving the remaining `head_dim - rotary_dim` dimensions unchanged.

Default: partial_rotary_factor = 0.5, so half the dims get RoPE.

This fused kernel:
1. Applies RoPE to dims [0, rotary_dim)
2. Passes through dims [rotary_dim, head_dim) unchanged
3. Does both Q and K in a single kernel launch

Optimization: By fusing the partial split + rotation + concat, we avoid
extra global memory reads/writes from separate slice/rotate/concat ops.
"""

from gpu.host import DeviceContext
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim, grid_dim
from math import cos, sin


# ============================================================================
# GPU Kernel: Partial RoPE (fused for Nemotron)
# ============================================================================

fn partial_rope_kernel[
    dtype: DType,
    head_dim: Int,
    rotary_dim: Int,
    block_size: Int = 256,
](
    # Q output and input: [batch, num_heads, seq_len, head_dim]
    q_output: LayoutTensor[dtype, Layout.row_major(4), MutableAnyOrigin],
    q_input: LayoutTensor[dtype, Layout.row_major(4), MutableAnyOrigin],
    # K output and input: [batch, num_kv_heads, seq_len, head_dim]
    k_output: LayoutTensor[dtype, Layout.row_major(4), MutableAnyOrigin],
    k_input: LayoutTensor[dtype, Layout.row_major(4), MutableAnyOrigin],
    # cos, sin: [batch, seq_len, rotary_dim]
    cos_cache: LayoutTensor[dtype, Layout.row_major(3), MutableAnyOrigin],
    sin_cache: LayoutTensor[dtype, Layout.row_major(3), MutableAnyOrigin],
    # Dimensions
    batch_size: Int,
    num_q_heads: Int,
    num_kv_heads: Int,
    seq_len: Int,
):
    """Fused partial RoPE kernel for Nemotron.

    Each thread handles one (batch, head, position, dim_pair) combination.
    For dims in [0, rotary_dim): apply rotation
    For dims in [rotary_dim, head_dim): copy through

    Parameters:
        dtype: Data type.
        head_dim: Full head dimension.
        rotary_dim: Number of dimensions to apply RoPE to.
        block_size: Threads per block.

    Args:
        q_output/q_input: Query tensors.
        k_output/k_input: Key tensors.
        cos_cache/sin_cache: Precomputed cos/sin values.
        batch_size: Batch dimension.
        num_q_heads/num_kv_heads: Head counts.
        seq_len: Sequence length.
    """
    alias half_rotary = rotary_dim // 2

    # Global thread index
    var global_tid = block_idx.x * block_dim.x + thread_idx.x

    # Total work items: process Q heads
    var total_q_work = batch_size * num_q_heads * seq_len * (head_dim // 2)

    if global_tid < total_q_work:
        # Decompose linear index into (batch, head, pos, dim_pair)
        var remaining = global_tid
        var dim_pair = remaining % (head_dim // 2)
        remaining = remaining // (head_dim // 2)
        var pos = remaining % seq_len
        remaining = remaining // seq_len
        var head = remaining % num_q_heads
        var batch = remaining // num_q_heads

        var d0 = dim_pair * 2
        var d1 = dim_pair * 2 + 1

        if d0 < rotary_dim:
            # Apply RoPE rotation to this dimension pair
            var cos_val = cos_cache.load[1](batch * seq_len * rotary_dim + pos * rotary_dim + d0)
            var sin_val = sin_cache.load[1](batch * seq_len * rotary_dim + pos * rotary_dim + d0)

            var q0 = q_input.load[1](
                batch * num_q_heads * seq_len * head_dim
                + head * seq_len * head_dim
                + pos * head_dim + d0
            )
            var q1 = q_input.load[1](
                batch * num_q_heads * seq_len * head_dim
                + head * seq_len * head_dim
                + pos * head_dim + d1
            )

            # Rotation: [cos*x0 - sin*x1, sin*x0 + cos*x1]
            var q0_rot = q0 * cos_val - q1 * sin_val
            var q1_rot = q0 * sin_val + q1 * cos_val

            q_output.store[1](
                batch * num_q_heads * seq_len * head_dim
                + head * seq_len * head_dim
                + pos * head_dim + d0,
                q0_rot,
            )
            q_output.store[1](
                batch * num_q_heads * seq_len * head_dim
                + head * seq_len * head_dim
                + pos * head_dim + d1,
                q1_rot,
            )
        else:
            # Pass through: dims beyond rotary_dim are unchanged
            var q0 = q_input.load[1](
                batch * num_q_heads * seq_len * head_dim
                + head * seq_len * head_dim
                + pos * head_dim + d0
            )
            var q1 = q_input.load[1](
                batch * num_q_heads * seq_len * head_dim
                + head * seq_len * head_dim
                + pos * head_dim + d1
            )
            q_output.store[1](
                batch * num_q_heads * seq_len * head_dim
                + head * seq_len * head_dim
                + pos * head_dim + d0,
                q0,
            )
            q_output.store[1](
                batch * num_q_heads * seq_len * head_dim
                + head * seq_len * head_dim
                + pos * head_dim + d1,
                q1,
            )

    # Process K heads (similar logic, fewer heads for GQA)
    var total_k_work = batch_size * num_kv_heads * seq_len * (head_dim // 2)
    var k_tid = global_tid - total_q_work

    if k_tid >= 0 and k_tid < total_k_work:
        var remaining = k_tid
        var dim_pair = remaining % (head_dim // 2)
        remaining = remaining // (head_dim // 2)
        var pos = remaining % seq_len
        remaining = remaining // seq_len
        var head = remaining % num_kv_heads
        var batch = remaining // num_kv_heads

        var d0 = dim_pair * 2
        var d1 = dim_pair * 2 + 1

        if d0 < rotary_dim:
            var cos_val = cos_cache.load[1](batch * seq_len * rotary_dim + pos * rotary_dim + d0)
            var sin_val = sin_cache.load[1](batch * seq_len * rotary_dim + pos * rotary_dim + d0)

            var k0 = k_input.load[1](
                batch * num_kv_heads * seq_len * head_dim
                + head * seq_len * head_dim
                + pos * head_dim + d0
            )
            var k1 = k_input.load[1](
                batch * num_kv_heads * seq_len * head_dim
                + head * seq_len * head_dim
                + pos * head_dim + d1
            )

            k_output.store[1](
                batch * num_kv_heads * seq_len * head_dim
                + head * seq_len * head_dim
                + pos * head_dim + d0,
                k0 * cos_val - k1 * sin_val,
            )
            k_output.store[1](
                batch * num_kv_heads * seq_len * head_dim
                + head * seq_len * head_dim
                + pos * head_dim + d1,
                k0 * sin_val + k1 * cos_val,
            )
        else:
            # Pass through
            k_output.store[1](
                batch * num_kv_heads * seq_len * head_dim
                + head * seq_len * head_dim
                + pos * head_dim + d0,
                k_input.load[1](
                    batch * num_kv_heads * seq_len * head_dim
                    + head * seq_len * head_dim
                    + pos * head_dim + d0
                ),
            )
            k_output.store[1](
                batch * num_kv_heads * seq_len * head_dim
                + head * seq_len * head_dim
                + pos * head_dim + d1,
                k_input.load[1](
                    batch * num_kv_heads * seq_len * head_dim
                    + head * seq_len * head_dim
                    + pos * head_dim + d1
                ),
            )


# ============================================================================
# Host-side launch
# ============================================================================

fn launch_partial_rope[
    dtype: DType,
    head_dim: Int,
    rotary_dim: Int,
](
    ctx: DeviceContext,
    q_output: LayoutTensor[dtype, Layout.row_major(4), MutableAnyOrigin],
    q_input: LayoutTensor[dtype, Layout.row_major(4), MutableAnyOrigin],
    k_output: LayoutTensor[dtype, Layout.row_major(4), MutableAnyOrigin],
    k_input: LayoutTensor[dtype, Layout.row_major(4), MutableAnyOrigin],
    cos_cache: LayoutTensor[dtype, Layout.row_major(3), MutableAnyOrigin],
    sin_cache: LayoutTensor[dtype, Layout.row_major(3), MutableAnyOrigin],
    batch_size: Int,
    num_q_heads: Int,
    num_kv_heads: Int,
    seq_len: Int,
):
    """Launch the partial RoPE kernel.

    Parameters:
        dtype: Data type.
        head_dim: Full head dimension.
        rotary_dim: Rotary dimension (head_dim * partial_rotary_factor).

    Args:
        ctx: GPU device context.
        q_output/q_input: Query tensors.
        k_output/k_input: Key tensors.
        cos_cache/sin_cache: Precomputed values.
        batch_size, num_q_heads, num_kv_heads, seq_len: Dimensions.
    """
    alias BLOCK_SIZE = 256

    var total_work = (
        batch_size * (num_q_heads + num_kv_heads) * seq_len * (head_dim // 2)
    )
    var grid_size = (total_work + BLOCK_SIZE - 1) // BLOCK_SIZE

    ctx.enqueue_function[
        partial_rope_kernel[dtype, head_dim, rotary_dim, BLOCK_SIZE]
    ](
        grid_dim=grid_size,
        block_dim=BLOCK_SIZE,
        q_output,
        q_input,
        k_output,
        k_input,
        cos_cache,
        sin_cache,
        batch_size,
        num_q_heads,
        num_kv_heads,
        seq_len,
    )
