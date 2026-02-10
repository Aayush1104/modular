"""Squared ReLU (ReLU²) custom GPU kernel in Mojo.

This kernel implements the activation function used in Nemotron-4:
    f(x) = relu(x)² = max(0, x)²

This fused kernel avoids materializing the intermediate ReLU result
in global memory, giving ~2x speedup over sequential relu + square ops.

Can be registered as a custom op in MAX Graph via:
    graph.custom("squared_relu", inputs=[x], custom_ops_path="path/to/pkg")
"""

from gpu.host import DeviceContext
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim, grid_dim


# ============================================================================
# GPU Kernel: Squared ReLU (fused relu + square)
# ============================================================================

fn squared_relu_kernel[
    dtype: DType,
    block_size: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(2), MutableAnyOrigin],
    input: LayoutTensor[dtype, Layout.row_major(2), MutableAnyOrigin],
    num_elements: Int,
):
    """Fused squared ReLU kernel for GPU.

    Each thread processes one element:
        output[i] = max(0, input[i])²

    Parameters:
        dtype: Data type (float32, float16, bfloat16).
        block_size: CUDA block size (threads per block).

    Args:
        output: Output tensor (same shape as input).
        input: Input tensor.
        num_elements: Total number of elements.
    """
    var tid = block_idx.x * block_dim.x + thread_idx.x

    if tid < num_elements:
        var x = input.load[1](tid)

        # Fused ReLU²: max(0, x)²
        var relu_x = max(x, Scalar[dtype](0))
        var result = relu_x * relu_x

        output.store[1](tid, result)


fn squared_relu_vectorized_kernel[
    dtype: DType,
    block_size: Int,
    vec_width: Int = 4,
](
    output: LayoutTensor[dtype, Layout.row_major(2), MutableAnyOrigin],
    input: LayoutTensor[dtype, Layout.row_major(2), MutableAnyOrigin],
    num_elements: Int,
):
    """Vectorized squared ReLU kernel.

    Processes vec_width elements per thread for better memory throughput.

    Parameters:
        dtype: Data type.
        block_size: Threads per block.
        vec_width: Elements per thread (default 4).

    Args:
        output: Output tensor.
        input: Input tensor.
        num_elements: Total number of elements.
    """
    var tid = (block_idx.x * block_dim.x + thread_idx.x) * vec_width

    if tid + vec_width <= num_elements:
        # Vectorized load
        var vals = input.load[vec_width](tid)

        # Fused ReLU² on vector
        @parameter
        for i in range(vec_width):
            var x = vals[i]
            var relu_x = max(x, Scalar[dtype](0))
            vals[i] = relu_x * relu_x

        # Vectorized store
        output.store[vec_width](tid, vals)
    elif tid < num_elements:
        # Handle tail elements
        for i in range(min(vec_width, num_elements - tid)):
            var x = input.load[1](tid + i)
            var relu_x = max(x, Scalar[dtype](0))
            output.store[1](tid + i, relu_x * relu_x)


# ============================================================================
# Host-side launch function
# ============================================================================

fn launch_squared_relu[
    dtype: DType,
](
    ctx: DeviceContext,
    output: LayoutTensor[dtype, Layout.row_major(2), MutableAnyOrigin],
    input: LayoutTensor[dtype, Layout.row_major(2), MutableAnyOrigin],
    num_elements: Int,
):
    """Launch the squared ReLU kernel on GPU.

    Automatically selects block size and grid dimensions.

    Parameters:
        dtype: Data type.

    Args:
        ctx: GPU device context.
        output: Output tensor.
        input: Input tensor.
        num_elements: Total number of elements.
    """
    alias BLOCK_SIZE = 256
    alias VEC_WIDTH = 4

    var num_vec_elements = num_elements // VEC_WIDTH
    var grid_size = (num_vec_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    ctx.enqueue_function[
        squared_relu_vectorized_kernel[dtype, BLOCK_SIZE, VEC_WIDTH]
    ](
        grid_dim=grid_size,
        block_dim=BLOCK_SIZE,
        output,
        input,
        num_elements,
    )
