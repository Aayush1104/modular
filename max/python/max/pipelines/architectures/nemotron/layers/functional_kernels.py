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

"""Fused kernel wrappers for Nemotron attention (no MoE)."""

from max import functional as F
from max.nn.legacy.kernels import (
    flash_attention_ragged as _flash_attention_ragged,
)
from max.nn.legacy.kernels import fused_qk_ragged_rope as _fused_qk_ragged_rope
from max.nn.legacy.kernels import (
    fused_qkv_ragged_matmul as _fused_qkv_ragged_matmul,
)

flash_attention_ragged = F.functional(_flash_attention_ragged)
fused_qkv_ragged_matmul = F.functional(_fused_qkv_ragged_matmul)
fused_qk_ragged_rope = F.functional(_fused_qk_ragged_rope)

__all__ = [
    "flash_attention_ragged",
    "fused_qk_ragged_rope",
    "fused_qkv_ragged_matmul",
]
