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

"""LayerNorm with weight + 1 offset (Nemotron-style)."""

from __future__ import annotations

from max.nn.norm.layer_norm import LayerNorm, layer_norm
from max.tensor import Tensor


class NemotronLayerNorm1P(LayerNorm):
    """LayerNorm with weight + 1 offset (Nemotron-style).

    Computes layer_norm(x, weight + 1, bias) instead of layer_norm(x, weight, bias).
    Weights initialize to ones, so the effective initial scale is 2.0.
    """

    def forward(self, x: Tensor) -> Tensor:
        gamma, beta = self._affine_params(x)
        gamma = gamma + 1.0
        return layer_norm(x, gamma, beta, self.eps, self.keep_dtype)
