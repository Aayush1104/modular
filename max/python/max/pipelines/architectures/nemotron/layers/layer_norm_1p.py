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

"""Nemotron LayerNorm1P: LayerNorm with weight+1 trick."""

from __future__ import annotations

from max.nn.norm.layer_norm import LayerNorm, layer_norm
from max.tensor import Tensor


class NemotronLayerNorm1P(LayerNorm):
    """LayerNorm variant where the weight is offset by 1.

    This is the normalization used by Nemotron models, where the learned
    scale parameter gamma is applied as (gamma + 1) instead of gamma.
    This helps with training stability by initializing the effective scale
    near 1.0.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__(
            dim=dim,
            eps=eps,
            keep_dtype=True,
            elementwise_affine=True,
            use_bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        gamma, beta = self._affine_params(x)
        gamma = gamma + 1.0  # The "+1" trick
        return layer_norm(x, gamma, beta, self.eps, self.keep_dtype)
