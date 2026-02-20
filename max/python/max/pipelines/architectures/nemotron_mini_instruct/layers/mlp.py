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

"""Non-gated MLP with squared ReLU activation for Nemotron models."""

from __future__ import annotations

from max import functional as F
from max.nn.linear import Linear
from max.nn.module import Module
from max.tensor import Tensor


class NemotronMLP(Module[[Tensor], Tensor]):
    """Non-gated MLP with squared ReLU activation.

    Computes: down_proj(relu(up_proj(x))^2)

    Only 2 projections (no gate_proj), using squared ReLU activation.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.up_proj = Linear(
            in_dim=hidden_dim,
            out_dim=intermediate_size,
            bias=bias,
        )
        self.down_proj = Linear(
            in_dim=intermediate_size,
            out_dim=hidden_dim,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.up_proj(x)
        h = F.relu(h)
        h = h * h  # squared ReLU
        return self.down_proj(h)
