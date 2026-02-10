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
"""Non-gated MLP with ReLU² for Nemotron."""
=======
"""Nemotron MLP with ReLU-squared activation."""
>>>>>>> upstream/main

from __future__ import annotations

from max import functional as F
from max.nn import Linear, Module
from max.tensor import Tensor


def relu_squared(x: Tensor) -> Tensor:
<<<<<<< HEAD
    """ReLU-squared: (ReLU(x))^2."""
=======
    """ReLU-squared activation: relu(x)^2."""
>>>>>>> upstream/main
    r = F.relu(x)
    return r * r


class NemotronMLP(Module[[Tensor], Tensor]):
<<<<<<< HEAD
    """Dense MLP: down_proj(relu_squared(up_proj(x))). No gate. Optional bias."""
=======
    """Non-gated MLP with ReLU-squared activation for Nemotron.

    Architecture: down_proj(relu_squared(up_proj(x)))
    """
>>>>>>> upstream/main

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
<<<<<<< HEAD
        bias: bool = True,
=======
        bias: bool = False,
>>>>>>> upstream/main
    ) -> None:
        super().__init__()
        self.up_proj = Linear(
            in_dim=hidden_size,
            out_dim=intermediate_size,
            bias=bias,
        )
        self.down_proj = Linear(
            in_dim=intermediate_size,
            out_dim=hidden_size,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(relu_squared(self.up_proj(x)))
