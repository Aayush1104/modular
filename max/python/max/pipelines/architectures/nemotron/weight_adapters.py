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
"""Weight adapters for Nemotron models.

Maps HuggingFace safetensor keys to the new Module-based naming hierarchy.

HuggingFace safetensor keys::

    model.embed_tokens.weight
    model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    model.layers.{i}.mlp.{up,down}_proj.weight
    model.layers.{i}.input_layernorm.{weight,bias}
    model.layers.{i}.post_attention_layernorm.{weight,bias}
    model.norm.{weight,bias}
    lm_head.weight

MAX Module hierarchy keys::

    language_model.embed_tokens.weight
    language_model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    language_model.layers.{i}.mlp.{up,down}_proj.weight
    language_model.layers.{i}.input_layernorm.{weight,bias}
    language_model.layers.{i}.post_attention_layernorm.{weight,bias}
    language_model.norm.{weight,bias}
    language_model.lm_head.weight
"""

from __future__ import annotations

from max.graph.weights import WeightData, Weights
from max.pipelines.lib import PipelineConfig

# Ordered mapping from HuggingFace safetensor prefixes to MAX internal names.
NEMOTRON_SAFETENSOR_MAPPING: dict[str, str] = {
    "model.embed_tokens.": "language_model.embed_tokens.",
    "model.norm.": "language_model.norm.",
    "lm_head.": "language_model.lm_head.",
    "model.layers.": "language_model.layers.",
}


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    pipeline_config: PipelineConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Converts HuggingFace safetensor keys to MAX weight names."""
    new_state_dict: dict[str, WeightData] = {}
    for safetensor_name, value in state_dict.items():
        max_name = safetensor_name
        for before, after in NEMOTRON_SAFETENSOR_MAPPING.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = value.data()

    # Handle dtype casting if configured.
    model_config = pipeline_config.model
    if model_config._applied_dtype_cast_from:
        cast_from = model_config._applied_dtype_cast_from
        cast_to = model_config._applied_dtype_cast_to
        assert cast_to, (
            "Invalid configuration: _applied_dtype_cast_to is not set but "
            "_applied_dtype_cast_from is set."
        )
        for key, weight_data in new_state_dict.items():
            if weight_data.dtype == cast_from.dtype:
                new_state_dict[key] = weight_data.astype(cast_to.dtype)

    return new_state_dict
