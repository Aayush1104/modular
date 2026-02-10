"""Nemotron-4 architecture for MAX Serve.

This package provides a Nemotron-4 model implementation compatible with
MAX Serve's custom architecture system.

Usage with MAX Serve:
    max serve \\
        --model nvidia/nemotron-3-8b-base-4k-hf \\
        --custom-architectures ./nemotron

The ARCHITECTURES list below makes the model discoverable by MAX's
architecture registry.
"""

from .config import NemotronConfig
from .model import NemotronModel, NemotronStandaloneModel
from .weight_adapters import NemotronWeightAdapter

# This is required by MAX's custom architecture discovery mechanism.
# The string must match the model's `architectures` field in HuggingFace config.json
# For Nemotron-4: "NemotronForCausalLM"
ARCHITECTURES = ["NemotronForCausalLM"]

__all__ = [
    "ARCHITECTURES",
    "NemotronConfig",
    "NemotronModel",
    "NemotronStandaloneModel",
    "NemotronWeightAdapter",
]
