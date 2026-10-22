# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 hardware-isolated model entry point."""

import importlib
from typing import TYPE_CHECKING

from vllm.platforms import current_platform

from .quant_config import DeepseekV4FP8Config

# Quantization lookup imports this package for both V4 and V4.1. Keep the
# platform-specific model implementations out of that shared import path.
if TYPE_CHECKING:
    if current_platform.is_rocm():
        from .amd.dspark import DSparkDeepseekV4ForCausalLM
        from .amd.vl_model import DeepseekV41ForCausalLM
    else:
        from .nvidia.dspark import DSparkDeepseekV4ForCausalLM
        from .nvidia.vl_model import DeepseekV41ForCausalLM

__all__ = [
    "DSparkDeepseekV4ForCausalLM",
    "DeepseekV4FP8Config",
    "DeepseekV41ForCausalLM",
]


def __getattr__(name: str):
    if name not in {"DSparkDeepseekV4ForCausalLM", "DeepseekV41ForCausalLM"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    platform = "amd" if current_platform.is_rocm() else "nvidia"
    module_name = "dspark" if name.startswith("DSpark") else "vl_model"
    module = importlib.import_module(f".{platform}.{module_name}", __name__)
    return getattr(module, name)
