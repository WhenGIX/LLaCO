# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Independent Bridge class for native Nemotron spherical runs."""

from megatron.bridge.diffusion.conversion.nemotron_labs_diffusion.nemotron_labs_diffusion_bridge import (
    NemotronLabsDiffusionBridge as NativeNemotronLabsDiffusionBridge,
)


class NemotronLabsDiffusionBridge(NativeNemotronLabsDiffusionBridge):
    """Native HF-to-MCore conversion bridge used by the spherical recipe."""

    def provider_bridge(self, hf_pretrained):
        """Return the independent spherical provider while retaining native mappings."""
        from .provider import NemotronLabsDiffusionModelProvider

        native = super().provider_bridge(hf_pretrained)
        values = {field: getattr(native, field) for field in native.__dataclass_fields__ if hasattr(native, field)}
        return NemotronLabsDiffusionModelProvider(**values)


__all__ = ["NemotronLabsDiffusionBridge"]
