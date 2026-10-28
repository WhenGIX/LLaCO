"""Recipe factory for Plan 3."""
from megatron.bridge.diffusion.models.nemotron_rms_plan3.provider import Plan3RMSConfig

def plan3_rms_config(**overrides):
    fields = set(Plan3RMSConfig.__dataclass_fields__)
    unknown = sorted(set(overrides) - fields)
    if unknown: raise ValueError(f"unknown Plan 3 config fields: {', '.join(unknown)}")
    return Plan3RMSConfig(**overrides)

__all__ = ["Plan3RMSConfig", "plan3_rms_config"]
