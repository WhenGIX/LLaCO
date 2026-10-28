# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shared constants for the native Nemotron spherical training protocol."""


# v5 changes only the semantic-time sampler.  The v4 checkpoint remains a
# valid warm start because model, optimizer, topology, and data are unchanged.
LEGACY_SPHERICAL_PROTOCOL_VERSION = "nemotron-spherical-qlog-t-v4"
SPHERICAL_PROTOCOL_VERSION = "nemotron-spherical-qlog-t-v5"
# Keep the data identity on the pre-sampling-change contract.  Changing the
# sampler must not select a different assistant-target index or rebuild data.
SPHERICAL_DATA_FINGERPRINT_PROTOCOL_VERSION = LEGACY_SPHERICAL_PROTOCOL_VERSION
SPHERICAL_DATA_MANIFEST_FINGERPRINT = (
    "838fb265c4e70a0d07f6c9f3ee9d3cfacb034103d5f758780058eb6d21ac7982"
)
SPHERICAL_DATA_TOTAL_TARGETS = 355_000
SPHERICAL_DATA_VALID_TARGETS = 347_165
SPHERICAL_DATA_DROPPED_TARGETS = 7_835
# Native v3 was trained with a 512-wide condition vector.  Keep that as the
# compatibility default while allowing a fresh HF-parity model to use 1024.
SPHERICAL_CONDITION_DIM = 512
HF_SPHERICAL_CONDITION_DIM = 1024
SUPPORTED_SPHERICAL_CONDITION_DIMS = (SPHERICAL_CONDITION_DIM, HF_SPHERICAL_CONDITION_DIM)
# The default v5 sampler is pure semantic-time sampling. A configured optional
# gamma-uniform mixture is recorded in runtime protocol metadata; default noisy
# blocks draw an integer from the complete inclusive grid t=k/1000.
GAMMA_UNIFORM_FRACTION = 0.0
T_UNIFORM_FRACTION = 1.0
T_UNIFORM_GRID_SIZE = 1001
T_UNIFORM_GRID_DENOMINATOR = 1000
T_UNIFORM_GRID_STEP = 0.001
T_UNIFORM_ENDPOINT_INCLUSIVE = True


__all__ = [
    "SPHERICAL_DATA_DROPPED_TARGETS",
    "SPHERICAL_DATA_FINGERPRINT_PROTOCOL_VERSION",
    "SPHERICAL_DATA_MANIFEST_FINGERPRINT",
    "SPHERICAL_DATA_TOTAL_TARGETS",
    "SPHERICAL_DATA_VALID_TARGETS",
    "SPHERICAL_CONDITION_DIM",
    "HF_SPHERICAL_CONDITION_DIM",
    "LEGACY_SPHERICAL_PROTOCOL_VERSION",
    "SPHERICAL_PROTOCOL_VERSION",
    "SUPPORTED_SPHERICAL_CONDITION_DIMS",
    "GAMMA_UNIFORM_FRACTION",
    "T_UNIFORM_FRACTION",
    "T_UNIFORM_ENDPOINT_INCLUSIVE",
    "T_UNIFORM_GRID_DENOMINATOR",
    "T_UNIFORM_GRID_SIZE",
    "T_UNIFORM_GRID_STEP",
]
