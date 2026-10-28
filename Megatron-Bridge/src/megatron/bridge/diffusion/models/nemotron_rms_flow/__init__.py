# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RMS-normalized embedding continuous denoising for Nemotron."""

from .conditioning import GammaConditioner
from .geometry import RMSGeometry, compute_global_rms, linear_path, terminal_noise
from .identity import cosine_identity_metrics, endpoint_normalized_metric
from .inference import euclidean_reverse_step, rms_block_sample
from .metadata import load_checkpoint
from .optimizer import ShardedBF16AdamW, ShardedFP32AdamW
from .prompting import render_user_prompt
from .provider import NemotronRMSFlowProvider, RMSFlowConfig
from .scheduler import gamma_from_t, t_from_gamma
from .statistics import StatisticsManifest
from .step import RMSContinuousDenoiser


__all__ = [
    "GammaConditioner",
    "RMSContinuousDenoiser",
    "RMSGeometry",
    "compute_global_rms",
    "gamma_from_t",
    "linear_path",
    "load_checkpoint",
    "rms_block_sample",
    "t_from_gamma",
    "terminal_noise",
    "cosine_identity_metrics",
    "endpoint_normalized_metric",
    "euclidean_reverse_step",
    "StatisticsManifest",
    "NemotronRMSFlowProvider",
    "RMSFlowConfig",
    "ShardedFP32AdamW",
    "ShardedBF16AdamW",
    "render_user_prompt",
]
