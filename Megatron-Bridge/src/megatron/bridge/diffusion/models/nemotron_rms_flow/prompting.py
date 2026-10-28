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

"""Prompt rendering shared by training diagnostics and standalone sampling."""

from __future__ import annotations

import torch


def _flat_token_ids(value) -> list[int]:
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict):
        value = value["input_ids"]
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return list(map(int, value))


def render_user_prompt(tokenizer, prompt: str, *, device=None) -> torch.Tensor:
    """Render a user message with the same generation prefix used by conversion."""
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    ids = torch.tensor([_flat_token_ids(encoded)], dtype=torch.long)
    return ids.to(device=device) if device is not None else ids


__all__ = ["render_user_prompt"]
