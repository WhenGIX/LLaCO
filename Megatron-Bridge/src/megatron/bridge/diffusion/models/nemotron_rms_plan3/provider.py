"""Plan 3 model configuration and narrow assembly point."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from .noise import NoiseMode
from .step import Plan3ContinuousDenoiser

@dataclass(frozen=True)
class Plan3RMSConfig:
    block_size: int = 32
    condition_dim: int = 1024
    posterior_mode: str = "full"
    posterior_k: int | None = None
    solver_steps_per_block: int = 32
    solver_update_count: int = 32
    solver_nfe: int = 33
    input_embedding_frozen: bool = True
    tensor_parallelism: int = 1
    pipeline_parallelism: int = 1
    def __post_init__(self):
        if self.block_size != 32 or self.solver_steps_per_block != 32 or self.solver_update_count != 32 or self.solver_nfe != 33:
            raise ValueError("Plan 3 solver is fixed at 32 updates and 33 NFE")
        if self.posterior_mode not in {"full", "topk"} or (self.posterior_mode == "topk" and (self.posterior_k is None or self.posterior_k <= 0)):
            raise ValueError("posterior must be full or topk with positive k")
        if not self.input_embedding_frozen or self.tensor_parallelism != 1 or self.pipeline_parallelism != 1:
            raise ValueError("Plan 3 requires frozen embeddings and TP=PP=1")

class Plan3RMSModel(Plan3ContinuousDenoiser):
    def __init__(self, base, mask_token_id, schedule, mode, *, condition_dim=1024):
        super().__init__(base, mask_token_id, training_mode=mode, condition_dim=condition_dim, schedule=schedule)

def freeze_input_embedding(model):
    model.get_input_embeddings().weight.requires_grad_(False)
    return model.get_input_embeddings()

__all__ = ["Plan3RMSConfig", "Plan3RMSModel", "freeze_input_embedding"]
