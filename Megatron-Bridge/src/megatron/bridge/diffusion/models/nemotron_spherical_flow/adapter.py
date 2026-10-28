"""Exact full-vocabulary spherical codebook adapter."""

import torch
from torch import nn

from .geometry import normalize


class SphericalCodebookAdapter(nn.Module):
    """Map spherical states to input embeddings without materializing [N,V,D]."""

    def __init__(
        self,
        embedding: nn.Embedding,
        chunk_size: int = 4096,
        freeze_embedding: bool = True,
        beta_init: float = 16.0,
    ):
        super().__init__()
        if not torch.isfinite(torch.tensor(beta_init)) or beta_init <= 0:
            raise ValueError("beta_init must be a finite positive number")
        self.embedding = embedding
        if freeze_embedding:
            self.embedding.weight.requires_grad_(False)
        # Geometry must remain independent from the native input embedding.
        # Keep the exact BF16 snapshot as a frozen buffer; adapter updates must
        # never mutate or alias the checkpoint compatibility path.
        self.register_buffer(
            "geometry_embedding",
            embedding.weight.detach().clone(),
            # Geometry is model state, not a cache.  In particular, a resumed
            # run must use the same frozen snapshot even if the source input
            # embedding module is upgraded or reloaded independently.
            persistent=True,
        )
        self.chunk_size = chunk_size
        self.log_beta = nn.Parameter(
            torch.tensor(beta_init, dtype=embedding.weight.dtype).log()
        )
        self.bias = nn.Parameter(torch.zeros(embedding.num_embeddings, dtype=embedding.weight.dtype))
        # A few reserved rows can be zero; keep them as zero codewords rather than
        # silently changing the checkpoint geometry or failing model construction.
        weights = self.geometry_embedding.float()
        self.register_buffer(
            "codebook",
            (weights / weights.norm(dim=-1, keepdim=True).clamp_min(1e-6)).to(embedding.weight.dtype),
            persistent=True,
        )

    @property
    def beta(self):
        return self.log_beta.float().exp()

    def forward(
        self,
        state: torch.Tensor,
        return_stats: bool = False,
        source_ids: torch.Tensor | None = None,
    ):
        flat = normalize(state.reshape(-1, state.shape[-1]))
        if source_ids is not None:
            source_ids = source_ids.reshape(-1).to(device=flat.device, dtype=torch.long)
            if source_ids.numel() != flat.shape[0]:
                raise ValueError("source_ids must match the non-feature dimensions of state")
        emb = self.geometry_embedding
        code = self.codebook.to(device=flat.device)
        bias = self.bias.to(device=flat.device)
        max_logit = torch.full((flat.shape[0], 1), -torch.inf, device=flat.device, dtype=torch.float32)
        denom = torch.zeros_like(max_logit)
        weighted = torch.zeros((flat.shape[0], emb.shape[1]), device=flat.device, dtype=torch.float32)
        logit_first_moment = torch.zeros((flat.shape[0],), device=flat.device, dtype=torch.float32)
        top_logit = torch.full((flat.shape[0],), -torch.inf, device=flat.device, dtype=torch.float32)
        top_id = torch.zeros((flat.shape[0],), device=flat.device, dtype=torch.long)
        for start in range(0, emb.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, emb.shape[0])
            logits = self.beta * (flat.float() @ code[start:end].float().T) + bias[start:end].float()
            chunk_max = logits.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(max_logit, chunk_max)
            old_scale = torch.exp(max_logit - new_max)
            probs = torch.exp(logits - new_max)
            denom = denom * old_scale + probs.sum(dim=-1, keepdim=True)
            weighted = weighted * old_scale + probs @ emb[start:end].float()
            logit_first_moment = logit_first_moment * old_scale.squeeze(-1) + (probs * logits).sum(dim=-1)
            chunk_top_logit, chunk_top_offset = logits.max(dim=-1)
            replace_top = chunk_top_logit > top_logit
            top_logit = torch.where(replace_top, chunk_top_logit, top_logit)
            top_id = torch.where(replace_top, chunk_top_offset + start, top_id)
            max_logit = new_max
        out = weighted / denom.clamp_min(1e-12)
        out = out.reshape(*state.shape[:-1], state.shape[-1]).to(dtype=state.dtype)
        if not return_stats:
            return out
        log_z = max_logit.squeeze(-1) + torch.log(denom.squeeze(-1).clamp_min(1e-12))
        expected_logit = logit_first_moment / denom.squeeze(-1).clamp_min(1e-12)
        entropy = (log_z - expected_logit).reshape(state.shape[:-1])
        stats = {
            "adapter_entropy": entropy,
            "adapter_top_id": top_id.reshape(state.shape[:-1]),
            "beta": self.beta.detach(),
        }
        if source_ids is not None:
            source_logits = self.beta * (flat.float() * code[source_ids].float()).sum(dim=-1) + bias[source_ids].float()
            stats["adapter_source_weight"] = torch.exp(source_logits - log_z).reshape(state.shape[:-1])
            stats["adapter_source_top1"] = top_id.eq(source_ids).reshape(state.shape[:-1])
        return out, stats
