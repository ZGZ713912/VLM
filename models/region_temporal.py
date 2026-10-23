"""Frozen-backbone regional VLM with causal temporal learning and normal memory.

Inputs: appearance (B,T,R,D), motion (B,T,R,F), external text (K,D).
The predictor sees frames <= t to predict frame t+1; test labels never enter it.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RegionTemporalModel(nn.Module):
    def __init__(self, appearance_dim: int, motion_dim: int, hidden_dim: int = 128,
                 fusion: str = "concat", num_heads: int = 4, top_regions: int = 3) -> None:
        super().__init__()
        if fusion not in ("concat", "crossattn") or hidden_dim % num_heads or top_regions < 1:
            raise ValueError("Invalid fusion, head count or region reduction")
        self.appearance_dim, self.motion_dim = appearance_dim, motion_dim
        self.fusion, self.top_regions = fusion, top_regions
        self.visual_proj = nn.Linear(appearance_dim, hidden_dim)
        self.text_proj = nn.Linear(appearance_dim, hidden_dim)
        self.motion_proj = nn.Linear(motion_dim, hidden_dim)
        self.cross = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0)
        self.concat = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.appearance_predictor = nn.Linear(hidden_dim, appearance_dim)
        self.motion_predictor = nn.Linear(hidden_dim, motion_dim)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
                                        nn.Linear(hidden_dim // 2, 1))
        self.alignment = nn.Linear(hidden_dim, appearance_dim)
        self.register_buffer("motion_mean", torch.zeros(motion_dim))
        self.register_buffer("motion_std", torch.ones(motion_dim))
        # Start with a persistence predictor; learn changes rather than background.
        nn.init.zeros_(self.appearance_predictor.weight)
        nn.init.zeros_(self.appearance_predictor.bias)

    def forward(self, appearance: Tensor, motion: Tensor, text: Tensor) -> dict[str, Tensor | None]:
        if appearance.ndim != 4 or appearance.shape[-1] != self.appearance_dim:
            raise ValueError("Appearance must be (B,T,R,D)")
        if motion.shape != (*appearance.shape[:3], self.motion_dim):
            raise ValueError("Motion must be aligned (B,T,R,F)")
        if text.ndim != 2 or text.shape[-1] != self.appearance_dim or len(text) < 2:
            raise ValueError("Text must contain normal and abnormal prompts (K,D)")
        b, t, r, _ = appearance.shape
        standardized = ((motion - self.motion_mean) / self.motion_std).clamp(-20, 20)
        visual = self.visual_proj(appearance) + self.motion_proj(standardized)
        queries = visual.reshape(b, t * r, -1)
        language = self.text_proj(text).unsqueeze(0).expand(b, -1, -1)
        if self.fusion == "crossattn":
            context, _ = self.cross(queries, language, language, need_weights=False)
            fused = self.norm(queries + context)
        else:
            attention = (F.normalize(appearance, dim=-1) @ F.normalize(text, dim=-1).T / .07).softmax(-1)
            context = attention @ self.text_proj(text)
            fused = self.norm(self.concat(torch.cat([visual, context], -1)).reshape(b, t * r, -1))
        fused = fused.reshape(b, t, r, -1)
        # Each spatial cell is its own time series; GRU is strictly unidirectional.
        sequence = fused.permute(0, 2, 1, 3).reshape(b * r, t, -1)
        recurrent, _ = self.gru(sequence)
        hidden = recurrent.reshape(b, r, t, -1).permute(0, 2, 1, 3)
        prediction = F.normalize(appearance + self.appearance_predictor(hidden), dim=-1)
        motion_prediction = self.motion_predictor(hidden)
        region_logits = self.classifier(fused).squeeze(-1)
        k = min(r, self.top_regions)
        logits = region_logits.topk(k, dim=-1).values.mean(-1)
        embedding = F.normalize(self.alignment(hidden.mean((1, 2))), dim=-1)
        return {"anomaly_score": logits.sigmoid().amax(1), "frame_score": logits.sigmoid(),
                "embedding": embedding, "explanation": None, "frame_logits": logits,
                "predicted_appearance": prediction, "predicted_motion": motion_prediction,
                "standardized_motion": standardized, "region_logits": region_logits}


class RegionMemory(nn.Module):
    """Location-specific kNN on normal appearance and motion; no anomaly labels."""
    def __init__(self, appearance: Tensor, motion: Tensor, k: int = 3, query_chunk: int = 64) -> None:
        super().__init__()
        if appearance.ndim != 3 or motion.ndim != 3 or appearance.shape[:2] != motion.shape[:2]:
            raise ValueError("Banks must be (R,M,D)/(R,M,F)")
        if not 1 <= k <= appearance.shape[1] or query_chunk < 1:
            raise ValueError("Invalid k or query chunk")
        self.k, self.query_chunk = k, query_chunk
        self.register_buffer("appearance_bank", F.normalize(appearance.float(), dim=-1))
        self.register_buffer("motion_bank", motion.float())
        self.register_buffer("motion_mean", torch.zeros(motion.shape[-1]))
        self.register_buffer("motion_std", torch.ones(motion.shape[-1]))

    @classmethod
    def fit(cls, videos: dict[str, dict[str, Tensor]], bank_frames: int, seed: int, k: int = 3):
        if bank_frames < k or not videos:
            raise ValueError("Require normal videos and enough memory frames")
        appearance = torch.cat([v["appearance"].float() for v in videos.values()])
        motion = torch.cat([v["motion"].float() for v in videos.values()])
        indices = torch.randperm(len(appearance), generator=torch.Generator().manual_seed(seed))[:bank_frames]
        mean = motion.mean((0, 1))
        std = motion.std((0, 1)).clamp_min(.05)
        standardized = ((motion - mean) / std).clamp(-20, 20)
        model = cls(appearance[indices].permute(1, 0, 2), standardized[indices].permute(1, 0, 2), k)
        model.motion_mean.copy_(mean)
        model.motion_std.copy_(std)
        return model, indices

    @torch.no_grad()
    def forward(self, appearance: Tensor, motion: Tensor) -> dict:
        if appearance.ndim != 4 or motion.shape[:3] != appearance.shape[:3]:
            raise ValueError("Require aligned (B,T,R,D)/(B,T,R,F)")
        b, t, r, d = appearance.shape
        if r != len(self.appearance_bank) or d != self.appearance_bank.shape[-1]:
            raise ValueError("Memory/query region or dimension mismatch")
        appearance = F.normalize(appearance.reshape(-1, r, d).float(), dim=-1)
        motion = ((motion.reshape(-1, r, motion.shape[-1]) - self.motion_mean) / self.motion_std).clamp(-20, 20)
        app_scores, motion_scores = [], []
        bank_norm = self.motion_bank.square().sum(-1).unsqueeze(0)
        for start in range(0, len(appearance), self.query_chunk):
            a, m = appearance[start:start + self.query_chunk], motion[start:start + self.query_chunk]
            cosine = 1 - torch.einsum("nrd,rmd->nrm", a, self.appearance_bank)
            distance = (m.square().sum(-1, keepdim=True) + bank_norm
                        - 2 * torch.einsum("nrf,rmf->nrm", m, self.motion_bank)).clamp_min(0)
            app_scores.append(cosine.topk(self.k, largest=False).values.mean(-1))
            motion_scores.append(distance.topk(self.k, largest=False).values.mean(-1).sqrt())
        app = torch.cat(app_scores).view(b, t, r)
        mot = torch.cat(motion_scores).view(b, t, r)
        score = app.topk(min(3, r), dim=-1).values.mean(-1)
        return {"anomaly_score": score.amax(1), "frame_score": score,
                "embedding": appearance.view(b, t, r, d).mean((1, 2)), "explanation": None,
                "appearance": app, "motion": mot}
