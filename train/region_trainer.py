"""Normal-only temporal loss + explicitly synthetic BCE + visual/text alignment."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def synthetic_corruption(appearance: Tensor, motion: Tensor, text: Tensor,
                         polarity: Tensor) -> tuple[Tensor, Tensor]:
    """Create out-of-distribution feature perturbations, never real GT anomalies.

    A random region per frame is mixed with an abnormal text direction and motion
    is amplified there. Perturbation labels mean 'synthetic', not event truth.
    """
    b, t, r, _ = appearance.shape
    abnormal = text[polarity > 0]
    if not len(abnormal):
        raise ValueError("Synthetic exposure requires explicit abnormal prompts")
    region = torch.randint(r, (b, t, 1), device=appearance.device)
    mask = torch.zeros((b, t, r, 1), device=appearance.device).scatter_(2, region.unsqueeze(-1), 1)
    direction = abnormal[torch.randint(len(abnormal), (b, t), device=text.device)].unsqueeze(2)
    strength = torch.empty((b, 1, 1, 1), device=text.device).uniform_(.15, .45)
    corrupted = F.normalize(appearance + mask * strength * direction, dim=-1)
    synthetic_motion = motion + mask * (2 * motion + .2 * torch.randn_like(motion))
    return corrupted, synthetic_motion


def region_loss(model, appearance: Tensor, motion: Tensor, text: Tensor,
                polarity: Tensor, weights: dict, *, synthetic: bool = True) -> dict[str, Tensor]:
    output = model(appearance, motion, text)
    # Predict t+1 using only <=t; the target is never an input to that GRU state.
    appearance_loss = (1 - F.cosine_similarity(output["predicted_appearance"][:, :-1],
                                               appearance[:, 1:], dim=-1)).mean()
    motion_loss = F.smooth_l1_loss(output["predicted_motion"][:, :-1],
                                  output["standardized_motion"][:, 1:])
    logits = output["frame_logits"]
    bce = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    if synthetic:
        fake_a, fake_m = synthetic_corruption(appearance, motion, text, polarity)
        fake_logits = model(fake_a, fake_m, text)["frame_logits"]
        bce = .5 * (bce + F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits)))
    similarities = output["embedding"] @ F.normalize(text, dim=-1).T / .07
    normal = polarity < 0
    if not normal.any() or not (polarity > 0).any():
        raise ValueError("Both prompt polarities are required")
    targets = normal.to(similarities.dtype) / normal.sum()
    contrastive = -(similarities.log_softmax(-1) * targets).sum(-1).mean()
    loss = (weights["appearance"] * appearance_loss + weights["motion"] * motion_loss
            + weights["bce"] * bce + weights["contrastive"] * contrastive)
    return {"loss": loss, "appearance": appearance_loss, "motion": motion_loss,
            "bce": bce, "contrastive": contrastive,
            "prediction": weights["appearance"] * appearance_loss + weights["motion"] * motion_loss}


def rng_state(generator: torch.Generator) -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}


def restore_rng(state: dict, generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])
