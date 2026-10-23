"""Deployable frozen composite matching the reported regional model scores."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from eval.region_evaluation import component_scores, fuse
from models.region_temporal import RegionMemory, RegionTemporalModel


class RegionDetector(nn.Module):
    """Inference on cached full-frame features, with no labels or calibration fitting.

    Model.forward returns the predeclared full composite, not the synthetic-only
    auxiliary classifier. A forward call is one complete video per batch item;
    start a new call for a new video (recurrent/smoothing state resets).
    """
    def __init__(self, state: dict, device: str = "cpu") -> None:
        super().__init__()
        self.config = state["config"]
        self.prompts = state["prompts"]
        self.calibration = state["calibration"]
        m = state["memory"]
        d, f = m["appearance_bank"].shape[-1], m["motion_bank"].shape[-1]
        self.temporal = RegionTemporalModel(d, f, **self.config["model"])
        self.temporal.load_state_dict(state["model"])
        self.memory = RegionMemory(m["appearance_bank"], m["motion_bank"], self.config["memory"]["k"])
        self.memory.load_state_dict(m)
        self.register_buffer("text", state["text"].float())
        self.register_buffer("polarity", state["polarity"].long())
        self.to(device)
        self.eval()

    @torch.inference_mode()
    def forward(self, appearance: Tensor, motion: Tensor) -> dict:
        if appearance.ndim != 4 or motion.shape[:3] != appearance.shape[:3]:
            raise ValueError("Require full-video features (B,T,R,D)/(B,T,R,F)")
        cfg = self.config["eval"]
        variant = cfg["primary_variant"]
        scores, components = [], []
        self.temporal.eval()
        for app, mot in zip(appearance, motion):
            raw, _ = component_scores(self.temporal, self.memory, {"appearance": app, "motion": mot},
                                       self.text, self.polarity, self.config["model"]["top_regions"])
            scores.append(fuse(raw, self.calibration, cfg["variants"][variant], cfg["smooth_window"]))
            components.append(raw)
        score = torch.from_numpy(np.stack(scores)).to(self.text.device)
        threshold = self.calibration["thresholds"][variant]
        return {"anomaly_score": score.amax(1), "frame_score": score,
                "embedding": F.normalize(appearance.float().mean((1, 2)), dim=-1).to(self.text.device),
                "explanation": None, "alert": score > threshold,
                "components": torch.from_numpy(np.stack(components)).to(self.text.device)}


def load_region_detector(checkpoint: str | Path, device: str = "cpu") -> RegionDetector:
    # Checkpoints contain only this project's trusted training artifacts.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("version") != 1 or "calibration" not in state or "memory" not in state:
        raise ValueError("Require finalized regional detector checkpoint")
    return RegionDetector(state, device)
