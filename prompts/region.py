"""Explicit normal/abnormal scene prompt sets, kept outside model forward."""
from pathlib import Path

import torch
import yaml


def region_prompts(path: str | Path, dataset: str, prompt_type: str) -> tuple[list[str], torch.Tensor]:
    if prompt_type not in ("label", "scene", "contrast"):
        raise ValueError("Prompt type must be label, scene or contrast")
    spec = yaml.safe_load(Path(path).read_text())[dataset][prompt_type]
    normal, abnormal = spec["normal"], spec["abnormal"]
    if (not normal or not abnormal or set(normal) & set(abnormal)
            or not all(isinstance(p, str) and p.strip() for p in normal + abnormal)):
        raise ValueError("Require distinct nonempty explicitly polarized prompts")
    return normal + abnormal, torch.tensor([-1] * len(normal) + [1] * len(abnormal))
