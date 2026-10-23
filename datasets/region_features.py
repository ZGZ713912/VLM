"""Verified full-frame spatial features; labels are deliberately not read here.

One video contains appearance (N,R,D), motion (N,R,18), global (N,D).
R is the ViT patch grid. Resizing the entire frame preserves its edges.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


REGION_VERSION = "full_frame_clip_tokens_flow_v1"
MOTION_DIM = 18


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def motion_descriptor(previous: np.ndarray | None, gray: np.ndarray,
                      grid: tuple[int, int]) -> np.ndarray:
    """Per-cell flow statistics, not anomaly labels. Input gray is uint8 (H,W)."""
    rows, cols = grid
    if gray.ndim != 2 or gray.dtype != np.uint8 or min(rows, cols) < 1:
        raise ValueError("Expected uint8 grayscale frame and positive grid")
    if previous is None:
        return np.zeros((rows * cols, MOTION_DIM), dtype=np.float32)
    if previous.shape != gray.shape:
        raise ValueError("Adjacent frames must have the same resolution")
    flow = cv2.calcOpticalFlowFarneback(previous, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    diff = np.abs(gray.astype(np.float32) - previous.astype(np.float32)) / 255
    result = []
    for row in range(rows):
        for col in range(cols):
            ys = slice(row * len(gray) // rows, (row + 1) * len(gray) // rows)
            xs = slice(col * gray.shape[1] // cols, (col + 1) * gray.shape[1] // cols)
            uv = flow[ys, xs].reshape(-1, 2)
            mag = np.linalg.norm(uv, axis=-1)
            angle = np.arctan2(uv[:, 1], uv[:, 0])
            hist, _ = np.histogram(angle, bins=8, range=(-np.pi, np.pi), weights=np.log1p(mag))
            stats = [*uv.mean(0), *uv.std(0), np.log1p(mag).mean(),
                     np.log1p(mag).std(), np.log1p(np.quantile(mag, .9)),
                     np.log1p(mag.max()), (mag > .5).mean()]
            result.append([*stats, *(hist / len(uv)), diff[ys, xs].mean()])
    return np.asarray(result, dtype=np.float32)


def load_region_cache(directory: str | Path, video_dir: str | Path,
                      expected_backbone: dict | None = None) -> tuple[dict[str, dict[str, Tensor]], dict]:
    """Fail closed on stale sources, truncated features, incompatible encoders."""
    directory, video_dir = Path(directory), Path(video_dir)
    meta = json.loads((directory / "manifest.json").read_text())
    if meta.get("version") != REGION_VERSION or meta.get("resize") != "full_frame_squash":
        raise ValueError("Incompatible region feature preprocessing")
    if expected_backbone is not None and meta.get("backbone") != expected_backbone:
        raise ValueError("Region cache backbone mismatch")
    sources = {p.stem: p for p in video_dir.glob("*.avi")}
    if not sources or set(sources) != set(meta.get("videos", {})):
        raise ValueError("Region cache and source video IDs differ")
    features = {}
    for vid, source in sorted(sources.items()):
        record = meta["videos"][vid]
        target = directory / f"{vid}.pt"
        if sha256(source) != record["source_sha256"] or sha256(target) != record["feature_sha256"]:
            raise ValueError(f"Region cache/source checksum mismatch: {vid}")
        item = torch.load(target, map_location="cpu", weights_only=True)
        n, r, d = record["num_frames"], meta["num_regions"], meta["appearance_dim"]
        shapes = {"appearance": (n, r, d), "motion": (n, r, MOTION_DIM), "global": (n, d)}
        for key, shape in shapes.items():
            value = item.get(key)
            if not isinstance(value, Tensor) or tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid region feature shape/values: {vid}/{key}")
        if not torch.equal(item["frame_indices"], torch.arange(n)):
            raise ValueError(f"Non-dense frame indices: {vid}")
        capture = cv2.VideoCapture(str(source))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if count != n:
            raise ValueError(f"Source frame count mismatch: {vid}")
        features[vid] = item
    return features, meta


class RegionClips(Dataset):
    """Whole-video partition is selected by caller before making windows."""

    def __init__(self, features: dict[str, dict[str, Tensor]], length: int, step: int) -> None:
        if not features or length < 2 or step < 1:
            raise ValueError("Require videos, length >= 2 and step >= 1")
        self.features, self.length = features, length
        self.indices = []
        for vid, item in features.items():
            n = len(item["appearance"])
            starts = list(range(0, max(1, n - length + 1), step))
            last = max(0, n - length)
            if last not in starts:
                starts.append(last)
            self.indices.extend((vid, start) for start in starts)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        vid, start = self.indices[index]
        item = self.features[vid]
        indices = torch.arange(start, start + self.length).clamp_max(len(item["appearance"]) - 1)
        return {key: item[key][indices].float() for key in ("appearance", "motion")}
