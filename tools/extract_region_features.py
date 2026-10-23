#!/usr/bin/env python3
"""Extract full-frame CLIP region tokens and local optical flow without labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from datasets.region_features import REGION_VERSION, motion_descriptor, sha256
from models.backbone import CLIPBackbone
from utils.io import save_json
from utils.provenance import runtime_info
from utils.reproducibility import set_seed


@torch.inference_mode()
def extract(video_dir: Path, out: Path, model_name: str, pretrained: str,
            batch_size: int, device: str) -> None:
    videos = sorted(video_dir.glob("*.avi"))
    if not videos or batch_size < 1:
        raise ValueError("Require source AVI videos and a positive batch size")
    backbone_id = {"provider": "open_clip_patch", "name": model_name, "pretrained": pretrained}
    if (out / "manifest.json").exists():
        from datasets.region_features import load_region_cache
        load_region_cache(out, video_dir, backbone_id)
        print(f"Verified completed cache {out}", flush=True)
        return
    set_seed(42)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    backbone = CLIPBackbone(model_name, pretrained).to(device).eval()
    visual = backbone.vision_encoder
    if not hasattr(visual, "output_tokens") or visual.attn_pool is not None:
        raise ValueError("Use an OpenCLIP ViT with projected patch tokens for this baseline")
    visual.output_tokens = True
    size = backbone.preprocess_cfg["size"]
    size = (size, size) if isinstance(size, int) else tuple(size)
    grid = tuple(visual.grid_size)
    out.mkdir(parents=True, exist_ok=True)
    identity = {"version": REGION_VERSION, "backbone": backbone_id, "resize": "full_frame_squash",
                "size": list(size), "grid": list(grid), "num_regions": int(np.prod(grid)),
                "appearance_dim": backbone.dim, "precision": "float32_compute_float16_cache",
                "motion": "Farneback_0.5_3_15_3_5_1.2_0_grid18", "labels_used": False}
    # A partial cache can resume only under exactly the same extraction contract.
    partial = out / "extraction.json"
    if partial.exists() and json.loads(partial.read_text()) != identity:
        raise ValueError("Partial cache extraction configuration differs")
    save_json(identity, partial)
    records = {}
    for video in videos:
        target, receipt = out / f"{video.stem}.pt", out / f"{video.stem}.json"
        source_hash = sha256(video)
        if target.exists() and receipt.exists():
            record = json.loads(receipt.read_text())
            if record["source_sha256"] == source_hash and record["feature_sha256"] == sha256(target):
                records[video.stem] = record
                print(f"Resume verified {video.stem}", flush=True)
                continue
            raise ValueError(f"Partial feature cache/source changed: {video}")
        capture = cv2.VideoCapture(str(video))
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        previous, frames, appearances, globals_, motions = None, [], [], [], []

        def encode_batch() -> None:
            if not frames:
                return
            rgb = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().to(device) / 255
            rgb = F.interpolate(rgb, size=size, mode="bicubic", align_corners=False, antialias=True)
            mean = torch.tensor(backbone.preprocess_cfg["mean"], device=device).view(1, 3, 1, 1)
            std = torch.tensor(backbone.preprocess_cfg["std"], device=device).view(1, 3, 1, 1)
            pooled, tokens = visual((rgb - mean) / std)
            # OpenCLIP ln_post is already applied unless final_ln_after_pool=True.
            if visual.final_ln_after_pool:
                tokens = visual.ln_post(tokens)
            if visual.proj is not None:
                tokens = tokens @ visual.proj
            appearances.append(F.normalize(tokens, dim=-1).half().cpu())
            globals_.append(F.normalize(pooled, dim=-1).half().cpu())
            frames.clear()

        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            # Motion and patch indices refer to the same full-frame spatial grid.
            gray = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (size[1], size[0]))
            motions.append(torch.from_numpy(motion_descriptor(previous, gray, grid)))
            previous = gray
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if len(frames) == batch_size:
                encode_batch()
        encode_batch()
        capture.release()
        if len(motions) != expected or not motions:
            raise ValueError(f"Decoded frame count mismatch: {video}")
        item = {"appearance": torch.cat(appearances), "global": torch.cat(globals_),
                "motion": torch.stack(motions), "frame_indices": torch.arange(expected)}
        torch.save(item, target.with_suffix(".tmp"))
        target.with_suffix(".tmp").replace(target)
        record = {"source_sha256": source_hash, "feature_sha256": sha256(target), "num_frames": expected}
        save_json(record, receipt)
        records[video.stem] = record
        print(f"{video.stem}: {expected} frames -> {tuple(item['appearance'].shape)}", flush=True)
    save_json({**identity, "videos": records, "environment": runtime_info()}, out / "manifest.json")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--pretrained", default="laion2b_s34b_b79k")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    extract(a.video_dir, a.out, a.model, a.pretrained, a.batch_size, a.device)


if __name__ == "__main__":
    main()
