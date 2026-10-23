#!/usr/bin/env python3
"""Label-free inference: source video -> verified spatial features -> frozen VLM."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from datasets.region_features import sha256
from models.region_detector import load_region_detector
from tools.extract_region_features import extract
from utils.io import save_json
from utils.visualization import plot_temporal_heatmap


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=24)
    a = p.parse_args()
    if not a.video.is_file() or a.out_dir.exists():
        raise ValueError("Require existing source video and a new output directory")
    detector = load_region_detector(a.checkpoint, a.device)
    cfg = detector.config
    a.out_dir.mkdir(parents=True)
    # Single-video directory adapter; source remains untouched and no labels are opened.
    with TemporaryDirectory() as tmp:
        source = Path(tmp) / "input.avi"
        source.symlink_to(a.video.resolve())
        extract(Path(tmp), a.out_dir / "features", cfg["backbone"]["name"],
                cfg["backbone"]["pretrained"], a.batch_size, a.device)
    item = torch.load(a.out_dir / "features/input.pt", weights_only=True)
    output = detector(item["appearance"].unsqueeze(0), item["motion"].unsqueeze(0))
    scores = output["frame_score"][0].cpu().numpy()
    alert = output["alert"][0].cpu().numpy()
    np.savez_compressed(a.out_dir / "prediction.npz", frame_score=scores, alert=alert,
                        components=output["components"][0].cpu().numpy(), frame_indices=item["frame_indices"].numpy())
    top = np.argsort(-scores, kind="stable")[:10].tolist()
    threshold = detector.calibration["thresholds"][cfg["eval"]["primary_variant"]]
    save_json({"video": str(a.video), "video_sha256": sha256(a.video),
               "checkpoint_sha256": sha256(a.checkpoint), "num_frames": len(scores),
               "top_abnormal_frames_zero_based": top, "num_alert_frames": int(alert.sum()),
               "anomaly_score": float(scores.max()), "training_threshold": threshold,
               "labels_read": False, "explanation": f"第 {top[0]} 帧相对正常训练模式的综合偏离分数最高。"
                                                         "分数并非异常概率，不确认具体事件类别。"}, a.out_dir / "prediction.json")
    plot_temporal_heatmap(scores, title="Label-free regional VLM prediction",
                          save_path=a.out_dir / "timeline.png", threshold=threshold)
    print(f"Predicted {len(scores)} frames; {int(alert.sum())} exceed normal-training threshold", flush=True)


if __name__ == "__main__":
    main()
