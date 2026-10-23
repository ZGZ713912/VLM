#!/usr/bin/env python3
"""Render scored video with training-calibrated decision, official GT and timeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import yaml

from utils.io import save_json
from datasets.region_features import sha256


def render(run: Path, video_id: str, output: Path, max_frames: int | None = None) -> dict:
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    report = json.loads((run / "official/metrics.json").read_text())
    variant = report["primary_variant"]
    scores = np.load(run / f"official/{variant}_scores.npz")[video_id]
    labels = np.load(run / "official/frame_labels.npz")[video_id]
    source = Path(cfg["data"]["root"]) / "testing_videos" / f"{video_id}.avi"
    if output.exists():
        raise FileExistsError(output)
    reader = cv2.VideoCapture(str(source))
    n = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = reader.get(cv2.CAP_PROP_FPS)
    h, w = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    if len(scores) != n or len(labels) != n:
        raise ValueError("Scores/labels/video frame counts differ")
    width, height = 800, round(h * 800 / w)
    panel = 145
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height + panel))
    if not writer.isOpened():
        raise RuntimeError("MP4 VideoWriter unavailable")
    threshold = report["calibration"]["thresholds"][variant]
    low, high = min(scores.min(), threshold), max(scores.max(), threshold)
    extent = max(high - low, 1e-5)
    xs = np.linspace(15, width - 15, n).astype(int)
    ys = (height + panel - 15 - (scores - low) / extent * 65).astype(int)
    points = np.stack([xs, ys], -1).reshape(-1, 1, 2)
    count = 0
    limit = min(n, max_frames) if max_frames else n
    while count < limit:
        ok, frame = reader.read()
        if not ok:
            raise ValueError("Video ended early")
        canvas = np.full((height + panel, width, 3), 22, np.uint8)
        canvas[:height] = cv2.resize(frame, (width, height))
        alert = scores[count] > threshold
        color = (70, 70, 245) if alert else (110, 200, 80)
        cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), color, 3)
        lines = [f"{cfg['dataset']} / {video_id} / frame {count} (zero-based)",
                 f"score {scores[count]:.3f} | normal-training threshold {threshold:.3f} | alert {int(alert)}",
                 f"OFFICIAL GT: {'ANOMALOUS' if labels[count] else 'NORMAL'} | text retrieval is unverified"]
        for index, line in enumerate(lines):
            cv2.putText(canvas, line, (15, height + 22 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX, .48,
                        (220, 220, 220), 1, cv2.LINE_AA)
        cv2.polylines(canvas, [points], False, (220, 180, 60), 1, cv2.LINE_AA)
        ty = round(height + panel - 15 - (threshold - low) / extent * 65)
        cv2.line(canvas, (15, ty), (width - 15, ty), (80, 120, 220), 1)
        cv2.line(canvas, (xs[count], height + 78), (xs[count], height + panel - 8), (245, 245, 245), 1)
        writer.write(canvas)
        count += 1
    writer.release()
    reader.release()
    verify = cv2.VideoCapture(str(output))
    recorded = int(verify.get(cv2.CAP_PROP_FRAME_COUNT))
    verify.release()
    if recorded != count:
        raise ValueError("Rendered video frame count differs")
    receipt = {"source": str(source), "source_sha256": sha256(source), "run": str(run),
               "variant": variant, "video_id": video_id, "source_frames": n, "rendered_frames": count,
               "complete": count == n, "output_sha256": sha256(output), "fps": fps,
               "official_label_source": report["label_source"], "threshold": threshold}
    save_json(receipt, output.with_suffix(".json"))
    return receipt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--video-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-frames", type=int)
    a = p.parse_args()
    print(json.dumps(render(a.run, a.video_id, a.output, a.max_frames)), flush=True)


if __name__ == "__main__":
    main()
