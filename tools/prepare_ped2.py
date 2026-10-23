#!/usr/bin/env python3
"""Acquire official UCSD Ped2, validate masks, produce lossless videos + labels."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch

from datasets.region_features import sha256
from datasets.cache import validate_official_cache
from datasets.labels import load_frame_labels
from utils.io import save_json


UCSD_URL = "http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz"


def safe_extract_ped2(archive: Path, output: Path) -> None:
    """Extract only regular Ped2 files; never honor archive links/absolute paths."""
    prefix = "UCSD_Anomaly_Dataset.v1p2/UCSDped2/"
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            if not member.name.startswith(prefix) or not member.isfile():
                continue
            relative = Path(member.name[len(prefix):])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe dataset archive path")
            if relative.name.startswith("._"):
                continue
            target = output / "raw" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            data = stream.extractfile(member)
            if data is None:
                raise ValueError("Unreadable archive member")
            target.write_bytes(data.read())


def parse_matlab_intervals(text: str) -> list[list[tuple[int, int]]]:
    """Read numeric gt_frame ranges; never execute downloaded MATLAB code."""
    entries = re.findall(r"\.gt_frame\s*=\s*\[([^\]]+)\]", text)
    result = []
    for entry in entries:
        ranges = re.findall(r"(\d+)\s*:\s*(\d+)", entry)
        if not ranges or re.sub(r"\d+\s*:\s*\d+|[\s,;]", "", entry):
            raise ValueError("Unsupported MATLAB annotation format")
        result.append([(int(start), int(end)) for start, end in ranges])
    if not result:
        raise ValueError("No MATLAB ground truth intervals")
    return result


def verify_prepared(archive: Path, out: Path, label_out: Path) -> dict:
    import json
    preparation = json.loads((out / "preparation.json").read_text())
    manifest = validate_official_cache(label_out / "testing")
    if preparation["source"]["sha256"] != sha256(archive) or manifest["source"] != preparation["source"]:
        raise ValueError("Prepared Ped2 archive identity changed")
    if set(manifest["videos"]) != set(preparation["testing"]):
        raise ValueError("Prepared Ped2 label video IDs changed")
    for split in ("training", "testing"):
        expected = preparation[split]
        actual = {p.stem for p in (out / f"{split}_videos").glob("*.avi")}
        if actual != set(expected):
            raise ValueError("Prepared Ped2 video IDs changed")
        for vid, record in expected.items():
            if sha256(out / f"{split}_videos" / f"{vid}.avi") != record["video_sha256"]:
                raise ValueError(f"Prepared Ped2 video checksum changed: {vid}")
            if split == "testing":
                label_path = label_out / "testing" / f"{vid}.pt"
                if (sha256(label_path) != record["label_sha256"]
                        or manifest["videos"][vid] != record
                        or int(load_frame_labels(label_path, record["num_frames"]).sum()) != record["num_anomaly_frames"]):
                    raise ValueError(f"Prepared Ped2 labels changed: {vid}")
    return manifest


def prepare(archive: Path, out: Path, label_out: Path) -> None:
    if (out / "preparation.json").exists() or (label_out / "testing/manifest.json").exists():
        manifest = verify_prepared(archive, out, label_out)
        print(f"Verified existing Ped2: {manifest['num_frames']} testing frames; no files overwritten", flush=True)
        return
    source_hash = sha256(archive)
    safe_extract_ped2(archive, out)
    intervals = parse_matlab_intervals((out / "raw/Test/UCSDped2.m").read_text())
    source = {"url": UCSD_URL, "archive": str(archive), "sha256": source_hash,
              "publisher": "UCSD Statistical Visual Computing Laboratory"}
    records, training_records = {}, {}
    for split, source_split in (("training", "Train"), ("testing", "Test")):
        clips = sorted(p for p in (out / "raw" / source_split).glob(f"{source_split}*")
                       if p.is_dir() and not p.name.endswith("_gt"))
        expected = 16 if split == "training" else 12
        if len(clips) != expected:
            raise ValueError(f"Expected {expected} {split} clips, got {len(clips)}")
        video_dir = out / f"{split}_videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        for clip_index, clip in enumerate(clips):
            paths = sorted(clip.glob("*.tif"))
            if [int(p.stem) for p in paths] != list(range(1, len(paths) + 1)):
                raise ValueError(f"Missing or non-contiguous TIFF frames: {clip}")
            first = cv2.imread(str(paths[0]))
            if first is None:
                raise ValueError(f"Unreadable frame: {paths[0]}")
            h, w = first.shape[:2]
            video = video_dir / f"{clip.name}.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 10, (w, h))
            if not writer.isOpened():
                raise RuntimeError("Lossless FFV1 VideoWriter unavailable")
            frame_hashes, mask_hashes, labels = [], [], []
            mask_by_index = {}
            if split == "testing":
                for mask_file in (clip.parent / f"{clip.name}_gt").glob("*.bmp"):
                    match = re.fullmatch(r"(?:frame)?(\d+)", mask_file.stem)
                    if match is None or int(match[1]) in mask_by_index:
                        raise ValueError("Invalid or duplicate official mask index")
                    mask_by_index[int(match[1])] = mask_file
                if set(mask_by_index) != set(range(1, len(paths) + 1)):
                    raise ValueError("Mask/frame indices differ")
            for path in paths:
                frame = cv2.imread(str(path))
                if frame is None or frame.shape != first.shape:
                    raise ValueError(f"Invalid TIFF frame: {path}")
                writer.write(frame)
                frame_hashes.append(sha256(path))
                if split == "testing":
                    mask_path = mask_by_index[int(path.stem)]
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is None or mask.shape != (h, w) or not set(np.unique(mask)).issubset({0, 1, 255}):
                        raise ValueError(f"Invalid official mask: {mask_path}")
                    mask_hashes.append(sha256(mask_path))
                    labels.append(int(np.any(mask)))
            writer.release()
            # Confirm lossless decoding and exact frame indexing, not only metadata.
            reader = cv2.VideoCapture(str(video))
            for path in paths:
                ok, decoded = reader.read()
                if not ok or not np.array_equal(decoded, cv2.imread(str(path))):
                    raise ValueError(f"Lossless video roundtrip failed: {video}")
            if reader.read()[0]:
                raise ValueError("Unexpected extra video frames")
            reader.release()
            record = {"num_frames": len(paths), "height": h, "width": w, "mask_shape": [h, w],
                      "source_frame_sha256": frame_hashes, "video_sha256": sha256(video)}
            if split == "testing":
                mask_files = list((clip.parent / f"{clip.name}_gt").glob("*.bmp"))
                if len(mask_files) != len(paths):
                    raise ValueError("Mask/frame counts differ")
                label = torch.tensor(labels, dtype=torch.long)
                label_dir = label_out / "testing"
                label_dir.mkdir(parents=True, exist_ok=True)
                label_path = label_dir / f"{clip.name}.pt"
                torch.save(label, label_path)
                interval_label = np.zeros(len(paths), dtype=np.int64)
                for start, end in intervals[clip_index]:
                    if not 1 <= start <= end <= len(paths):
                        raise ValueError("Official interval outside video bounds")
                    interval_label[start - 1:end] = 1
                mismatch = np.flatnonzero(interval_label != labels).tolist()
                record.update(num_anomaly_frames=int(label.sum()), label_file=label_path.name,
                              label_sha256=sha256(label_path), source_mask_sha256=mask_hashes,
                              matlab_interval_disagreement_zero_based=mismatch)
                records[clip.name] = record
            else:
                training_records[clip.name] = record
            print(f"{split}/{clip.name}: {len(paths)} frames", flush=True)
    manifest = {"version": 1, "label_mode": "official", "split": "testing", "source": source,
                "frame_index_base": 0, "reduction": "any_nonzero_pixel",
                "annotation_type": "pixel_mask_derived_frame_labels", "videos": records,
                "num_videos": len(records), "num_frames": sum(r["num_frames"] for r in records.values()),
                "num_anomaly_frames": sum(r["num_anomaly_frames"] for r in records.values()),
                "interval_crosscheck": "Any disagreements retained; official pixel masks define this protocol"}
    save_json(manifest, label_out / "testing/manifest.json")
    save_json({"source": source, "conversion": "FFV1 lossless verified each frame",
               "training": training_records, "testing": records,
               "training_ground_truth": "normal-only dataset protocol, no new manual labels",
               "annotations": str(label_out / "testing/manifest.json")}, out / "preparation.json")
    print(f"Official mask labels: {manifest['num_frames']} frames / {manifest['num_anomaly_frames']} anomalous", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=Path("data/downloads/UCSD_Anomaly_Dataset.tar.gz"))
    p.add_argument("--out", type=Path, default=Path("data/UCSDped2"))
    p.add_argument("--label-out", type=Path, default=Path("data/labels_ped2_official"))
    a = p.parse_args()
    if not a.archive.exists():
        a.archive.parent.mkdir(parents=True, exist_ok=True)
        partial = a.archive.with_suffix(a.archive.suffix + ".part")
        subprocess.run(["curl", "-fL", "--retry", "2", "--connect-timeout", "10", "--max-time", "900",
                        "-C", "-", "-o", str(partial), UCSD_URL], check=True)
        partial.replace(a.archive)
    prepare(a.archive, a.out, a.label_out)


if __name__ == "__main__":
    main()
