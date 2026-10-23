"""Training-calibrated component fusion and independent official frame scoring."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from datasets.cache import validate_official_cache
from datasets.labels import load_frame_labels
from datasets.region_features import sha256
from eval.metrics import binary_ranking_metrics
from utils.io import save_json
from utils.visualization import plot_temporal_heatmap


COMPONENTS = ("appearance", "motion", "prediction", "semantic", "classifier")


@torch.inference_mode()
def component_scores(model, memory, item: dict[str, Tensor], text: Tensor,
                     polarity: Tensor, top_regions: int) -> tuple[np.ndarray, dict]:
    device = text.device
    appearance = item["appearance"].float().unsqueeze(0).to(device)
    motion = item["motion"].float().unsqueeze(0).to(device)
    output = model(appearance, motion, text)
    normality = memory(appearance, motion)
    similarity = F.normalize(appearance[0], dim=-1) @ F.normalize(text, dim=-1).T
    semantic = similarity[..., polarity > 0].mean(-1) - similarity[..., polarity < 0].mean(-1)
    # Predicted t uses recurrent state from t-1; first frame has no temporal evidence.
    prediction = torch.zeros(appearance.shape[1:3], device=device)
    prediction[1:] = (1 - F.cosine_similarity(output["predicted_appearance"][0, :-1],
                                             appearance[0, 1:], dim=-1)
                       + .25 * F.smooth_l1_loss(output["predicted_motion"][0, :-1],
                                                output["standardized_motion"][0, 1:], reduction="none").mean(-1))
    regions = {"appearance": normality["appearance"][0], "motion": normality["motion"][0],
               "prediction": prediction, "semantic": semantic, "classifier": output["region_logits"][0]}
    k = min(top_regions, appearance.shape[2])
    scores = torch.stack([regions[key].topk(k, dim=-1).values.mean(-1) for key in COMPONENTS], -1)
    if not torch.isfinite(scores).all():
        raise ValueError("Nonfinite video scores")
    details = {key: value.cpu().numpy() for key, value in regions.items()}
    details["prompt_similarities"] = similarity.cpu().numpy()
    return scores.cpu().numpy(), details


def causal_smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1:
        raise ValueError("Smoothing window must be positive")
    values = np.asarray(values, dtype=np.float64)
    sums = np.concatenate([[0.], np.cumsum(values)])
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)
    return (sums[end] - sums[start]) / (end - start)


def calibrate(raw: dict[str, np.ndarray], cfg: dict) -> dict:
    if not raw:
        raise ValueError("Require held-out training videos for calibration")
    all_scores = np.concatenate(list(raw.values()))
    center = np.median(all_scores, axis=0)
    scale = np.maximum(np.quantile(all_scores, .99, axis=0) - center, 1e-6)
    result = {"component_order": list(COMPONENTS), "center": center.tolist(), "scale": scale.tolist(),
              "source_video_ids": sorted(raw), "source": "held_out_normal_training_videos",
              "normality_assumption": "Predominantly normal; no frame-level training GT",
              "thresholds": {}}
    normalized = {vid: np.clip((scores - center) / scale, -5, 20) for vid, scores in raw.items()}
    for variant, weights in cfg["variants"].items():
        vector = weight_vector(weights)
        values = np.concatenate([causal_smooth(scores @ vector, cfg["smooth_window"]) for scores in normalized.values()])
        result["thresholds"][variant] = float(np.quantile(values, cfg["calibration_quantile"]))
    return result


def weight_vector(weights: dict) -> np.ndarray:
    if set(weights) - set(COMPONENTS) or not weights or min(weights.values()) < 0 or sum(weights.values()) <= 0:
        raise ValueError("Invalid component fusion weights")
    return np.array([weights.get(key, 0) for key in COMPONENTS], dtype=float) / sum(weights.values())


def fuse(raw: np.ndarray, calibration: dict, weights: dict, window: int) -> np.ndarray:
    if calibration["component_order"] != list(COMPONENTS):
        raise ValueError("Calibration component order mismatch")
    normalized = np.clip((raw - calibration["center"]) / calibration["scale"], -5, 20)
    return causal_smooth(normalized @ weight_vector(weights), window)


def video_bootstrap(scores: dict, labels: dict, samples: int, seed: int) -> dict:
    """Cluster bootstrap resamples complete videos, preserving temporal dependence."""
    rng = np.random.default_rng(seed)
    ids, aucs = sorted(scores), []
    for _ in range(samples):
        selected = rng.choice(ids, len(ids), replace=True)
        metric = binary_ranking_metrics(np.concatenate([scores[x] for x in selected]),
                                         np.concatenate([labels[x] for x in selected]), "frame")["frame_auc"]
        if metric is not None:
            aucs.append(metric)
    return {"unit": "video", "samples": samples, "valid_samples": len(aucs),
            "frame_auc_95_percentile_interval": np.quantile(aucs, [.025, .975]).tolist() if aucs else None}


def evaluate(model, memory, videos: dict, text: Tensor, polarity: Tensor, prompts: list[str],
             calibration: dict, cfg: dict, label_root: Path, out: Path,
             protocol: dict, seed: int) -> dict:
    meta = validate_official_cache(label_root / "testing")
    if set(meta["videos"]) != set(videos):
        raise ValueError("Official annotation/feature video IDs differ")
    out.mkdir(parents=True, exist_ok=False)
    labels, scores_by_variant, raw_scores, explanations = {}, {v: {} for v in cfg["variants"]}, {}, {}
    for vid, item in videos.items():
        path = label_root / "testing" / f"{vid}.pt"
        if sha256(path) != meta["videos"][vid]["label_sha256"]:
            raise ValueError(f"Official label checksum mismatch: {vid}")
        n = len(item["appearance"])
        label = load_frame_labels(path, n).numpy()
        if n != meta["videos"][vid]["num_frames"] or label.sum() != meta["videos"][vid]["num_anomaly_frames"]:
            raise ValueError("Official label count mismatch")
        labels[vid] = label
        raw, details = component_scores(model, memory, item, text, polarity, protocol["top_regions"])
        raw_scores[vid] = raw
        for variant, weights in cfg["variants"].items():
            scores_by_variant[variant][vid] = fuse(raw, calibration, weights, cfg["smooth_window"])
        primary = scores_by_variant[cfg["primary_variant"]][vid]
        ranked = np.argsort(-primary, kind="stable")[:10].tolist()
        peak = ranked[0]
        region = int(np.argmax(details["appearance"][peak]))
        prompt = int(np.argmax(details["prompt_similarities"][peak, region]))
        explanations[vid] = {"top_abnormal_frames_zero_based": ranked, "peak_region_index": region,
                             "candidate_text": prompts[prompt], "candidate_semantics_verified": False,
                             "components_at_peak": dict(zip(COMPONENTS, raw[peak].tolist())),
                             "explanation": f"第 {peak} 帧的区域 {region} 偏离正常训练模式；"
                                            f"相近候选文本为 {prompts[prompt]}。该文本是检索候选，不是已确认事件。"}
        plot_temporal_heatmap(primary, label, title=f"{vid}: fixed region VLM / official GT",
                              save_path=out / "plots" / f"{vid}.png",
                              threshold=calibration["thresholds"][cfg["primary_variant"]])
    report = {"primary_variant": cfg["primary_variant"], "variants": {}, "protocol": protocol,
              "calibration": calibration, "label_source": meta["source"],
              "label_manifest_sha256": sha256(label_root / "testing/manifest.json"),
              "num_frames": sum(len(v) for v in labels.values()),
              "num_anomaly_frames": sum(int(v.sum()) for v in labels.values()),
              "score_protocol": "pooled raw training-calibrated scores; no per-test-video normalization",
              "video_auc": None, "video_auc_reason": "all testing videos contain anomalies"}
    all_labels = np.concatenate(list(labels.values()))
    for variant, scores in scores_by_variant.items():
        dense = np.concatenate(list(scores.values()))
        prediction = dense > calibration["thresholds"][variant]
        positives, negatives = all_labels == 1, all_labels == 0
        report["variants"][variant] = {**binary_ranking_metrics(dense, all_labels, "frame"),
            "per_video": {vid: binary_ranking_metrics(scores[vid], labels[vid], "frame") for vid in videos},
            "test_fpr_at_training_threshold": float(prediction[negatives].mean()) if negatives.any() else None,
            "test_tpr_at_training_threshold": float(prediction[positives].mean()) if positives.any() else None,
            "bootstrap": video_bootstrap(scores, labels, cfg["bootstrap_samples"], seed)}
        np.savez_compressed(out / f"{variant}_scores.npz", **scores)
    np.savez_compressed(out / "frame_labels.npz", **labels)
    np.savez_compressed(out / "raw_components.npz", **raw_scores)
    save_json(explanations, out / "explanations.json")
    save_json(report, out / "metrics.json")
    return report
