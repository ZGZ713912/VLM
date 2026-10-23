#!/usr/bin/env python3
"""Summarize all declared experiments, paired video bootstrap and seed variation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import roc_auc_score
import yaml

from datasets.region_features import sha256
from utils.io import save_json


def paired_video_difference(new: dict, old: dict, labels: dict, samples: int = 2000) -> dict:
    ids = sorted(labels)
    if set(new) != set(old) or set(new) != set(ids):
        raise ValueError("Paired comparison video IDs differ")
    for vid in ids:
        if len(new[vid]) != len(labels[vid]) or len(old[vid]) != len(labels[vid]):
            raise ValueError("Paired comparison frame counts differ")
    rng, delta = np.random.default_rng(42), []
    for _ in range(samples):
        selected = rng.choice(ids, len(ids), replace=True)
        y = np.concatenate([labels[k] for k in selected])
        if len(np.unique(y)) < 2:
            continue
        newer = np.concatenate([new[k] for k in selected])
        older = np.concatenate([old[k] for k in selected])
        delta.append(roc_auc_score(y, newer) - roc_auc_score(y, older))
    y = np.concatenate([labels[k] for k in ids])
    return {"auc_difference": float(roc_auc_score(y, np.concatenate([new[k] for k in ids]))
                                     - roc_auc_score(y, np.concatenate([old[k] for k in ids]))),
            "paired_video_bootstrap_95_interval": np.quantile(delta, [.025, .975]).tolist(),
            "samples": samples, "positive_fraction": float(np.mean(np.asarray(delta) > 0)),
            "note": "Exploratory benchmark CI; does not estimate performance in unseen scenes"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=Path("configs/region_campaign.yaml"))
    a = p.parse_args()
    plan = yaml.safe_load(a.plan.read_text())
    out = Path("results") / plan["campaign"]
    rows, groups = [], {}
    for run in plan["runs"]:
        directory = Path("results") / run["run"]
        cfg = yaml.safe_load((directory / "config.yaml").read_text())
        report = json.loads((directory / "official/metrics.json").read_text())
        receipt = json.loads((directory / "training_receipt.json").read_text())
        history = json.loads((directory / "history.json").read_text())
        primary = report["variants"][report["primary_variant"]]
        row = {**run, "dataset": cfg["dataset"], "backbone": cfg["backbone"]["name"],
               "frame_auc": primary["frame_auc"], "frame_ap": primary["frame_ap"],
               "test_fpr": primary["test_fpr_at_training_threshold"],
               "test_tpr": primary["test_tpr_at_training_threshold"],
               "selected_epoch": receipt["selected_epoch"], "epochs": len(history),
               "updates": history[-1]["step"], "checkpoint_sha256": receipt["checkpoint_sha256"],
               "validation_prediction_start": history[0]["validation_prediction"],
               "validation_prediction_end": history[-1]["validation_prediction"]}
        rows.append(row)
        key = f"{row['dataset']}/{row['backbone']}/{row['prompt']}/{row['fusion']}"
        groups.setdefault(key, []).append(row)
    group_report = {key: {"seeds": [r["seed"] for r in group], "n": len(group),
                         "auc_mean": float(np.mean([r["frame_auc"] for r in group])),
                         "auc_std": float(np.std([r["frame_auc"] for r in group], ddof=1)) if len(group) > 1 else None,
                         "ap_mean": float(np.mean([r["frame_ap"] for r in group]))}
                    for key, group in groups.items()}
    comparisons = {}
    baseline_dir = Path("results/official_normality_verified")
    avenue = Path("results") / plan["primary"]["avenue"] / "official"
    if (baseline_dir / "frame_scores.npz").exists():
        comparisons["avenue_vs_global_normal_memory"] = paired_video_difference(
            dict(np.load(avenue / "full_scores.npz")), dict(np.load(baseline_dir / "frame_scores.npz")),
            dict(np.load(avenue / "frame_labels.npz")))
    zero_shot_dir = Path("results/official_zero_shot_verified/scene_only")
    if (zero_shot_dir / "frame_scores.npz").exists():
        expected_labels = dict(np.load(avenue / "frame_labels.npz"))
        old_labels = dict(np.load(zero_shot_dir / "frame_labels.npz"))
        if set(expected_labels) != set(old_labels) or any(not np.array_equal(v, old_labels[k]) for k, v in expected_labels.items()):
            raise ValueError("Historical zero-shot uses different labels")
        comparisons["avenue_vs_historical_scene_zero_shot"] = paired_video_difference(
            dict(np.load(avenue / "full_scores.npz")), dict(np.load(zero_shot_dir / "frame_scores.npz")), expected_labels)
    save_json({"plan_sha256": sha256(a.plan), "runs": rows, "groups": group_report,
               "paired_comparisons": comparisons, "total_epochs": sum(r["epochs"] for r in rows),
               "total_optimizer_updates": sum(r["updates"] for r in rows)}, out / "analysis.json")
    lines = ["| dataset | backbone | prompt | fusion | seed | AUC | AP | FPR | TPR |", "|---|---|---|---|---:|---:|---:|---:|---:|"]
    lines.extend(f"| {r['dataset']} | {r['backbone']} | {r['prompt']} | {r['fusion']} | {r['seed']} | "
                 f"{r['frame_auc']:.4f} | {r['frame_ap']:.4f} | {r['test_fpr']:.3f} | {r['test_tpr']:.3f} |" for r in rows)
    (out / "primary_results.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = ["#287b8e", "#dc7855"]
    for column, dataset in enumerate(("avenue", "ped2")):
        primary_dir = Path("results") / plan["primary"][dataset]
        history = json.loads((primary_dir / "history.json").read_text())
        axes[0, column].plot([h["epoch"] for h in history],
                             [h["train"]["prediction"] for h in history], label="Training prediction", color=colors[0])
        axes[0, column].plot([h["epoch"] for h in history],
                             [h["validation_prediction"] for h in history], label="Held-out normal prediction", color=colors[1])
        axes[0, column].set(title=f"{dataset.upper()}: primary model learning", xlabel="Epoch", ylabel="Prediction loss")
        axes[0, column].legend(frameon=False, fontsize=9)
        report = json.loads((primary_dir / "official/metrics.json").read_text())
        variants = ["semantic", "appearance", "motion", "prediction", "spatial_motion", "full"]
        values = [report["variants"][v]["frame_auc"] for v in variants]
        bars = axes[1, column].barh(variants, values, color=[colors[1] if v == "full" else colors[0] for v in variants])
        for bar, value in zip(bars, values):
            axes[1, column].text(value + .006, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
        axes[1, column].set(xlim=(0, 1.07), xlabel="Official pooled frame ROC-AUC", title=f"{dataset.upper()}: fixed ablations")
        axes[1, column].axvline(.5, color="gray", linestyle="--", linewidth=.8)
    fig.suptitle("Normal-only region VLM: fixed primary model and component ablations\n"
                 "Training videos only for fitting/selection/calibration; test labels only for scoring", fontsize=12)
    fig.savefig(out / "learning_and_ablation.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"groups": group_report, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
