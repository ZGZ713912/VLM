#!/usr/bin/env python3
"""Train causal region VLM on normal training videos; freeze before test scoring."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from datasets.region_features import load_region_cache, RegionClips, sha256
from eval.region_evaluation import calibrate, component_scores, evaluate
from models.region_temporal import RegionMemory, RegionTemporalModel
from prompts.region import region_prompts
from train.region_trainer import region_loss, restore_rng, rng_state
from utils.io import save_json
from utils.provenance import runtime_info
from utils.region_config import load_region_config
from utils.reproducibility import set_seed


def atomic_save(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def split_videos(features: dict, fraction: float, seed: int) -> tuple[dict, dict]:
    if len(features) < 2 or not 0 < fraction < 1:
        raise ValueError("Need at least 2 training videos and valid holdout fraction")
    ids = sorted(features)
    random.Random(seed).shuffle(ids)
    n = min(len(ids) - 1, max(1, math.ceil(len(ids) * fraction)))
    held_out = set(ids[:n])
    return ({k: v for k, v in features.items() if k not in held_out},
            {k: v for k, v in features.items() if k in held_out})


def encode_text(cfg: dict, prompts: list[str], device: torch.device) -> torch.Tensor:
    from models.backbone import CLIPBackbone
    backbone = CLIPBackbone(cfg["backbone"]["name"], cfg["backbone"]["pretrained"]).to(device).eval()
    with torch.inference_mode():
        text = backbone.encode_text_pooled(prompts).clone()
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # Leave inference mode tensors out of autograd operations.
    return text.clone().detach()


def make_model(cfg: dict, dim: int, motion_dim: int, device: torch.device):
    return RegionTemporalModel(dim, motion_dim, **cfg["model"]).to(device)


def train(cfg: dict, run_name: str, resume: bool = False, stop_after_epoch: int | None = None,
          evaluate_test: bool = False, evaluate_only: bool = False) -> dict:
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run_name must be a single path component")
    seed = int(cfg["seed"])
    set_seed(seed)
    torch.set_num_threads(2)
    device = torch.device(cfg["device"])
    out, checkpoints = Path("results") / run_name, Path("checkpoints") / run_name
    if (out.exists() or checkpoints.exists()) and not (resume or evaluate_only):
        raise FileExistsError("Use a fresh run name or explicit --resume/--evaluate-only")
    if evaluate_only and not (checkpoints / "detector.pt").is_file():
        raise FileNotFoundError("Evaluation requires a finalized detector.pt")
    out.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    contract_path = out / "config.yaml"
    if contract_path.exists() and yaml.safe_load(contract_path.read_text()) != cfg:
        raise ValueError("Run/resume configuration changed")
    contract_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    # Read training cache only. Test cache + annotations are opened after finalization.
    features, cache_meta = load_region_cache(Path(cfg["data"]["feature_root"]) / "training",
                        Path(cfg["data"]["root"]) / "training_videos", cfg["backbone"])
    fit, held_out = split_videos(features, cfg["data"]["validation_fraction"], cfg["data"]["split_seed"])
    protocol = {"training": "normal-only, synthetic feature exposure, no true anomaly training labels",
                "fit_video_ids": sorted(fit), "calibration_video_ids": sorted(held_out),
                "split_seed": cfg["data"]["split_seed"], "training_seed": seed,
                "test_labels_used_for_training": False, "test_labels_used_for_calibration": False,
                "test_labels_used_for_selection": False,
                "selection_metric": "held-out normal temporal prediction loss",
                "top_regions": cfg["model"]["top_regions"],
                "train_cache_manifest_sha256": sha256(Path(cfg["data"]["feature_root"]) / "training/manifest.json")}
    save_json(protocol, out / "split_manifest.json")
    save_json(runtime_info(), out / "environment.json")
    sources = ["models/region_temporal.py", "train/region_trainer.py", "eval/region_evaluation.py",
               "datasets/region_features.py", "tools/extract_region_features.py", "scripts/train_region_model.py",
               "prompts/region.py", cfg["prompt"]["config"]]
    source_hashes = {p: sha256(p) for p in sources}
    if (out / "source_hashes.json").exists() and json.loads((out / "source_hashes.json").read_text()) != source_hashes:
        raise ValueError("Experiment source code changed; use a new run to keep the training contract")
    save_json(source_hashes, out / "source_hashes.json")
    prompts, polarity = region_prompts(cfg["prompt"]["config"], cfg["dataset"], cfg["prompt"]["type"])
    state = None
    if resume or evaluate_only:
        state = torch.load(checkpoints / ("detector.pt" if evaluate_only else "last.pt"),
                           map_location=device, weights_only=False)
        if state["config"] != cfg or state["protocol"] != protocol or state["prompts"] != prompts:
            raise ValueError("Checkpoint input/partition/prompt contract changed")
        text = state["text"].to(device)
    else:
        text = encode_text(cfg, prompts, device)
    polarity = polarity.to(device)
    model = make_model(cfg, cache_meta["appearance_dim"], next(iter(fit.values()))["motion"].shape[-1], device)
    memory, bank_indices = RegionMemory.fit(fit, seed=seed, **cfg["memory"])
    memory = memory.to(device).eval()
    model.motion_mean.copy_(memory.motion_mean)
    model.motion_std.copy_(memory.motion_std)
    if evaluate_only:
        model.load_state_dict(state["model"])
        memory.load_state_dict(state["memory"])
        calibration = state["calibration"]
        model.eval()
    else:
        train_cfg = cfg["train"]
        loader_generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(RegionClips(fit, train_cfg["clip_length"], train_cfg["clip_step"]),
                                 batch_size=train_cfg["batch_size"], shuffle=True, num_workers=0,
                                 generator=loader_generator)
        val_loader = DataLoader(RegionClips(held_out, train_cfg["clip_length"], train_cfg["clip_step"]),
                               batch_size=train_cfg["batch_size"], shuffle=False, num_workers=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
        first_epoch, step, best, history = 0, 0, float("inf"), []
        if resume:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            first_epoch, step, best, history = state["epoch"], state["step"], state["best"], state["history"]
            # RNG byte tensors must be on CPU even when model is restored on CUDA.
            rng = state["rng"]
            rng["torch"], rng["loader"] = rng["torch"].cpu(), rng["loader"].cpu()
            rng["cuda"] = [value.cpu() for value in rng["cuda"]]
            restore_rng(rng, loader_generator)
        writer = SummaryWriter(str(Path("logs") / run_name), purge_step=step if resume else None)
        epochs = min(train_cfg["epochs"], stop_after_epoch or train_cfg["epochs"])
        for epoch in range(first_epoch, epochs):
            model.train()
            totals, count = {}, 0
            for batch in train_loader:
                appearance, motion = batch["appearance"].to(device), batch["motion"].to(device)
                optimizer.zero_grad(set_to_none=True)
                losses = region_loss(model, appearance, motion, text, polarity, train_cfg["loss"])
                if not torch.isfinite(losses["loss"]):
                    raise RuntimeError("Nonfinite training loss")
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                count += len(appearance)
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.) + float(value.detach()) * len(appearance)
                    writer.add_scalar(f"train/{key}", float(value.detach()), step)
                step += 1
            model.eval()
            val_total, val_count = 0., 0
            with torch.no_grad():
                for batch in val_loader:
                    losses = region_loss(model, batch["appearance"].to(device), batch["motion"].to(device),
                                         text, polarity, train_cfg["loss"], synthetic=False)
                    val_total += float(losses["prediction"]) * len(batch["appearance"])
                    val_count += len(batch["appearance"])
            validation = val_total / val_count
            scheduler.step()
            improved = validation < best
            best = min(best, validation)
            entry = {"epoch": epoch + 1, "step": step, "train": {k: v / count for k, v in totals.items()},
                     "validation_prediction": validation, "selected": improved}
            history.append(entry)
            writer.add_scalar("validation/prediction", validation, step)
            payload = {"version": 1, "config": cfg, "protocol": protocol, "prompts": prompts,
                       "text": text.detach().cpu(), "epoch": epoch + 1, "step": step, "best": best,
                       "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "scheduler": scheduler.state_dict(), "rng": rng_state(loader_generator), "history": history}
            atomic_save(payload, checkpoints / "last.pt")
            if improved:
                atomic_save(payload, checkpoints / "best.pt")
            save_json(history, out / "history.json")
            print(json.dumps(entry), flush=True)
        writer.close()
        if epochs < train_cfg["epochs"]:
            return {"status": "interrupted_after_epoch", "epoch": epochs}
        best_state = torch.load(checkpoints / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best_state["model"])
        model.eval()
        raw = {vid: component_scores(model, memory, item, text, polarity, cfg["model"]["top_regions"])[0]
               for vid, item in held_out.items()}
        calibration = calibrate(raw, cfg["eval"])
        np.savez_compressed(out / "calibration_components.npz", **raw)
        save_json(calibration, out / "calibration.json")
        atomic_save({"version": 1, "config": cfg, "protocol": protocol, "prompts": prompts,
                     "text": text.detach().cpu(), "polarity": polarity.cpu(), "model": model.state_dict(),
                     "memory": memory.state_dict(), "bank_indices": bank_indices, "calibration": calibration,
                     "selected_epoch": best_state["epoch"], "source_hashes": source_hashes}, checkpoints / "detector.pt")
        save_json({"status": "trained_and_frozen", "selected_epoch": best_state["epoch"],
                   "best_validation_prediction": best, "checkpoint_sha256": sha256(checkpoints / "detector.pt")}, out / "training_receipt.json")
    if evaluate_test or evaluate_only:
        test, test_meta = load_region_cache(Path(cfg["data"]["feature_root"]) / "testing",
                        Path(cfg["data"]["root"]) / "testing_videos", cfg["backbone"])
        protocol = {**protocol, "test_cache_manifest_sha256": sha256(Path(cfg["data"]["feature_root"]) / "testing/manifest.json"),
                    "checkpoint_sha256": sha256(checkpoints / "detector.pt")}
        report = evaluate(model, memory, test, text, polarity, prompts, calibration, cfg["eval"],
                          Path(cfg["data"]["label_root"]), out / "official", protocol, seed)
        print("OFFICIAL", json.dumps({v: {k: x for k, x in m.items() if k in ("frame_auc", "frame_ap")}
                                      for v, m in report["variants"].items()}), flush=True)
        return report
    return {"status": "trained_and_frozen"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/region_temporal.yaml")
    p.add_argument("--run-name", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--stop-after-epoch", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--prompt-type", choices=["label", "scene", "contrast"])
    p.add_argument("--fusion", choices=["concat", "crossattn"])
    a = p.parse_args()
    cfg = load_region_config(a.config)
    if a.seed is not None:
        cfg["seed"] = a.seed
    if a.prompt_type:
        cfg["prompt"]["type"] = a.prompt_type
    if a.fusion:
        cfg["model"]["fusion"] = a.fusion
    train(cfg, a.run_name, a.resume, a.stop_after_epoch, a.evaluate, a.evaluate_only)


if __name__ == "__main__":
    main()
