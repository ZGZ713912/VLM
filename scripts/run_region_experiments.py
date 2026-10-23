#!/usr/bin/env python3
"""Run a declared finite matrix; freeze all models before official evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from datasets.region_features import sha256
from utils.io import save_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", default="configs/region_campaign.yaml", type=Path)
    p.add_argument("--phase", choices=["train", "evaluate", "all"], default="all")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    plan = yaml.safe_load(a.plan.read_text())
    out = Path("results") / plan["campaign"]
    if a.dry_run:
        print(yaml.safe_dump(plan))
        return
    out.mkdir(parents=True, exist_ok=True)
    receipt = {"plan": plan, "plan_sha256": sha256(a.plan)}
    receipt_path = out / "plan.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise ValueError("Campaign plan changed; use a fresh campaign")
    save_json(receipt, receipt_path)
    statuses = {}
    for phase in (["train", "evaluate"] if a.phase == "all" else [a.phase]):
        # Do not start the test pass until every training run has finalized.
        if phase == "evaluate":
            missing = [r["run"] for r in plan["runs"] if not (Path("checkpoints") / r["run"] / "detector.pt").exists()]
            if missing:
                raise RuntimeError(f"Training incomplete; keep official test pass closed: {missing}")
        for run in plan["runs"]:
            name = run["run"]
            completed = (Path("checkpoints") / name / "detector.pt" if phase == "train"
                         else Path("results") / name / "official/metrics.json")
            if completed.exists():
                statuses[f"{phase}/{name}"] = {"status": "existing", "sha256": sha256(completed)}
                continue
            cmd = [sys.executable, "scripts/train_region_model.py", "--config", run["config"],
                   "--run-name", name, "--seed", str(run["seed"]), "--prompt-type", run["prompt"], "--fusion", run["fusion"]]
            if phase == "evaluate":
                cmd.append("--evaluate-only")
            elif (Path("checkpoints") / name / "last.pt").exists():
                cmd.append("--resume")
            print(f"{phase}: {name}", flush=True)
            log = out / f"{phase}_{name}.log"
            with log.open("a") as stream:
                result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
            statuses[f"{phase}/{name}"] = {"returncode": result.returncode, "log": str(log)}
            save_json(statuses, out / "statuses.json")
            if result.returncode or not completed.exists():
                raise RuntimeError(f"Experiment failed: {name}; inspect {log}")
        save_json(statuses, out / "statuses.json")
    if a.phase in ("all", "evaluate"):
        rows = []
        for run in plan["runs"]:
            report = json.loads((Path("results") / run["run"] / "official/metrics.json").read_text())
            for variant, metric in report["variants"].items():
                rows.append({**run, "variant": variant, "primary": variant == report["primary_variant"],
                             "frame_auc": metric["frame_auc"], "frame_ap": metric["frame_ap"],
                             "test_fpr": metric["test_fpr_at_training_threshold"],
                             "test_tpr": metric["test_tpr_at_training_threshold"]})
        save_json({**receipt, "results": rows}, out / "summary.json")
        lines = ["| run | variant | AUC | AP |", "|---|---|---:|---:|"]
        lines.extend(f"| {r['run']} | {r['variant']} | {r['frame_auc']:.4f} | {r['frame_ap']:.4f} |" for r in rows)
        (out / "summary.md").write_text("\n".join(lines) + "\n")
        print(f"Campaign finished: {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
