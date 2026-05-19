"""Sequential experiment runner: trains a queue of configs one at a time,
records final test/T-mAP, keeps running until the queue is drained.

Each job: cd experiments/stackoverflow && python -m hotpp.train --config-name NAME
Logs stream to experiments/stackoverflow/lightning_logs/_<NAME>_train.log.

Usage:
    python scripts/analysis/run_queue.py config1 config2 ...
or just edit QUEUE below and run without args.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path
import yaml


QUEUE = [
    "diffusion_gru_detpp_v3_fat",
    "diffusion_gru_detpp_v7_alpha005",
    "diffusion_gru_detpp_v7_learnable",
]

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "stackoverflow"
RESULTS = EXP / "results"
LOGS = EXP / "lightning_logs"


def train_one(name: str) -> dict:
    log_path = LOGS / f"_{name}_train.log"
    err_path = LOGS / f"_{name}_train.err"
    # Delete any stale result to tell completion later.
    result_path = RESULTS / f"{name}.yaml"
    if result_path.exists():
        # Keep old result — skip training if user wants to.
        print(f"[queue] SKIP {name}: result already exists at {result_path}")
        return read_result(result_path)

    print(f"[queue] START {name} at {time.strftime('%H:%M:%S')}")
    with open(log_path, "w", encoding="utf-8") as log_f, open(err_path, "w", encoding="utf-8") as err_f:
        proc = subprocess.Popen(
            [
                str(ROOT / ".venv" / "Scripts" / "python.exe"),
                "-u", "-m", "hotpp.train",
                "--config-dir", "configs",
                "--config-name", name,
            ],
            cwd=EXP,
            stdout=log_f,
            stderr=err_f,
        )
        rc = proc.wait()
    print(f"[queue] FINISHED {name} rc={rc} at {time.strftime('%H:%M:%S')}")
    if rc != 0:
        # Still try to report what we have
        with open(err_path, "r", encoding="utf-8", errors="replace") as f:
            tail = f.read()[-2000:]
        print(f"[queue] stderr tail for {name}:\n{tail}")
    if not result_path.exists():
        return {"name": name, "status": f"no result file, rc={rc}"}
    return read_result(result_path)


def read_result(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    keep = {
        k: data.get(k) for k in [
            "test/T-mAP",
            "test/T-mAP-weighted",
            "test/horizon-max-f-score",
            "test/next-item-accuracy",
            "test/optimal-transport-distance",
            "test/sequence-labels-entropy",
            "test/loss_diffusion",
            "test/loss_decoder_labels",
            "test/v5_residual_alpha",
            "test/v5_residual_to_base_ratio",
            "test/v7_residual_alpha",
            "test/v7_residual_to_base_ratio",
        ]
    }
    keep["name"] = path.stem
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", default=None)
    ap.add_argument("--no-skip", action="store_true", help="Re-run even if result exists.")
    args = ap.parse_args()

    queue = args.configs if args.configs else QUEUE
    results = []
    for name in queue:
        if args.no_skip:
            p = RESULTS / f"{name}.yaml"
            if p.exists():
                p.unlink()
        res = train_one(name)
        results.append(res)
        print(f"[queue] RESULT {name}: test/T-mAP = {res.get('test/T-mAP')}")

    print("\n=========== SUMMARY ===========")
    for r in results:
        tmap = r.get("test/T-mAP")
        nxt = r.get("test/next-item-accuracy")
        otd = r.get("test/optimal-transport-distance")
        print(f"{r['name']:50s}  T-mAP={tmap}  next-acc={nxt}  OTD={otd}")


if __name__ == "__main__":
    main()
