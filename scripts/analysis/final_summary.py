"""Collect all single-seed and multiseed results and build the final table."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "stackoverflow" / "results"

SINGLE = [
    "detection_hybrid",
    "diffusion_gru",
    "diffusion_gru_detpp",
    "diffusion_gru_detpp_v1_balance_b_floor",
    "diffusion_gru_detpp_v3",
    "diffusion_gru_detpp_v3_wide",
    "diffusion_gru_detpp_v3_xwide",
    "diffusion_gru_detpp_v3_fat",
    "diffusion_gru_detpp_v4",
    "diffusion_gru_detpp_v5",
    "diffusion_gru_detpp_v5_alpha02",
    "diffusion_gru_detpp_v7",
    "diffusion_gru_detpp_v9",
    "diffusion_gru_detpp_v12",
    "diffusion_gru_detpp_v13",
    "diffusion_gru_detpp_v14",
    "diffusion_gru_detpp_v25",
    "diffusion_gru_detpp_v28",
    "diffusion_gru_detpp_v29",
    "diffusion_gru_detpp_v33",
    "diffusion_gru_detpp_v40",
    "diffusion_gru_detpp_v41",
    "diffusion_gru_detpp_v42",
    "diffusion_gru_detpp_v43",
    "diffusion_gru_detpp_v44",
    "diffusion_gru_detpp_v45",
]
MULTI = [
    "multiseed_detection_hybrid",
    "multiseed_diffusion_gru_detpp_v5",
    "multiseed_diffusion_gru_detpp_v9",
    "multiseed_diffusion_gru_detpp_v14",
    "multiseed_diffusion_gru_detpp_v43",
    "multiseed_diffusion_gru_detpp_v44",
]

WANTED = [
    "test/T-mAP",
    "test/T-mAP-std",
    "test/optimal-transport-distance",
    "test/next-item-accuracy",
    "test/sequence-labels-entropy",
    "test/mean-predicted-length",
    "test/v5_residual_alpha",
    "test/v5_residual_to_base_ratio",
    "val/T-mAP",
]


def load(name):
    p = RES / f"{name}.yaml"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def format_pm(mean, std):
    if mean is None:
        return "n/a"
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def main():
    print("\n" + "=" * 100)
    print("COMPREHENSIVE SUMMARY — StackOverflow")
    print("=" * 100)

    print("\n-- MULTISEED (5 seeds each) --")
    print(f"{'Model':55s}  {'test T-mAP':24s}  {'val T-mAP':12s}  {'OTD':8s}  {'next-acc':10s}")
    for name in MULTI:
        d = load(name)
        if d is None:
            continue
        tmap = d.get("test/T-mAP")
        tmap_std = d.get("test/T-mAP-std")
        val_tmap = d.get("val/T-mAP")
        otd = d.get("test/optimal-transport-distance")
        nxt = d.get("test/next-item-accuracy")
        print(f"{name:55s}  {format_pm(tmap, tmap_std):24s}  {val_tmap or 0:.4f}     {otd or 0:.3f}     {nxt or 0:.4f}")

    print("\n-- SINGLE SEED (seed=42) --")
    print(f"{'Model':55s}  {'test T-mAP':12s}  {'val T-mAP':12s}  {'OTD':8s}  {'next-acc':10s}  {'eff-a':8s}")
    for name in SINGLE:
        d = load(name)
        if d is None:
            continue
        tmap = d.get("test/T-mAP")
        val_tmap = d.get("val/T-mAP")
        otd = d.get("test/optimal-transport-distance")
        nxt = d.get("test/next-item-accuracy")
        r2b = d.get("test/v5_residual_to_base_ratio")
        alpha_eff = f"{r2b:.3f}" if r2b is not None else "-"
        print(f"{name:55s}  {tmap or 0:.4f}        {val_tmap or 0:.4f}      {otd or 0:.3f}    {nxt or 0:.4f}      {alpha_eff}")


if __name__ == "__main__":
    main()
