"""Prepare warm-start checkpoint for V45 (= V44 + warm-start from detection_hybrid)."""
import sys
from pathlib import Path
import torch
import hydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "experiments" / "stackoverflow" / "checkpoints"
DETPP = CKPT_DIR / "detection_hybrid.ckpt"
OUT = CKPT_DIR / "v45_warmstart.ckpt"


def main():
    print(f"Loading DeTPP ckpt: {DETPP}")
    detpp_sd = torch.load(DETPP, map_location="cpu")
    if isinstance(detpp_sd, dict) and "state_dict" in detpp_sd:
        detpp_sd = detpp_sd["state_dict"]
    print(f"  {len(detpp_sd)} keys")

    print("Instantiating fresh V44 module...")
    cfg_dir = ROOT / "experiments" / "stackoverflow" / "configs"
    with hydra.initialize_config_dir(config_dir=str(cfg_dir), version_base="1.2"):
        cfg = hydra.compose(config_name="diffusion_gru_detpp_v44")
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    v44 = hydra.utils.instantiate(cfg.module)
    fresh = v44.state_dict()
    print(f"  V44 has {len(fresh)} keys")

    # V44 has no _head queries (queries are inside _decoder). Remap:
    #   _seq_encoder.* -> same
    #   _loss.<inner>  -> _loss._next_item.<inner>  (DetectionLoss state)
    # _head.* (DeTPP's ConditionalHead) — V44's _head is a thin Linear, not
    # ConditionalHead, so we skip those keys.
    remap = {}
    for k, v in detpp_sd.items():
        if k.startswith("_seq_encoder."):
            remap[k] = v
        elif k.startswith("_loss."):
            remap["_loss._next_item." + k[len("_loss."):]] = v
        elif k == "_head.queries":
            # DeTPP queries (K, query_size) -> V44 decoder queries (same shape).
            remap["_loss._decoder.queries"] = v

    merged = dict(fresh)
    matched = 0
    skipped_shape = 0
    skipped_missing = 0
    for k, v in remap.items():
        if k not in fresh:
            skipped_missing += 1
            continue
        if v.shape != fresh[k].shape:
            skipped_shape += 1
            continue
        merged[k] = v
        matched += 1

    print(f"\nResults:")
    print(f"  matched + copied: {matched}")
    print(f"  skipped (missing in V44 layout): {skipped_missing}")
    print(f"  skipped (shape mismatch): {skipped_shape}")
    print(f"\nSaving to {OUT}")
    torch.save(merged, OUT)
    print("Done.")


if __name__ == "__main__":
    main()
