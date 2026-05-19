"""Prepare a warm-start checkpoint for V14 by combining detection_hybrid weights
with a fresh V9 model structure.

V9 stores ConditionalHead at `_loss._base_head.*` and the inner NextItemLoss at
`_loss._next_item.*`, whereas detection_hybrid stores them at `_head.*` and
`_loss.*` respectively. This script remaps the DeTPP keys into V9's layout so
we can `init_from_checkpoint` without key mismatches.

Output: experiments/stackoverflow/checkpoints/v14_warmstart.ckpt — a full state
dict compatible with V9's architecture, with base_head + seq_encoder + inner
DetectionLoss warm-started from detection_hybrid, and the diffusion/residual
modules left at fresh-initialisation values.
"""
import sys
from pathlib import Path
import torch
import hydra
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "experiments" / "stackoverflow" / "checkpoints"
DETPP_CKPT = CKPT_DIR / "detection_hybrid.ckpt"
OUT_CKPT = CKPT_DIR / "v14_warmstart.ckpt"


def build_fresh_v9_state_dict():
    """Instantiate V9 via Hydra and grab its fresh state dict."""
    # Route Hydra to the v9 config.
    cfg_dir = ROOT / "experiments" / "stackoverflow" / "configs"
    with hydra.initialize_config_dir(config_dir=str(cfg_dir), version_base="1.2"):
        cfg = hydra.compose(config_name="diffusion_gru_detpp_v9")
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    model = hydra.utils.instantiate(cfg.module)
    return model.state_dict()


def remap_detpp_to_v9(detpp_sd):
    """Rename keys from detection_hybrid's layout to V9's layout.

    DeTPP                        -> V9
    _seq_encoder.*               -> _seq_encoder.*
    _head.*                      -> _loss._base_head.*
    _loss.<inner>                -> _loss._next_item.<inner>
    _loss._matching_priors       -> _loss._next_item._matching_priors
    _loss._matching_thresholds   -> _loss._next_item._matching_thresholds
    """
    out = {}
    for k, v in detpp_sd.items():
        if k.startswith("_seq_encoder."):
            out[k] = v
        elif k.startswith("_head."):
            new_k = "_loss._base_head." + k[len("_head."):]
            out[new_k] = v
        elif k.startswith("_loss."):
            new_k = "_loss._next_item." + k[len("_loss."):]
            out[new_k] = v
        else:
            print(f"[remap] unknown prefix, skipping: {k}", file=sys.stderr)
    return out


def main():
    print(f"Loading DeTPP ckpt: {DETPP_CKPT}")
    detpp_sd = torch.load(DETPP_CKPT, map_location="cpu")
    if isinstance(detpp_sd, dict) and "state_dict" in detpp_sd:
        detpp_sd = detpp_sd["state_dict"]
    print(f"  {len(detpp_sd)} keys")

    print("Instantiating fresh V9 model...")
    v9_sd = build_fresh_v9_state_dict()
    print(f"  {len(v9_sd)} keys")

    print("Remapping DeTPP -> V9 layout...")
    remapped = remap_detpp_to_v9(detpp_sd)
    print(f"  {len(remapped)} remapped keys")

    # Merge: start from fresh V9, overwrite with DeTPP remapped keys.
    merged = dict(v9_sd)
    matched = 0
    shape_mismatched = []
    not_in_v9 = []
    for k, v in remapped.items():
        if k not in v9_sd:
            not_in_v9.append(k)
            continue
        if v.shape != v9_sd[k].shape:
            shape_mismatched.append((k, tuple(v.shape), tuple(v9_sd[k].shape)))
            continue
        merged[k] = v
        matched += 1

    print(f"\nResults:")
    print(f"  matched + copied: {matched}")
    print(f"  not in V9 schema: {len(not_in_v9)} -> {not_in_v9[:10]}")
    print(f"  shape mismatched: {len(shape_mismatched)} -> {shape_mismatched[:10]}")
    print(f"  remaining fresh (diffusion modules): {len(v9_sd) - matched} keys")

    print(f"\nSaving warmstart ckpt to {OUT_CKPT}")
    torch.save(merged, OUT_CKPT)
    print("Done.")


if __name__ == "__main__":
    main()
