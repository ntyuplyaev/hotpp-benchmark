"""Side-by-side forensic comparison of diffusion_gru, detection_hybrid, V3.

For one fixed batch we trace every intermediate tensor in each model and
print shapes/stats. Goal: confirm V3 implements `diffusion-in-latent ->
matching` correctly with no hidden bugs in padding, presence, or matching.
"""
import sys
from pathlib import Path

import torch
import hydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "stackoverflow"
CFG_DIR = EXP / "configs"


def load_model(config_name, ckpt_name=None):
    with hydra.initialize_config_dir(config_dir=str(CFG_DIR), version_base="1.2"):
        cfg = hydra.compose(config_name=config_name)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    model = hydra.utils.instantiate(cfg.module)
    if ckpt_name is not None:
        ckpt_path = EXP / "checkpoints" / ckpt_name
        if ckpt_path.exists():
            sd = torch.load(ckpt_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            try:
                model.load_state_dict(sd)
                print(f"  loaded weights from {ckpt_name}")
            except Exception as e:
                print(f"  WARN: could not load weights ({e}); using fresh init")
        else:
            print(f"  WARN: no checkpoint {ckpt_name}, using fresh init")
    return model.eval(), cfg


def get_test_batch(cfg, n_seqs=4):
    dm = hydra.utils.instantiate(cfg.data_module)
    dm.setup("test")
    dl = dm.test_dataloader(rank=0, world_size=1)
    batch = next(iter(dl))
    x, _ = batch
    # Shrink to first n sequences for clarity.
    payload = {k: (v[:n_seqs] if hasattr(v, "shape") else v) for k, v in x.payload.items()}
    seq_lens = x.seq_lens[:n_seqs]
    from hotpp.data import PaddedBatch
    return PaddedBatch(payload, seq_lens, x.seq_names)


def stats(t, name="?"):
    if t is None:
        return f"{name}: None"
    if not isinstance(t, torch.Tensor):
        return f"{name}: {type(t).__name__} {t}"
    s = (
        f"shape={tuple(t.shape)} dtype={str(t.dtype).replace('torch.', '')} "
        f"min={t.float().min().item():.4f} max={t.float().max().item():.4f} "
        f"mean={t.float().mean().item():.4f} std={t.float().std().item():.4f}"
    )
    return f"{name}: {s}"


@torch.no_grad()
def trace_diffusion_gru(model, batch):
    print("\n" + "=" * 80)
    print("DIFFUSION_GRU (pure baseline)")
    print("=" * 80)
    print("Inputs:")
    for k, v in batch.payload.items():
        print("  " + stats(v, k))
    print(f"  seq_lens: {batch.seq_lens.tolist()}")

    # Forward: backbone + head
    outputs, states = model(batch, return_states=False)
    print(f"\nBackbone/head output:")
    print("  " + stats(outputs.payload, "head_out"))
    print(f"  -> condition_size for diffusion = {outputs.payload.shape[-1]}")

    loss = model._loss
    # Show diffusion fields used
    print(f"\nDiffusion fields (per-slot reconstruction targets): {loss._diffusion_fields if hasattr(loss, '_diffusion_fields') else 'inputs.fields'}")
    print(f"  K (predicted slots): {loss._k}")
    print(f"  generation_steps: {loss._generation_steps}")

    # Run prediction path: noise -> denoise -> decode
    mask = outputs.seq_len_mask.bool()
    cond = outputs.payload[mask]
    print(f"\nCondition tensor (after position mask): {stats(cond, 'cond')}")

    x = loss._noise(len(cond), device=cond.device, dtype=outputs.payload.dtype)
    print(f"  initial noise: {stats(x.payload, 'noise')}")

    for i in range(loss._generation_steps, 0, -1):
        x = loss._denoising_step(x, cond, i)
    print(f"  denoised latent: {stats(x.payload, 'denoised')}")

    decoded = loss._decoder(x)
    print(f"  decoded (per-slot): {stats(decoded.payload, 'decoded')}  -> P_base = {decoded.payload.shape[-1]}")

    # Predict via NextItemLoss
    preds = loss._next_item.predict_next(decoded, None,
                                         fields=loss._next_item._order,
                                         logits_fields_mapping={"labels": "labels_logits"})
    print(f"\nPredicted next per slot:")
    for k, v in preds.payload.items():
        print(f"  " + stats(v, k))


@torch.no_grad()
def trace_detection_hybrid(model, batch):
    print("\n" + "=" * 80)
    print("DETECTION_HYBRID (pure DeTPP)")
    print("=" * 80)
    print("Inputs:")
    for k, v in batch.payload.items():
        print("  " + stats(v, k))
    print(f"  seq_lens: {batch.seq_lens.tolist()}")

    outputs, states = model(batch, return_states=False)
    print(f"\nBackbone+ConditionalHead output:")
    print("  " + stats(outputs.payload, "head_out"))
    print(f"  -> shape (B, L, K*P_base): K*P_base = {outputs.payload.shape[-1]}")

    loss = model._loss
    print(f"\nDetectionLoss config:")
    print(f"  K: {loss._k}, prefetch_k: {loss._prefetch_k}, horizon: {loss._horizon}")
    print(f"  data_fields: {loss.data_fields}")
    print(f"  next_item_loss.input_size = P_base = {loss._next_item.input_size}")

    # Reshape to (B, L, K, P_base)
    b, l, kp = outputs.payload.shape
    reshaped = outputs.payload.view(b, l, loss._k, loss._next_item.input_size)
    print(f"\nReshape to (B, L, K, P_base): {tuple(reshaped.shape)}")
    print("  " + stats(reshaped, "reshaped_outputs"))

    # Show how matching extracts target windows.
    target_windows = loss.extract_structured_windows(batch)
    print(f"\nTarget windows for matching (horizon prefetch):")
    for k, v in target_windows.payload.items():
        print("  " + stats(v, k))


@torch.no_grad()
def trace_v3_hybrid(model, batch):
    print("\n" + "=" * 80)
    print("V3 (DIFFUSION-IN-LATENT + DETPP MATCHING)")
    print("=" * 80)
    print("Inputs:")
    for k, v in batch.payload.items():
        print("  " + stats(v, k))
    print(f"  seq_lens: {batch.seq_lens.tolist()}")

    outputs, states = model(batch, return_states=False)
    print(f"\nBackbone+head output (-> diffusion condition):")
    print("  " + stats(outputs.payload, "head_out"))
    print(f"  -> condition_size for diffusion = {outputs.payload.shape[-1]}")

    loss = model._loss
    print(f"\nV3 config:")
    print(f"  K: {loss._k}, horizon: {loss._horizon}, generation_steps: {loss._generation_steps}")
    print(f"  diffusion_fields (denoiser targets): {loss._diffusion_fields}")
    print(f"  embedder.output_size D: {loss._embedder.output_size}")
    print(f"  inner NextItemLoss.input_size P_base = {loss._next_item._next_item.input_size}")
    print(f"  per-slot decoder = self._decoder = {loss._decoder.__class__.__name__}")

    # Build horizon windows
    targets = loss._build_horizon_windows(batch)
    print(f"\nHorizon windows (B, L, K+1):")
    for k, v in targets.payload.items():
        print("  " + stats(v, k))
    pres = targets.payload["_presence"].float()
    print(f"  presence ratio per position (mean): {pres.mean().item():.3f}  -> "
          f"{pres.sum(-1).float().mean().item():.2f} valid events per K+1 window on avg")

    # Embed targets (with mask_embedding for absent slots)
    embeddings = loss._embed_targets(
        # Need to flatten first like in forward
        # targets seq_lens here = inputs.seq_lens, but _embed_targets expects flat
        # Actually we can just call it on the full (B, L, K+1) targets — _embedder works on seq dim 2 too?
        # Easier: replicate the forward path
        targets,
    )
    print(f"\nEmbedded targets (after mask_embedding override for absent slots):")
    print("  " + stats(embeddings.payload, "emb"))

    # Trace inference path (predict_next_k like)
    mask = outputs.seq_len_mask.bool()
    cond = outputs.payload[mask]
    print(f"\nCondition (after position mask): {stats(cond, 'cond')}")

    x = loss._noise(len(cond), device=cond.device, dtype=outputs.payload.dtype)
    print(f"  initial noise: {stats(x.payload, 'noise')}")
    for i in range(loss._generation_steps, 0, -1):
        x = loss._denoising_step(x, cond, i)
    print(f"  denoised latent: {stats(x.payload, 'denoised')}")
    decoded = loss._decoder(x)
    print(f"  decoded per-slot (V, K, P_base): {stats(decoded.payload, 'decoded')}")

    # Flatten -> scatter -> hand to DetectionLoss
    decoded_flat = decoded.payload.flatten(1)
    print(f"  flatten last dims -> (V, K*P_base): {stats(decoded_flat, 'flat')}")

    bsz, seq_len = outputs.shape
    det_payload = torch.zeros(
        bsz, seq_len, decoded_flat.shape[-1],
        dtype=decoded_flat.dtype, device=decoded_flat.device,
    )
    det_payload.masked_scatter_(mask.unsqueeze(-1), decoded_flat)
    from hotpp.data import PaddedBatch
    det_outputs = PaddedBatch(det_payload, outputs.seq_lens)
    print(f"\nFinal scatter -> (B, L, K*P_base) given to DetectionLoss:")
    print("  " + stats(det_outputs.payload, "det_outputs"))

    # DetectionLoss reshapes
    b, l = outputs.shape
    reshaped = det_outputs.payload.view(b, l, loss._next_item._k, loss._next_item._next_item.input_size)
    print(f"\nDetectionLoss reshape -> (B, L, K, P_base): {tuple(reshaped.shape)}")
    print("  " + stats(reshaped, "det_reshaped"))

    # Compare prediction stats per slot at one valid position to see slot diversity
    sample_pos = (batch.seq_lens[0].item() - 1, 0)  # last valid pos of seq 0
    b_idx, _ = 0, 0
    pos = batch.seq_lens[0].item() - 1
    pred_per_slot = reshaped[0, pos]  # (K, P_base)
    print(f"\nPer-slot diversity at sample (b=0, l={pos}):")
    print(f"  pred_per_slot: {stats(pred_per_slot, 'pred')}")
    # Variance across slots in predictions - low variance = degenerate matching
    slot_var = pred_per_slot.float().var(dim=0)
    print(f"  variance ACROSS K slots (per P_base channel):")
    print(f"    " + stats(slot_var, 'slot_var'))


def main():
    print("Loading test batch from default data module...")
    _, cfg = load_model("diffusion_gru_detpp_v3", "diffusion_gru_detpp_v3.ckpt")
    batch = get_test_batch(cfg, n_seqs=4)
    print(f"Test batch: {len(batch.payload)} fields, {len(batch.seq_lens)} seqs")

    # 1) Diffusion baseline
    model_diff, _ = load_model("diffusion_gru", "diffusion_gru.ckpt")
    trace_diffusion_gru(model_diff, batch)

    # 2) DeTPP baseline
    model_det, _ = load_model("detection_hybrid", "detection_hybrid.ckpt")
    trace_detection_hybrid(model_det, batch)

    # 3) V3 hybrid
    model_v3, _ = load_model("diffusion_gru_detpp_v3", "diffusion_gru_detpp_v3.ckpt")
    trace_v3_hybrid(model_v3, batch)


if __name__ == "__main__":
    main()
