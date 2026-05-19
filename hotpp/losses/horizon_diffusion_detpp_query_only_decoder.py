"""Pure hybrid: per-slot decoder uses only [z_k, q_k] (no h_p in decoder).

Architecture aligned with the original research statement: diffusion produces
K per-slot latents z_k from history h_p (h_p serves only as the diffusion
condition); the per-slot decoder then computes shared_MLP([z_k; q_k]) -> P
for each slot. Learnable queries q_k carry slot identity (as in DeTPP's
ConditionalHead). The history representation h_p is NOT fed into the
decoder directly --- it is already encoded in z_k via the diffusion process.

Differences from V44 (HorizonDiffusionLossDetppAllCompsInDecoder):
- Decoder input does NOT include the context h_p; only [z_k, q_k].

Otherwise identical to V44/V42:
- Diffusion supervision without horizon mask (inherited from V42): events
  past horizon are kept as real embeddings, and horizon filtering is
  performed only at the matching step.
- Hungarian matching between K decoded slots and T ground-truth events.
"""
import time
import torch

from hotpp.data import PaddedBatch
from ..fields import PRESENCE
from .detection import DetectionLoss
from .horizon_diffusion_fixed_perslot_decoder import HorizonDiffusionLossFixedPerslotDecoder
from .horizon_diffusion_v42 import HorizonDiffusionLossV42


class QueryOnlyHead(torch.nn.Module):
    """Per-slot decoder that fuses (latent_k, query_k) only.

    Forward expects ``x`` to be a PaddedBatch of latents shape (V, K, D_latent).
    The shared MLP processes concatenated (latent, query) per slot; no
    external context is required.
    """

    def __init__(
        self,
        latent_dim,
        output_size,
        k,
        query_size=64,
        hidden_dims=(128, 256),
        use_batch_norm=True,
        query_init_scale=0.02,
    ):
        super().__init__()
        self.queries = torch.nn.Parameter(
            torch.randn(k, query_size) * float(query_init_scale)
        )
        self.k = k
        self.output_size = output_size

        layers = []
        last_dim = latent_dim + query_size
        if use_batch_norm:
            layers.append(torch.nn.BatchNorm1d(last_dim))
        for dim in hidden_dims or []:
            layers.append(torch.nn.Linear(last_dim, dim, bias=not use_batch_norm))
            if use_batch_norm:
                layers.append(torch.nn.BatchNorm1d(dim))
            layers.append(torch.nn.ReLU())
            last_dim = dim
        layers.append(torch.nn.Linear(last_dim, output_size))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        latents = x.payload  # (V, K, D_latent)
        v = latents.shape[0]
        if latents.shape[1] != self.k:
            raise ValueError(f"Expected K={self.k}, got {latents.shape[1]}")

        queries = self.queries.unsqueeze(0).expand(v, -1, -1)  # (V, K, D_query)
        combined = torch.cat([latents, queries], dim=-1)  # (V, K, D_latent + D_query)
        flat = combined.flatten(0, 1)  # (V*K, D_latent + D_query)
        out = self.mlp(flat)
        out = out.view(v, self.k, self.output_size)
        return PaddedBatch(out, x.seq_lens)


class HorizonDiffusionLossDetppQueryOnlyDecoder(HorizonDiffusionLossV42):
    """V42 supervision + per-slot decoder with input [z_k; q_k] only."""

    def __init__(
        self,
        *args,
        query_size=64,
        query_hidden_dims=(128, 256),
        query_use_batch_norm=True,
        query_init_scale=0.02,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if isinstance(self._next_item, DetectionLoss):
            inner_input_size = self._next_item._next_item.input_size
            self._decoder = QueryOnlyHead(
                latent_dim=self._embedder.output_size,
                output_size=inner_input_size,
                k=self._k,
                query_size=query_size,
                hidden_dims=tuple(query_hidden_dims) if query_hidden_dims else None,
                use_batch_norm=query_use_batch_norm,
                query_init_scale=query_init_scale,
            )
        # No override of _diffusion_loss / predict_next_k needed: V3's versions
        # (inherited via V42) call self._decoder(x) without a context kwarg,
        # which matches QueryOnlyHead.forward signature.


class HorizonDiffusionLossDetppQueryOnlyDecoderHorizonMask(
    HorizonDiffusionLossFixedPerslotDecoder
):
    """Same QueryOnlyHead decoder, but with V3-style supervision.

    Differences from HorizonDiffusionLossDetppQueryOnlyDecoder:
    - Inherits ``_build_horizon_windows`` from V3 (= FixedPerslotDecoder).
      Events past horizon get presence=0 and are replaced with mask_embedding
      at the diffusion target stage; only events within horizon stay as
      real embeddings.

    Otherwise identical: per-slot decoder takes [z_k; q_k] (no h_p).
    """

    def __init__(
        self,
        *args,
        query_size=64,
        query_hidden_dims=(128, 256),
        query_use_batch_norm=True,
        query_init_scale=0.02,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if isinstance(self._next_item, DetectionLoss):
            inner_input_size = self._next_item._next_item.input_size
            self._decoder = QueryOnlyHead(
                latent_dim=self._embedder.output_size,
                output_size=inner_input_size,
                k=self._k,
                query_size=query_size,
                hidden_dims=tuple(query_hidden_dims) if query_hidden_dims else None,
                use_batch_norm=query_use_batch_norm,
                query_init_scale=query_init_scale,
            )


class LatentContextHead(torch.nn.Module):
    """Per-slot decoder that fuses only (latent_k, context) --- no queries.

    Forward expects ``x`` to be a PaddedBatch of latents shape (V, K, D_latent)
    and ``context`` (V, D_context) passed via ``forward(x, context=...)``.
    The shared MLP processes concatenated (latent, context) per slot. There
    are no learnable per-slot queries; slot identity comes purely from the
    differences between z_k produced by the diffusion process.
    """

    def __init__(
        self,
        latent_dim,
        context_dim,
        output_size,
        k,
        hidden_dims=(128, 256),
        use_batch_norm=True,
    ):
        super().__init__()
        self.k = k
        self.output_size = output_size
        self.context_dim = context_dim

        layers = []
        last_dim = latent_dim + context_dim
        if use_batch_norm:
            layers.append(torch.nn.BatchNorm1d(last_dim))
        for dim in hidden_dims or []:
            layers.append(torch.nn.Linear(last_dim, dim, bias=not use_batch_norm))
            if use_batch_norm:
                layers.append(torch.nn.BatchNorm1d(dim))
            layers.append(torch.nn.ReLU())
            last_dim = dim
        layers.append(torch.nn.Linear(last_dim, output_size))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x, context=None):
        latents = x.payload  # (V, K, D_latent)
        v = latents.shape[0]
        if latents.shape[1] != self.k:
            raise ValueError(f"Expected K={self.k}, got {latents.shape[1]}")
        if context is None:
            raise ValueError("LatentContextHead requires `context` arg.")
        if context.shape != (v, self.context_dim):
            raise ValueError(
                f"context shape {tuple(context.shape)} != expected {(v, self.context_dim)}"
            )

        context_per_slot = context.unsqueeze(1).expand(-1, self.k, -1)  # (V, K, D_context)
        combined = torch.cat([latents, context_per_slot], dim=-1)
        flat = combined.flatten(0, 1)
        out = self.mlp(flat).view(v, self.k, self.output_size)
        return PaddedBatch(out, x.seq_lens)


class LatentOnlyHead(torch.nn.Module):
    """Per-slot decoder that uses only the diffusion latent z_k.

    Forward expects ``x`` to be a PaddedBatch of latents shape (V, K, D_latent).
    A shared MLP is applied per slot. There are no learnable queries and no
    external context: slot identity comes purely from differences between
    z_k produced by the diffusion process.
    """

    def __init__(
        self,
        latent_dim,
        output_size,
        k,
        hidden_dims=(128, 256),
        use_batch_norm=True,
    ):
        super().__init__()
        self.k = k
        self.output_size = output_size

        layers = []
        last_dim = latent_dim
        if use_batch_norm:
            layers.append(torch.nn.BatchNorm1d(last_dim))
        for dim in hidden_dims or []:
            layers.append(torch.nn.Linear(last_dim, dim, bias=not use_batch_norm))
            if use_batch_norm:
                layers.append(torch.nn.BatchNorm1d(dim))
            layers.append(torch.nn.ReLU())
            last_dim = dim
        layers.append(torch.nn.Linear(last_dim, output_size))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        latents = x.payload  # (V, K, D_latent)
        v = latents.shape[0]
        if latents.shape[1] != self.k:
            raise ValueError(f"Expected K={self.k}, got {latents.shape[1]}")
        flat = latents.flatten(0, 1)
        out = self.mlp(flat).view(v, self.k, self.output_size)
        return PaddedBatch(out, x.seq_lens)


class HorizonDiffusionLossDetppLatentOnlyDecoder(HorizonDiffusionLossV42):
    """V42 supervision + per-slot decoder with input [z_k] only (no queries, no context).

    Forward path: V3's _diffusion_loss / predict_next_k (inherited via V42)
    call self._decoder(x) without a context kwarg, which matches
    LatentOnlyHead.forward.
    """

    def __init__(
        self,
        *args,
        hidden_dims=(128, 256),
        use_batch_norm=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if isinstance(self._next_item, DetectionLoss):
            inner_input_size = self._next_item._next_item.input_size
            self._decoder = LatentOnlyHead(
                latent_dim=self._embedder.output_size,
                output_size=inner_input_size,
                k=self._k,
                hidden_dims=tuple(hidden_dims) if hidden_dims else None,
                use_batch_norm=use_batch_norm,
            )


class HorizonDiffusionLossDetppLatentContextDecoder(
    HorizonDiffusionLossV42
):
    """V42 supervision + per-slot decoder with input [z_k; h_p] (no queries).

    Reuses V44's _diffusion_loss / predict_next_k (which call
    self._decoder(x, context=conditions)) by importing them via inheritance
    indirection: directly inherits from V42 and copies V44's forward
    methods to pass context to the decoder.
    """

    def __init__(
        self,
        *args,
        hidden_dims=(128, 256),
        use_batch_norm=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if isinstance(self._next_item, DetectionLoss):
            inner_input_size = self._next_item._next_item.input_size
            context_dim = self._denoiser.condition_size
            self._decoder = LatentContextHead(
                latent_dim=self._embedder.output_size,
                context_dim=context_dim,
                output_size=inner_input_size,
                k=self._k,
                hidden_dims=tuple(hidden_dims) if hidden_dims else None,
                use_batch_norm=use_batch_norm,
            )

    # Reuse the context-passing _diffusion_loss / predict_next_k from V44.
    # They call self._decoder(x, context=conditions); LatentContextHead
    # accepts that signature. We import V44 lazily to avoid a circular
    # import at module load time, then bind its methods.
    def _diffusion_loss(self, conditions, targets,
                        det_inputs=None, det_mask=None, det_lengths=None):
        from .horizon_diffusion_detpp_all_comps_in_decoder import (
            HorizonDiffusionLossDetppAllCompsInDecoder,
        )
        return HorizonDiffusionLossDetppAllCompsInDecoder._diffusion_loss(
            self, conditions, targets,
            det_inputs=det_inputs, det_mask=det_mask, det_lengths=det_lengths,
        )

    def predict_next_k(self, outputs, states,
                       fields=None, logits_fields_mapping=None):
        from .horizon_diffusion_detpp_all_comps_in_decoder import (
            HorizonDiffusionLossDetppAllCompsInDecoder,
        )
        return HorizonDiffusionLossDetppAllCompsInDecoder.predict_next_k(
            self, outputs, states,
            fields=fields, logits_fields_mapping=logits_fields_mapping,
        )


class HorizonDiffusionLossDetppQueryOnlyDecoderNoPadding(
    HorizonDiffusionLossDetppQueryOnlyDecoder
):
    """Same QueryOnlyHead decoder + V42 supervision, but NO padding substitution.

    Differences from HorizonDiffusionLossDetppQueryOnlyDecoder:
    - ``_embed_targets`` is overridden to skip the mask_embedding substitution.
      All K slots use the raw embedder output, including positions past the
      end of the original sequence (which carry placeholder pad-values:
      timestamps = current + horizon + 1, labels = 0).

    The diffusion model thus learns to reconstruct embedder(pad_values) at
    out-of-sequence positions, instead of a learnable mask vector.
    """

    def _embed_targets(self, targets):
        embed_inputs = PaddedBatch(
            {k: targets.payload[k] for k in self._diffusion_fields},
            targets.seq_lens,
            set(self._diffusion_fields),
        )
        embeddings = self._embedder(self._compute_time_deltas(embed_inputs))
        embeddings = PaddedBatch(
            embeddings.payload[:, 1:],
            (embeddings.seq_lens - 1).clip(min=0),
        )
        # Skip mask_embedding substitution: keep raw embedder output for all K slots.
        return embeddings


class HorizonDiffusionLossDetppQueryOnlyDecoderMatchingPadding(
    HorizonDiffusionLossDetppQueryOnlyDecoder
):
    """QueryOnly decoder + matching-padding for the diffusion target.

    Per training step (after warmup_steps):
      1. Predict via diffusion (full reverse process, no_grad), using QueryOnlyHead.
      2. Hungarian matching between K predictions and T GT events.
      3. Realign the diffusion target windows: matched slots get real events,
         unmatched slots get mask_embedding via presence=0.
      4. Standard noise/denoise/update step.

    Decoder is QueryOnlyHead (input = [z_k; q_k], no context arg) ---
    matches m1 (HorizonDiffusionLossMatchingPadding) on V3 base, but uses
    QueryOnlyHead instead of plain Head per-slot.
    """

    def __init__(self, *args, warmup_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._warmup_steps = int(warmup_steps)
        self.register_buffer("_align_step_count", torch.zeros(1, dtype=torch.long))

    def _predict_and_match_for_alignment(self, inputs, outputs):
        """Full diffusion inference (no_grad) + Hungarian matching."""
        targets_for_lengths = self._build_horizon_windows(inputs)
        lengths = targets_for_lengths.seq_lens.clone()
        det_inputs = PaddedBatch(
            {
                k: (
                    v[:, :targets_for_lengths.shape[1]].clone()
                    if (k in inputs.seq_names) and isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in inputs.payload.items()
            },
            lengths,
            inputs.seq_names,
        )

        mask = outputs.seq_len_mask.bool()
        conditions = outputs.payload[mask]                              # (V, D)
        batch_size = len(conditions)
        x = self._noise(
            batch_size,
            device=conditions.device,
            dtype=outputs.payload.dtype,
        )
        for step in range(self._generation_steps, 0, -1):
            x = self._denoising_step(x, conditions, step)
        # QueryOnlyHead is called WITHOUT context arg.
        decoded = self._decoder(x)                                      # (V, K, P_base)
        decoded_flat = decoded.payload.flatten(1)

        bsz, seq_len = outputs.shape
        det_payload = torch.zeros(
            bsz, seq_len, decoded_flat.shape[-1],
            dtype=decoded_flat.dtype, device=decoded_flat.device,
        )
        det_payload.masked_scatter_(mask.unsqueeze(-1), decoded_flat)

        P_base = decoded.payload.shape[-1]
        det_outputs_4d = PaddedBatch(
            det_payload.reshape(bsz, seq_len, self._k, P_base),
            outputs.seq_lens,
        )
        det_targets = self._next_item.extract_structured_windows(det_inputs)
        matches_pb, _, _ = self._next_item.match_targets(
            det_outputs_4d, det_targets,
        )
        return matches_pb.payload                                       # (B, L, K)

    def _realign_targets_by_matches(self, targets, matches):
        """Same as m1: gather windows according to matches+1 (anchor index 0
        stays untouched). Unmatched slots get presence=0."""
        gather_idx = (matches.clamp(min=0) + 1).long()                  # (B, L, K)
        unmatched = (matches < 0)

        new_payload = {}
        for name, raw in targets.payload.items():
            gathered = torch.gather(raw, dim=2, index=gather_idx)       # (B, L, K)
            if name == PRESENCE:
                gathered = torch.where(
                    unmatched, torch.zeros_like(gathered), gathered,
                )
            new_window = torch.cat([raw[:, :, :1], gathered], dim=2)    # (B, L, K+1)
            new_payload[name] = new_window
        return PaddedBatch(new_payload, targets.seq_lens, targets.seq_names)

    def forward(self, inputs, outputs, states):
        if not isinstance(self._next_item, DetectionLoss):
            return super().forward(inputs, outputs, states)

        t0 = time.perf_counter()
        targets = self._build_horizon_windows(inputs)

        do_realign = self.training and (self._align_step_count.item() > self._warmup_steps)
        if self.training:
            self._align_step_count += 1

        if do_realign:
            with torch.no_grad():
                matches = self._predict_and_match_for_alignment(inputs, outputs)
            targets = self._realign_targets_by_matches(targets, matches)

        # The rest mirrors V3's forward; uses self._decoder(x) without context
        # (inherited via QueryOnlyDecoder -> V42 -> V3 _diffusion_loss).
        lengths = targets.seq_lens.clone()
        outputs = PaddedBatch(outputs.payload[:, :targets.shape[1]], lengths)
        det_inputs = PaddedBatch(
            {
                k: (
                    v[:, :targets.shape[1]].clone()
                    if (k in inputs.seq_names) and isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in inputs.payload.items()
            },
            lengths,
            inputs.seq_names,
        )

        if self._loss_step > 1:
            lengths = (lengths - self._loss_step - 1).div(self._loss_step, rounding_mode="floor").clip(min=-1) + 1
            targets = PaddedBatch(
                {k: v[:, self._loss_step::self._loss_step] for k, v in targets.payload.items()},
                lengths,
                targets.seq_names,
            )
            outputs = PaddedBatch(outputs.payload[:, self._loss_step::self._loss_step], lengths)
            det_inputs = PaddedBatch(
                {
                    k: (
                        v[:, self._loss_step::self._loss_step].clone()
                        if (k in det_inputs.seq_names) and isinstance(v, torch.Tensor)
                        else v
                    )
                    for k, v in det_inputs.payload.items()
                },
                lengths,
                det_inputs.seq_names,
            )

        mask = targets.seq_len_mask.bool()
        flat_lengths = torch.full([mask.sum().item()], self._k + 1, device=mask.device, dtype=torch.long)
        targets = PaddedBatch(
            {k: v[mask] for k, v in targets.payload.items()},
            flat_lengths,
            targets.seq_names,
        )
        flat_outputs = outputs.payload[:, :mask.shape[1]][mask]

        losses, metrics = self._diffusion_loss(
            conditions=flat_outputs,
            targets=targets,
            det_inputs=det_inputs,
            det_mask=mask,
            det_lengths=lengths,
        )
        metrics["perf_build_windows_s"] = time.perf_counter() - t0
        metrics["perf_num_windows"] = float(len(targets))
        metrics["perf_presence_ratio"] = float(
            targets.payload[PRESENCE][:, 1:].float().mean().item()
        )
        metrics["matching_padding_active"] = float(do_realign)
        return losses, metrics


class HorizonDiffusionLossDetppQueryOnlyDecoderFarTarget(
    HorizonDiffusionLossDetppQueryOnlyDecoder
):
    """QueryOnly decoder + diffusion target on far events (delta in [min, max]).

    Same as a5 (HorizonDiffusionLossDetppAllCompsInDecoderFarTarget) but
    on top of QueryOnlyDecoder (no h_p in decoder, only [z_k; q_k]).

    Detection loss still operates on the near horizon [t, t+H]; diffusion
    is supervised on a non-overlapping far window [t + min_delta, t + max_delta].
    Latents z_k thus encode long-range patterns and feed them to the decoder
    via the per-slot diffusion output.
    """

    def __init__(
        self,
        *args,
        diffusion_target_min_delta=10.0,
        diffusion_target_max_delta=30.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._diff_target_min = float(diffusion_target_min_delta)
        self._diff_target_max = float(diffusion_target_max_delta)

    def _build_horizon_windows(self, inputs):
        """Build K+1 target window: target events at delta in [min, max]."""
        b, l = inputs.shape
        k1 = self._k + 1
        device = inputs.device

        base = torch.arange(l, device=device)[None, :, None]
        offs = torch.arange(k1, device=device)[None, None, :]
        valid_index = base + offs < inputs.seq_lens[:, None, None]

        current_timestamps = inputs.payload[self._timestamps_field]
        rolled_timestamps = torch.stack(
            [current_timestamps.roll(-i, 1) for i in range(k1)],
            dim=2,
        )
        delta = rolled_timestamps - current_timestamps.unsqueeze(2)
        # Long-range filter: keep events whose delta falls in [min, max].
        long_range_valid = (delta >= self._diff_target_min) & (delta < self._diff_target_max)
        long_range_valid[:, :, 0] = True  # keep current as i=0 anchor
        valid = valid_index & long_range_valid

        windows = {}
        for name in self._diffusion_fields:
            values = inputs.payload[name]
            window = torch.stack([values.roll(-i, 1) for i in range(k1)], dim=2)
            if name == self._timestamps_field:
                pad_value = current_timestamps.unsqueeze(2) + self._diff_target_max + 1
            else:
                pad_value = torch.zeros_like(window)
            window = torch.where(valid, window, pad_value)
            windows[name] = window

        windows[PRESENCE] = valid.long()
        return PaddedBatch(windows, inputs.seq_lens, set(self._diffusion_fields) | {PRESENCE})


class HorizonDiffusionLossDetppQueryOnlyDecoderMatchingRealign(
    HorizonDiffusionLossDetppQueryOnlyDecoderMatchingPadding
):
    """Matching-padding variant that keeps unmatched slots positionally supervised.

    Difference from HorizonDiffusionLossDetppQueryOnlyDecoderMatchingPadding:
    - Matched slots: receive their matched GT event (same as parent).
    - Unmatched slots: KEEP their original positional event from the
      sequence (instead of being replaced by mask_embedding with presence=0).

    Motivation: the parent class's "unmatched → mask" rule undoes the
    advantage of supervising all K slots on real events. With this variant,
    matching only realigns the supervision targets; it does not remove
    real-event supervision for any slot.
    """

    def _realign_targets_by_matches(self, targets, matches):
        """Realign matched slots; keep unmatched slots positional.

        matched slot k     -> gather targets[matches[k] + 1]   (real event from matching)
        unmatched slot k   -> gather targets[k + 1]            (positional, original supervision)

        ``targets`` is a PaddedBatch with payload {field: (B, L, K+1)};
        slot 0 is the anchor (current event), slots 1..K are the K future
        events from the sequence.
        """
        bsz, seq_len, _k1 = next(iter(targets.payload.values())).shape
        device = matches.device
        K = self._k

        # For matched slots, use matches[k] + 1 (where the matched event sits in K+1 windows).
        matched_idx = (matches.clamp(min=0) + 1).long()                      # (B, L, K)
        # For unmatched slots, keep positional index (slot k -> position k+1).
        positional_idx = (
            torch.arange(1, K + 1, device=device, dtype=torch.long)
            .expand(bsz, seq_len, K)
        )
        unmatched = (matches < 0)
        gather_idx = torch.where(unmatched, positional_idx, matched_idx)

        new_payload = {}
        for name, raw in targets.payload.items():
            gathered = torch.gather(raw, dim=2, index=gather_idx)            # (B, L, K)
            new_window = torch.cat([raw[:, :, :1], gathered], dim=2)         # (B, L, K+1)
            new_payload[name] = new_window
        return PaddedBatch(new_payload, targets.seq_lens, targets.seq_names)


class HorizonDiffusionLossDetppQueryOnlyDecoderRandomPad(
    HorizonDiffusionLossDetppQueryOnlyDecoder
):
    """QueryOnly decoder + random padding insertion in the diffusion target.

    Implements the advisor's request:
    "сделать encoder так, чтобы он вставлял случайные паддинги в GT цепочку
    и энкодил паддинги в некоторый обучаемый вектор".

    On every training step, with probability ``padding_prob`` per K slot,
    a real event embedding is replaced by the existing learnable
    ``_mask_embedding`` (the same one used for deterministic V42 padding).
    Affected slots get presence=0 *only inside the diffusion target*; the
    DetectionLoss matching is computed on raw inputs and is not affected
    --- so the K decoded slots still have to predict ALL real horizon
    events, even the ones that were random-masked in the diffusion target.
    The effect is a denoising-autoencoder-like regularizer on the
    diffusion branch (the decoder must extract event predictions even
    when the diffusion target was partially erased to mask).

    Active only during training. At inference the V42 supervision is used
    as-is, so deterministic paddings (out-of-sequence positions) keep
    their behavior.
    """

    def __init__(self, *args, padding_prob=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._random_padding_prob = float(padding_prob)

    def _embed_targets(self, targets):
        embed_inputs = PaddedBatch(
            {k: targets.payload[k] for k in self._diffusion_fields},
            targets.seq_lens,
            set(self._diffusion_fields),
        )
        embeddings = self._embedder(self._compute_time_deltas(embed_inputs))
        embeddings = PaddedBatch(
            embeddings.payload[:, 1:],
            (embeddings.seq_lens - 1).clip(min=0),
        )

        presence = targets.payload[PRESENCE][:, 1:].bool()

        # Random padding: with prob padding_prob, randomly mask out some real
        # event slots. Only during training; only on slots that were real
        # (presence=1).  Modifies the LOCAL ``presence`` view used for the
        # mask substitution below; ``targets.payload[PRESENCE]`` is left
        # unchanged so that DetectionLoss matching keeps using the original
        # K real events.
        if self.training and self._random_padding_prob > 0:
            random_drop = (
                torch.rand(presence.shape, device=presence.device)
                < self._random_padding_prob
            ) & presence
            presence = presence & ~random_drop

        embeddings.payload = torch.where(
            presence.unsqueeze(-1),
            embeddings.payload,
            self._mask_embedding.view(1, 1, -1),
        )
        return embeddings
