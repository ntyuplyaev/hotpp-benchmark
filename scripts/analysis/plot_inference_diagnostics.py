"""Export and plot inference-time diagnostics for DeTPP-style head analysis.

The script loads canonical configs/checkpoints, runs horizon generation on a
small number of test batches, and summarizes slot/head behavior. It is intended
for paper-style figures, not for replacing the full benchmark evaluation.

Example:
  python scripts/analysis/plot_inference_diagnostics.py --datasets stackoverflow amazon retweet --formats html,pdf
"""
from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import hydra
import plotly.graph_objects as go
import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hotpp.fields import LABELS_LOGITS, PRESENCE, PRESENCE_PROB  # noqa: E402


PAPER_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Times New Roman", size=18, color="#1f1f1f"),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=72, r=28, t=58, b=105),
)


def paper_layout(**overrides) -> dict:
    layout = dict(PAPER_LAYOUT)
    layout.update(overrides)
    return layout


@dataclass(frozen=True)
class RunSpec:
    name: str
    label: str
    color: str


DATASETS = {
    "stackoverflow": "StackOverflow",
    "amazon": "Amazon",
    "retweet": "Retweet",
}

RUNS = [
    RunSpec("b1_pure_diffusion_v1", "Diffusion", "#7e57c2"),
    RunSpec("b2_detpp_baseline_v1", "DeTPP", "#2a78b8"),
    RunSpec("a6_query_only_decoder_v3_paper_detection", "Hybrid (latent+query)", "#e08a3c"),
    RunSpec("a6_query_only_decoder_v8_full_components_paper_detection", "Hybrid (+context)", "#6f9fd8"),
    RunSpec("a6_query_only_decoder_v11_matching_padding_paper_detection", "Hybrid (matching padding)", "#b590d8"),
    RunSpec("a6_query_only_decoder_v12_far_target_paper_detection", "Hybrid (far-target)", "#3fa37b"),
]


@contextmanager
def working_directory(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--runs", nargs="+", default=[run.name for run in RUNS])
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "text_final" / "figures")
    parser.add_argument("--summary-dir", type=Path, default=ROOT / "artifacts" / "paper_diagnostics")
    parser.add_argument("--formats", default="pdf")
    parser.add_argument("--no-export", action="store_true", help="Only plot existing summaries.")
    return parser.parse_args()


def save_figure(fig: go.Figure, out_dir: Path, stem: str, formats: Iterable[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt = fmt.strip().lower()
        if not fmt:
            continue
        path = out_dir / f"{stem}.{fmt}"
        if fmt == "html":
            fig.write_html(str(path), include_plotlyjs="cdn")
            print(f"saved {path}")
            continue
        fig.write_image(str(path), format=fmt)
        print(f"saved {path}")


def apply_axis_style(fig: go.Figure) -> None:
    fig.update_xaxes(showline=True, linewidth=1, linecolor="#222", ticks="outside")
    fig.update_yaxes(showline=True, linewidth=1, linecolor="#222", ticks="outside",
                     gridcolor="#e6e6e6", zeroline=False, tickformat=".2f")


def load_model_and_dm(dataset: str, run_name: str, batch_size: int, device: str):
    exp_dir = ROOT / "experiments" / dataset
    cfg_dir, config_name, ckpt_path = resolve_run_paths(dataset, run_name)
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    with working_directory(exp_dir):
        with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
            cfg = compose(config_name=config_name)
        OmegaConf.set_struct(cfg, False)
        cfg.data_module.batch_size = batch_size
        cfg.data_module.num_workers = 0
        for key in ("train_path", "val_path", "test_path"):
            if cfg.data_module.get(key) is not None:
                cfg.data_module[key] = str(exp_dir / cfg.data_module[key])
        model = hydra.utils.instantiate(cfg.module)
        dm = hydra.utils.instantiate(cfg.data_module)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"{dataset}/{run_name}: missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)
    model.eval()
    return model, dm, cfg


def resolve_run_paths(dataset: str, run_name: str) -> tuple[Path, str, Path]:
    exp_dir = ROOT / "experiments" / dataset
    cfg_dir = exp_dir / "configs" / "canonical"
    ckpt_path = exp_dir / "canonical" / "checkpoints" / f"{run_name}.ckpt"
    if ckpt_path.exists():
        return cfg_dir, run_name, ckpt_path
    if dataset == "stackoverflow" and run_name == "b1_pure_diffusion_v1":
        return exp_dir / "configs", "diffusion_gru", exp_dir / "checkpoints" / "diffusion_gru.ckpt"
    return cfg_dir, run_name, ckpt_path


def batch_to_device(batch, device: str):
    x, y = batch
    return x.to(device), y


def collect_run_summary(dataset: str, run: RunSpec, split: str, max_batches: int,
                        batch_size: int, device: str) -> dict:
    model, dm, cfg = load_model_and_dm(dataset, run.name, batch_size, device)
    metric = hydra.utils.instantiate(cfg.metric)
    metric.to(device)
    loader = dm.test_dataloader(rank=0, world_size=1) if split == "test" else dm.val_dataloader(rank=0, world_size=1)

    slot_delta_sum = None
    slot_delta_count = None
    slot_prob_sum = None
    slot_prob_count = None
    slot_active_sum = None
    slot_valid_count = None
    slot_label_counts = None
    n_windows = 0
    n_sequences = 0
    example = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            x, _ = batch_to_device(batch, device)
            outputs, states = model(x, return_states="full" if model._need_states else False)
            indices = metric.select_horizon_indices(x.seq_lens)
            sequences = model.generate_sequences(x, indices)

            init_times = x.payload[model._timestamps_field].take_along_dim(indices.payload, 1)
            seq_mask = indices.seq_len_mask.bool()
            timestamps = sequences.payload[model._timestamps_field]
            labels = sequences.payload[model._labels_field]
            deltas = timestamps - init_times.unsqueeze(2)
            k = deltas.shape[2]
            valid = seq_mask.unsqueeze(2).expand_as(deltas)
            if PRESENCE in sequences.payload:
                presence = sequences.payload[PRESENCE].bool()
            else:
                presence = torch.ones_like(valid)
            if PRESENCE_PROB in sequences.payload:
                probs = sequences.payload[PRESENCE_PROB].float()
            else:
                probs = presence.float()

            active = torch.logical_and(valid, presence)
            active_horizon = torch.logical_and(active, deltas < float(cfg.metric.horizon))

            if slot_delta_sum is None:
                slot_delta_sum = torch.zeros(k, device=device)
                slot_delta_count = torch.zeros(k, device=device)
                slot_prob_sum = torch.zeros(k, device=device)
                slot_prob_count = torch.zeros(k, device=device)
                slot_active_sum = torch.zeros(k, device=device)
                slot_valid_count = torch.zeros(k, device=device)
                slot_label_counts = torch.zeros(k, int(cfg.num_classes), device=device)

            slot_delta_sum += (deltas * valid).sum((0, 1))
            slot_delta_count += valid.sum((0, 1)).float()
            slot_prob_sum += (probs * valid).sum((0, 1))
            slot_prob_count += valid.sum((0, 1)).float()
            slot_active_sum += active.sum((0, 1)).float()
            slot_valid_count += valid.sum((0, 1)).float()

            for slot in range(k):
                slot_labels = labels[:, :, slot][active_horizon[:, :, slot]]
                if len(slot_labels) > 0:
                    counts = torch.bincount(slot_labels.clip(min=0, max=int(cfg.num_classes) - 1),
                                            minlength=int(cfg.num_classes)).float()
                    slot_label_counts[slot] += counts

            n_windows += int(seq_mask.sum().item())
            n_sequences += len(x)

            if example is None and n_windows > 0:
                b_idx, i_idx = torch.nonzero(seq_mask, as_tuple=False)[0].tolist()
                logits = sequences.payload.get(LABELS_LOGITS)
                example = {
                    "history_index": int(indices.payload[b_idx, i_idx].detach().cpu().item()),
                    "predicted_delta": deltas[b_idx, i_idx].detach().cpu().tolist(),
                    "predicted_label": labels[b_idx, i_idx].detach().cpu().tolist(),
                    "predicted_presence": presence[b_idx, i_idx].float().detach().cpu().tolist(),
                    "predicted_presence_prob": probs[b_idx, i_idx].detach().cpu().tolist(),
                }
                if logits is not None:
                    example["max_label_score"] = logits[b_idx, i_idx].amax(-1).detach().cpu().tolist()

    if slot_delta_sum is None:
        raise RuntimeError(f"No windows exported for {dataset}/{run.name}")

    mean_delta = (slot_delta_sum / slot_delta_count.clamp(min=1)).detach().cpu()
    mean_prob = (slot_prob_sum / slot_prob_count.clamp(min=1)).detach().cpu()
    active_ratio = (slot_active_sum / slot_valid_count.clamp(min=1)).detach().cpu()
    label_counts = slot_label_counts.detach().cpu()
    label_probs = label_counts / label_counts.sum(1, keepdim=True).clamp(min=1)
    label_entropy = -(label_probs * label_probs.clamp(min=1e-6).log()).sum(1)
    order = torch.argsort(mean_delta)

    return {
        "dataset": dataset,
        "dataset_title": DATASETS[dataset],
        "run": run.name,
        "label": run.label,
        "color": run.color,
        "split": split,
        "max_batches": max_batches,
        "batch_size": batch_size,
        "num_sequences": n_sequences,
        "num_windows": n_windows,
        "horizon": float(cfg.metric.horizon),
        "k": int(len(mean_delta)),
        "sorted_slot": [int(v) + 1 for v in order.tolist()],
        "mean_delta": [float(mean_delta[i]) for i in order],
        "mean_presence_probability": [float(mean_prob[i]) for i in order],
        "active_ratio": [float(active_ratio[i]) for i in order],
        "label_entropy": [float(label_entropy[i]) for i in order],
        "label_counts": [[float(v) for v in label_counts[i].tolist()] for i in order],
        "example": example,
    }


def write_summary(summary: dict, summary_dir: Path) -> Path:
    path = summary_dir / summary["dataset"] / f"{summary['run']}_{summary['split']}_inference.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(summary, fp, sort_keys=False, allow_unicode=False)
    print(f"saved {path}")
    return path


def read_summary(summary_dir: Path, dataset: str, run_name: str, split: str) -> dict | None:
    path = summary_dir / dataset / f"{run_name}_{split}_inference.yaml"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def plot_slot_summary(summary: dict) -> go.Figure:
    x = list(range(1, summary["k"] + 1))
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("Mean predicted time delta", "Predicted presence", "Predicted label entropy"),
    )
    fig.add_trace(go.Scatter(
        x=x,
        y=summary["mean_delta"],
        mode="lines+markers",
        line=dict(color=summary["color"], width=2.4),
        marker=dict(size=5),
        name="time delta",
        customdata=summary["sorted_slot"],
        hovertemplate="rank=%{x}<br>slot=%{customdata}<br>delta=%{y:.2f}<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=summary["horizon"], row=1, col=1, line_color="#222", line_dash="dash",
                  line_width=1.2, annotation_text="horizon", annotation_position="top right")
    fig.add_trace(go.Scatter(
        x=x,
        y=summary["mean_presence_probability"],
        mode="lines+markers",
        line=dict(color="#e08a3c", width=2.2),
        marker=dict(size=5),
        name="presence probability",
        hovertemplate="rank=%{x}<br>prob=%{y:.2f}<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=x,
        y=summary["active_ratio"],
        mode="lines",
        line=dict(color="#555", width=1.5, dash="dot"),
        name="active ratio",
        hovertemplate="rank=%{x}<br>active=%{y:.2f}<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=x,
        y=summary["label_entropy"],
        marker=dict(color="#3fa37b", line=dict(color="#333", width=0.4)),
        name="label entropy",
        hovertemplate="rank=%{x}<br>entropy=%{y:.2f}<extra></extra>",
    ), row=3, col=1)
    fig.update_layout(**paper_layout(
        title=dict(text=f"{summary['dataset_title']}: {summary['label']} slot statistics", x=0.02, xanchor="left"),
        width=1080,
        height=860,
        showlegend=False,
        margin=dict(l=78, r=32, t=95, b=72),
    ))
    fig.update_xaxes(title="slot rank sorted by mean predicted time", row=3, col=1)
    fig.update_yaxes(title="time delta", row=1, col=1)
    fig.update_yaxes(title="probability / ratio", range=[0, 1.05], row=2, col=1)
    fig.update_yaxes(title="entropy", row=3, col=1)
    apply_axis_style(fig)
    return fig


def plot_slot_heatmap(summary: dict) -> go.Figure:
    z = list(map(list, zip(*summary["label_counts"])))
    x = list(range(1, summary["k"] + 1))
    y = list(range(len(z)))
    fig = go.Figure(go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale="Greys",
        colorbar=dict(title="count"),
        hovertemplate="slot rank=%{x}<br>label=%{y}<br>count=%{z:.2f}<extra></extra>",
    ))
    fig.update_layout(**paper_layout(
        title=dict(text=f"{summary['dataset_title']}: {summary['label']} label distribution by slot", x=0.02, xanchor="left"),
        width=1080,
        height=620,
        margin=dict(l=78, r=34, t=78, b=78),
        xaxis=dict(title="slot rank sorted by mean predicted time"),
        yaxis=dict(title="label id"),
    ))
    apply_axis_style(fig)
    return fig


def plot_presence_comparison(summaries: list[dict], dataset_title: str) -> go.Figure:
    fig = go.Figure()
    for summary in summaries:
        x = list(range(1, summary["k"] + 1))
        fig.add_trace(go.Scatter(
            x=x,
            y=summary["mean_presence_probability"],
            mode="lines+markers",
            name=summary["label"],
            line=dict(color=summary["color"], width=2.2),
            marker=dict(size=5),
            hovertemplate="<b>%{fullData.name}</b><br>rank=%{x}<br>presence=%{y:.2f}<extra></extra>",
        ))
    fig.update_layout(**paper_layout(
        title=dict(text=f"{dataset_title}: predicted presence across sorted slots", x=0.02, xanchor="left"),
        width=1080,
        height=560,
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        margin=dict(l=78, r=32, t=74, b=130),
        xaxis=dict(title="slot rank sorted by mean predicted time"),
        yaxis=dict(title="mean presence probability", range=[0, 1.05]),
    ))
    apply_axis_style(fig)
    return fig


def main() -> None:
    args = parse_args()
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    run_specs = [run for run in RUNS if run.name in set(args.runs)]

    for dataset in args.datasets:
        summaries = []
        for run in run_specs:
            _, _, checkpoint = resolve_run_paths(dataset, run.name)
            if not checkpoint.exists():
                print(f"skip {dataset}/{run.name}: no checkpoint")
                continue
            if not args.no_export:
                try:
                    summary = collect_run_summary(dataset, run, args.split, args.max_batches, args.batch_size, args.device)
                    write_summary(summary, args.summary_dir)
                except Exception as exc:
                    print(f"skip {dataset}/{run.name}: {exc.__class__.__name__}: {exc}")
                    continue
            summary = read_summary(args.summary_dir, dataset, run.name, args.split)
            if summary is None:
                continue
            summaries.append(summary)
            stem = f"{dataset}_inference_slots_{run.name}"
            save_figure(plot_slot_summary(summary), args.output_dir, stem, formats)
            save_figure(plot_slot_heatmap(summary), args.output_dir, f"{stem}_labels", formats)
        if summaries:
            save_figure(
                plot_presence_comparison(summaries, DATASETS[dataset]),
                args.output_dir,
                f"{dataset}_inference_presence_comparison",
                formats,
            )


if __name__ == "__main__":
    main()
