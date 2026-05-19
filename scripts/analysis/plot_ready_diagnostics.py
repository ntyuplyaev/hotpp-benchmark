"""Build ready-to-plot diagnostics from existing experiment artifacts.

This script does not run inference and does not train models. It uses only:
  * YAML reports with final validation/test metrics;
  * MLflow SQLite logs with validation metric histories;
  * checkpoint buffers/parameters such as matching priors and query vectors.

Examples:
  python scripts/analysis/plot_ready_diagnostics.py --dataset stackoverflow
  python scripts/analysis/plot_ready_diagnostics.py --dataset amazon --run a6_query_only_decoder_v3_paper_detection
"""
from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import torch
import yaml

try:
    pio.defaults.mathjax = None
except Exception:
    try:
        pio.kaleido.scope.mathjax = None
    except Exception:
        pass


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    group: str
    source: str = "canonical"
    color: str = "#8c8c8c"


DATASETS = {
    "stackoverflow": {
        "title": "StackOverflow",
        "canonical_results": ROOT / "experiments" / "stackoverflow" / "canonical" / "results",
        "legacy_results": ROOT / "experiments" / "stackoverflow" / "results",
        "checkpoints": ROOT / "experiments" / "stackoverflow" / "canonical" / "checkpoints",
        "mlflow": ROOT / "experiments" / "stackoverflow" / "lightning_logs" / "canonical_mlflow.db",
        "models": [
            ModelSpec("paper_rmtpp", "RMTPP", "Baseline", source="paper", color="#117733"),
            ModelSpec("paper_hypro", "HYPRO", "Baseline", source="paper", color="#cc6677"),
            ModelSpec("paper_detpp", "DeTPP", "Baseline", source="paper", color="#2a78b8"),
            ModelSpec("paper_diffusion", "Diffusion", "Baseline", source="paper", color="#7e57c2"),
            ModelSpec("a6_query_only_decoder_v3_paper_detection", "Hybrid (latent+query)", "Hybrid", color="#e08a3c"),
            ModelSpec("a6_query_only_decoder_v8_full_components_paper_detection", "Hybrid (+context)", "Hybrid", color="#6f9fd8"),
            ModelSpec("a6_query_only_decoder_v11_matching_padding_paper_detection", "Hybrid (matching padding)", "Hybrid", color="#b590d8"),
            ModelSpec("a6_query_only_decoder_v12_far_target_paper_detection", "Hybrid (far-target)", "Hybrid", color="#3fa37b"),
        ],
    },
    "amazon": {
        "title": "Amazon",
        "canonical_results": ROOT / "experiments" / "amazon" / "canonical" / "results",
        "legacy_results": ROOT / "experiments" / "amazon" / "results",
        "checkpoints": ROOT / "experiments" / "amazon" / "canonical" / "checkpoints",
        "mlflow": ROOT / "experiments" / "amazon" / "lightning_logs" / "canonical_mlflow.db",
        "models": [
            ModelSpec("paper_rmtpp", "RMTPP", "Baseline", source="paper", color="#117733"),
            ModelSpec("paper_hypro", "HYPRO", "Baseline", source="paper", color="#cc6677"),
            ModelSpec("paper_detpp", "DeTPP", "Baseline", source="paper", color="#2a78b8"),
            ModelSpec("paper_diffusion", "Diffusion", "Baseline", source="paper", color="#7e57c2"),
            ModelSpec("a6_query_only_decoder_v3_paper_detection", "Hybrid (latent+query)", "Hybrid", color="#e08a3c"),
            ModelSpec("a6_query_only_decoder_v8_full_components_paper_detection", "Hybrid (+context)", "Hybrid", color="#6f9fd8"),
            ModelSpec("a6_query_only_decoder_v11_matching_padding_paper_detection", "Hybrid (matching padding)", "Hybrid", color="#b590d8"),
            ModelSpec("a6_query_only_decoder_v12_far_target_paper_detection", "Hybrid (far-target)", "Hybrid", color="#3fa37b"),
        ],
    },
    "retweet": {
        "title": "Retweet",
        "canonical_results": ROOT / "experiments" / "retweet" / "canonical" / "results",
        "legacy_results": ROOT / "experiments" / "retweet" / "results",
        "checkpoints": ROOT / "experiments" / "retweet" / "canonical" / "checkpoints",
        "mlflow": ROOT / "experiments" / "retweet" / "lightning_logs" / "canonical_mlflow.db",
        "models": [
            ModelSpec("paper_rmtpp", "RMTPP", "Baseline", source="paper", color="#117733"),
            ModelSpec("paper_hypro", "HYPRO", "Baseline", source="paper", color="#cc6677"),
            ModelSpec("paper_detpp", "DeTPP", "Baseline", source="paper", color="#2a78b8"),
            ModelSpec("paper_diffusion", "Diffusion", "Baseline", source="paper", color="#7e57c2"),
            ModelSpec("a6_query_only_decoder_v3_paper_detection", "Hybrid (latent+query)", "Hybrid", color="#e08a3c"),
            ModelSpec("a6_query_only_decoder_v8_full_components_paper_detection", "Hybrid (+context)", "Hybrid", color="#6f9fd8"),
            ModelSpec("a6_query_only_decoder_v11_matching_padding_paper_detection", "Hybrid (matching padding)", "Hybrid", color="#b590d8"),
            ModelSpec("a6_query_only_decoder_v12_far_target_paper_detection", "Hybrid (far-target)", "Hybrid", color="#3fa37b"),
        ],
    },
}


PAPER_RESULTS = {
    "stackoverflow": {
        "paper_rmtpp": {
            "test_tmap": 0.1335,
            "otd": 13.25,
            "next_mae": 0.717,
            "entropy": 0.1592,
            "target_entropy": 1.4228,
        },
        "paper_hypro": {
            "test_tmap": 0.1484,
            "otd": 14.12,
            "next_mae": 0.718,
            "entropy": 1.5691,
            "target_entropy": 1.4228,
        },
        "paper_detpp": {
            "test_tmap": 0.2311,
            "otd": 12.06,
            "next_mae": 0.662,
            "entropy": 0.7275,
            "target_entropy": 1.4228,
        },
        "paper_diffusion": {
            "test_tmap": 0.1507,
            "otd": 13.01,
            "next_mae": 0.668,
            "entropy": 1.1695,
            "target_entropy": 1.4228,
        },
    },
    "amazon": {
        "paper_rmtpp": {
            "test_tmap": 0.1999,
            "otd": 6.62,
            "next_mae": 0.292,
            "entropy": 0.2579,
            "target_entropy": 1.0760,
        },
        "paper_hypro": {
            "test_tmap": 0.2028,
            "otd": 6.95,
            "next_mae": 0.288,
            "entropy": 1.2110,
            "target_entropy": 1.0760,
        },
        "paper_detpp": {
            "test_tmap": 0.3718,
            "otd": 5.98,
            "next_mae": 0.494,
            "entropy": 0.9451,
            "target_entropy": 1.0760,
        },
        "paper_diffusion": {
            "test_tmap": 0.3029,
            "otd": 6.52,
            "next_mae": 0.247,
            "entropy": 0.6508,
            "target_entropy": 1.0760,
        },
    },
    "retweet": {
        "paper_rmtpp": {
            "test_tmap": 0.4560,
            "otd": 168.0,
            "next_mae": 18.60,
            "entropy": 0.0873,
            "target_entropy": 0.4827,
        },
        "paper_hypro": {
            "test_tmap": 0.4687,
            "otd": 160.4,
            "next_mae": 18.61,
            "entropy": 0.4525,
            "target_entropy": 0.4827,
        },
        "paper_detpp": {
            "test_tmap": 0.5737,
            "otd": 134.4,
            "next_mae": 18.44,
            "entropy": 0.4221,
            "target_entropy": 0.4827,
        },
        "paper_diffusion": {
            "test_tmap": 0.5224,
            "otd": 158.0,
            "next_mae": 18.60,
            "entropy": 0.3279,
            "target_entropy": 0.4827,
        },
    },
}


PAPER_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Times New Roman", size=26, color="#1f1f1f"),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=90, r=40, t=80, b=120),
    title_font=dict(size=32),
)


def paper_layout(**overrides) -> dict:
    layout = dict(PAPER_LAYOUT)
    layout.update(overrides)
    return layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="stackoverflow")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=None,
                        help="Datasets for combined plots. Defaults to the selected --dataset.")
    parser.add_argument("--combined", action="store_true",
                        help="Also build paper-style combined plots across --datasets.")
    parser.add_argument("--run", default="a6_query_only_decoder_v3_paper_detection")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "text_final" / "figures")
    parser.add_argument("--formats", default="pdf",
                        help="Comma-separated output formats. Use html if kaleido is unavailable.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def metric(data: dict, key: str) -> float | None:
    value = data.get(key)
    return None if value is None else float(value)


def as_percent(value: float | None) -> float | None:
    return None if value is None else value * 100


def result_path(conf: dict, spec: ModelSpec) -> Path:
    root = conf["legacy_results"] if spec.source == "legacy" else conf["canonical_results"]
    return root / f"{spec.name}.yaml"


def collect_results(conf: dict) -> list[dict]:
    rows = []
    dataset_key = next(key for key, value in DATASETS.items() if value is conf)
    for spec in conf["models"]:
        if spec.source == "paper":
            data = PAPER_RESULTS.get(dataset_key, {}).get(spec.name)
            if data is None:
                continue
            rows.append({
                "name": spec.name,
                "label": spec.label,
                "group": spec.group,
                "color": spec.color,
                "test_tmap": data.get("test_tmap"),
                "val_tmap": data.get("val_tmap"),
                "otd": data.get("otd"),
                "entropy": data.get("entropy"),
                "target_entropy": data.get("target_entropy"),
                "mean_pred_len": data.get("mean_pred_len"),
                "next_acc": data.get("next_acc"),
                "next_mae": data.get("next_mae"),
            })
            continue
        path = result_path(conf, spec)
        if not path.exists():
            continue
        data = load_yaml(path)
        rows.append({
            "name": spec.name,
            "label": spec.label,
            "group": spec.group,
            "color": spec.color,
            "test_tmap": metric(data, "test/T-mAP"),
            "val_tmap": metric(data, "val/T-mAP"),
            "otd": metric(data, "test/optimal-transport-distance"),
            "entropy": metric(data, "test/sequence-labels-entropy"),
            "target_entropy": metric(data, "test/target-sequence-labels-entropy") or metric(data, "val/target-sequence-labels-entropy"),
            "mean_pred_len": metric(data, "test/mean-predicted-length"),
            "next_acc": metric(data, "test/next-item-accuracy"),
            "next_mae": metric(data, "test/next-item-mae"),
        })
    return rows


def collect_combined_results(dataset_names: Sequence[str]) -> list[dict]:
    rows = []
    for dataset_name in dataset_names:
        conf = DATASETS[dataset_name]
        for row in collect_results(conf):
            row = dict(row)
            row["dataset"] = conf["title"]
            rows.append(row)
    return rows


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
        try:
            scale = 2 if fmt == "png" else None
            if scale is None:
                fig.write_image(str(path), format=fmt)
            else:
                fig.write_image(str(path), format=fmt, scale=scale)
            print(f"saved {path}")
        except Exception as exc:
            print(f"skip {path}: {exc.__class__.__name__}: {exc}")


def apply_axis_style(fig: go.Figure) -> None:
    fig.update_xaxes(showline=True, linewidth=1, linecolor="#222", mirror=False, ticks="outside")
    fig.update_yaxes(showline=True, linewidth=1, linecolor="#222", mirror=False, ticks="outside",
                     gridcolor="#e6e6e6", zeroline=False)


def plot_ablation_tmap(rows: list[dict], dataset_title: str) -> go.Figure:
    labels = [r["label"] for r in rows if r["test_tmap"] is not None]
    values = [as_percent(r["test_tmap"]) for r in rows if r["test_tmap"] is not None]
    colors = [r["color"] for r in rows if r["test_tmap"] is not None]
    groups = [r["group"] for r in rows if r["test_tmap"] is not None]
    base = as_percent(next((r["test_tmap"] for r in rows if r["name"] == "a6_query_only_decoder_v3_paper_detection"), None))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker=dict(color=colors, line=dict(color="#333", width=0.7)),
        customdata=groups,
        hovertemplate="<b>%{x}</b><br>group=%{customdata}<br>test T-mAP=%{y:.2f}%<extra></extra>",
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
        textfont=dict(size=24),
    ))
    if base is not None:
        fig.add_hline(y=base, line_width=1.5, line_dash="dash", line_color="#e08a3c")
    fig.update_layout(**paper_layout(
        title=dict(text=f"{dataset_title}: T-mAP across baselines and hybrid variants", x=0.02, xanchor="left"),
        width=1500,
        height=820,
        showlegend=False,
        xaxis=dict(tickangle=-32, title=None, tickfont=dict(size=24)),
        yaxis=dict(title=dict(text="test T-mAP, %", font=dict(size=30)),
                   tickfont=dict(size=24),
                   range=[0, max(values) * 1.16]),
        margin=dict(l=110, r=50, t=110, b=210),
    ))
    apply_axis_style(fig)
    return fig


def plot_metric_overview(rows: list[dict], dataset_title: str) -> go.Figure:
    keep = [r for r in rows if r["name"] in {
        "paper_detpp",
        "paper_diffusion",
        "a6_query_only_decoder_v3_paper_detection",
        "a6_query_only_decoder_v12_far_target_paper_detection",
    }]
    labels = [r["label"] for r in keep]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("T-mAP, % (higher)", "OTD (lower)", "Entropy", "Mean predicted length"),
        horizontal_spacing=0.14,
        vertical_spacing=0.28,
    )
    metrics = [
        ("test_tmap", 1, 1),
        ("otd", 1, 2),
        ("entropy", 2, 1),
        ("mean_pred_len", 2, 2),
    ]
    for key, row, col in metrics:
        vals = [as_percent(r[key]) if key == "test_tmap" else r[key] for r in keep]
        finite_vals = [v for v in vals if v is not None]
        fig.add_trace(go.Bar(
            x=labels,
            y=vals,
            marker=dict(color=[r["color"] for r in keep], line=dict(color="#333", width=0.6)),
            text=[f"{v:.2f}" if v is not None else "" for v in vals],
            textposition="outside",
            textfont=dict(size=24),
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>%{y:.2f}<extra></extra>",
        ), row=row, col=col)
        if finite_vals:
            fig.update_yaxes(range=[0, max(finite_vals) * 1.22], row=row, col=col)

    fig.update_layout(**paper_layout(
        title=dict(text=f"{dataset_title}: main diagnostic metrics", x=0.02, xanchor="left"),
        width=1480,
        height=1080,
        margin=dict(l=100, r=50, t=120, b=170),
    ))
    fig.update_annotations(font_size=30)
    fig.update_xaxes(tickangle=-22, tickfont=dict(size=22))
    fig.update_yaxes(tickfont=dict(size=22))
    apply_axis_style(fig)
    return fig


def plot_label_entropy(rows: list[dict], dataset_title: str) -> go.Figure:
    labels = [r["label"] for r in rows if r["entropy"] is not None]
    vals = [r["entropy"] for r in rows if r["entropy"] is not None]
    colors = [r["color"] for r in rows if r["entropy"] is not None]
    target_entropy = next((r["target_entropy"] for r in rows if r["target_entropy"] is not None), None)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=vals,
        marker=dict(color=colors, line=dict(color="#333", width=0.7)),
        text=[f"{v:.2f}" for v in vals],
        textposition="outside",
        textfont=dict(size=24),
        hovertemplate="<b>%{x}</b><br>entropy=%{y:.2f}<extra></extra>",
    ))
    if target_entropy is not None:
        fig.add_hline(y=target_entropy, line_width=1.6, line_dash="dash", line_color="#222",
                      annotation_text="Ground-truth entropy", annotation_position="top right",
                      annotation_font=dict(size=24, color="#222"))
    fig.update_layout(**paper_layout(
        title=dict(text=f"{dataset_title}: entropy of predicted labels", x=0.02, xanchor="left"),
        width=1500,
        height=820,
        showlegend=False,
        xaxis=dict(tickangle=-32, tickfont=dict(size=24)),
        yaxis=dict(title=dict(text="sequence labels entropy", font=dict(size=30)),
                   tickfont=dict(size=24),
                   range=[0, max(vals + ([target_entropy] if target_entropy else [])) * 1.22]),
        margin=dict(l=130, r=50, t=110, b=210),
    ))
    apply_axis_style(fig)
    return fig


def compact_method_label(label: str) -> str:
    label = label.replace("<br>", " ")
    label = " ".join(label.split())
    replacements = {
        "Hybrid Diffusion + DeTPP": "Hybrid (latent+query)",
        "Hybrid Diffusion + DeTPP (diffusion latent + query + context)": "Hybrid (+context)",
        "Hybrid (latent+query)": "Hybrid (latent+query)",
        "Hybrid (+context)": "Hybrid (+context)",
        "Hybrid (matching padding)": "Hybrid (matching padding)",
        "Hybrid (far-target)": "Hybrid (far-target)",
        "Hybrid Diffusion + DeTPP (matching padding)": "Hybrid (matching padding)",
        "Hybrid Diffusion + DeTPP (far-target)": "Hybrid (far-target)",
    }
    if label in replacements:
        return replacements[label]
    return label


def method_order(labels: list[str]) -> list[str]:
    return list(reversed(labels))


def plot_combined_metric(rows: list[dict], metric_key: str, title: str, y_title: str,
                         show_target_entropy: bool = False) -> go.Figure:
    datasets = []
    for row in rows:
        if row["dataset"] not in datasets:
            datasets.append(row["dataset"])
    labels = []
    for row in rows:
        label = compact_method_label(row["label"])
        if label not in labels:
            labels.append(label)

    fig = make_subplots(
        rows=len(datasets),
        cols=1,
        subplot_titles=datasets,
        vertical_spacing=0.08,
    )
    order = method_order(labels)
    for row_idx, dataset in enumerate(datasets, start=1):
        dataset_rows = [
            r for r in rows
            if r["dataset"] == dataset and r[metric_key] is not None
        ]
        by_label = {compact_method_label(r["label"]): r for r in dataset_rows}
        y_labels = [label for label in order if label in by_label]
        x_values = [
            as_percent(by_label[label][metric_key]) if metric_key == "test_tmap" else by_label[label][metric_key]
            for label in y_labels
        ]
        fig.add_trace(go.Bar(
            x=x_values,
            y=y_labels,
            orientation="h",
            name=dataset,
            marker=dict(color=[by_label[label]["color"] for label in y_labels], line=dict(color="#333", width=0.8)),
            text=[f"   {v:.2f}" for v in x_values],
            textposition="outside",
            textfont=dict(size=28, color="#1f1f1f"),
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>dataset=" + dataset + "<br>value=%{x:.2f}<extra></extra>",
            showlegend=False,
        ), row=row_idx, col=1)
        if show_target_entropy:
            target_entropy = next((r["target_entropy"] for r in dataset_rows
                                   if r["target_entropy"] is not None), None)
            if target_entropy is not None:
                fig.add_trace(go.Scatter(
                    x=[target_entropy] * len(y_labels),
                    y=y_labels,
                    mode="lines",
                    name="Ground-truth entropy",
                    line=dict(color="#222", width=2.0, dash="dash"),
                    hovertemplate="<b>Ground-truth entropy</b><br>dataset=" + dataset + "<br>value=%{x:.2f}<extra></extra>",
                    showlegend=False,
                ), row=row_idx, col=1)
        finite = x_values + ([target_entropy] if show_target_entropy and target_entropy is not None else [])
        if finite:
            fig.update_xaxes(range=[0, max(finite) * 1.28], row=row_idx, col=1)
        fig.update_yaxes(categoryorder="array", categoryarray=order, row=row_idx, col=1)

    fig.update_layout(**paper_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=36)),
        width=1500,
        height=1640,
        font=dict(family="Times New Roman", size=30, color="#1f1f1f"),
        margin=dict(l=420, r=85, t=130, b=140),
        bargap=0.35,
        uniformtext=dict(minsize=22, mode="show"),
    ))
    fig.update_annotations(font_size=32)
    for row_idx in range(1, len(datasets) + 1):
        is_last = (row_idx == len(datasets))
        fig.update_xaxes(
            title=dict(text=y_title, font=dict(size=30)) if is_last else None,
            tickfont=dict(size=26),
            nticks=6,
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(title=None, tickfont=dict(size=26), row=row_idx, col=1)
    apply_axis_style(fig)
    return fig


def plot_entropy_otd_tradeoff(rows: list[dict], title: str) -> go.Figure:
    datasets = []
    for row in rows:
        if row["dataset"] not in datasets:
            datasets.append(row["dataset"])

    fig = make_subplots(
        rows=1,
        cols=len(datasets),
        subplot_titles=datasets,
        horizontal_spacing=0.10,
    )
    labels_for_legend = []
    for row in rows:
        label = compact_method_label(row["label"])
        if label not in labels_for_legend:
            labels_for_legend.append(label)

    for col, dataset in enumerate(datasets, start=1):
        dataset_rows = [
            r for r in rows
            if r["dataset"] == dataset and r["entropy"] is not None and r["otd"] is not None
        ]
        for row in dataset_rows:
            label = compact_method_label(row["label"])
            fig.add_trace(go.Scatter(
                x=[row["entropy"]],
                y=[row["otd"]],
                mode="markers",
                name=label,
                marker=dict(
                    color=row["color"],
                    size=36,
                    line=dict(color="#222", width=1.8),
                ),
                hovertemplate=(
                    "<b>" + label + "</b><br>"
                    "dataset=" + dataset + "<br>"
                    "entropy=%{x:.2f}<br>OTD=%{y:.2f}<extra></extra>"
                ),
                legendgroup=label,
                showlegend=(col == 1),
            ), row=1, col=col)

        target_entropy = next((r["target_entropy"] for r in dataset_rows
                               if r["target_entropy"] is not None), None)
        if target_entropy is not None:
            ys = [r["otd"] for r in dataset_rows]
            if ys:
                fig.add_trace(go.Scatter(
                    x=[target_entropy, target_entropy],
                    y=[min(ys) * 0.97, max(ys) * 1.03],
                    mode="lines",
                    line=dict(color="#222", width=2.0, dash="dash"),
                    hovertemplate="<b>Ground-truth entropy</b><br>dataset=" + dataset + "<br>value=%{x:.2f}<extra></extra>",
                    showlegend=False,
                ), row=1, col=col)
        xs = [r["entropy"] for r in dataset_rows]
        ys = [r["otd"] for r in dataset_rows]
        if xs:
            xmax = max(xs + ([target_entropy] if target_entropy is not None else []))
            xmin = min(xs + ([target_entropy] if target_entropy is not None else []))
            xpad = max((xmax - xmin) * 0.18, 0.05)
            fig.update_xaxes(range=[max(0, xmin - xpad), xmax + xpad], row=1, col=col)
        if ys:
            ymin, ymax = min(ys), max(ys)
            ypad = max((ymax - ymin) * 0.22, 0.1)
            fig.update_yaxes(range=[max(0, ymin - ypad), ymax + ypad], row=1, col=col)

    fig.update_layout(**paper_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=44)),
        width=2400,
        height=1300,
        font=dict(family="Times New Roman", size=38, color="#1f1f1f"),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.01,
            font=dict(size=46),
            itemsizing="trace",
            tracegroupgap=20,
        ),
        margin=dict(l=200, r=900, t=170, b=170),
    ))
    fig.update_annotations(font_size=42)
    for col in range(1, len(datasets) + 1):
        fig.update_xaxes(
            title=dict(text="entropy", font=dict(size=50)),
            tickfont=dict(size=44),
            row=1,
            col=col,
        )
        fig.update_yaxes(
            title=dict(text="test OTD", font=dict(size=50)) if col == 1 else None,
            tickfont=dict(size=44),
            row=1,
            col=col,
        )
    apply_axis_style(fig)
    return fig


def mlflow_run_uuid(db_path: Path, run_name: str) -> str | None:
    if not db_path.exists():
        return None
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    row = conn.execute(
        "SELECT run_uuid FROM runs WHERE name=? ORDER BY COALESCE(end_time,start_time) DESC LIMIT 1",
        (run_name,),
    ).fetchone()
    conn.close()
    return None if row is None else row[0]


def mlflow_metric_history(db_path: Path, run_name: str, keys: list[str]) -> dict[str, list[tuple[int, float]]]:
    run_uuid = mlflow_run_uuid(db_path, run_name)
    if run_uuid is None:
        return {}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    out = {}
    for key in keys:
        rows = conn.execute(
            "SELECT step, value FROM metrics WHERE run_uuid=? AND key=? ORDER BY step, timestamp",
            (run_uuid, key),
        ).fetchall()
        if rows:
            # Keep the last value per step.
            by_step = {}
            for step, value in rows:
                by_step[int(step)] = float(value)
            out[key] = sorted(by_step.items())
    conn.close()
    return out


def plot_training_quality(conf: dict, run_name: str, dataset_title: str) -> go.Figure | None:
    keys = [
        "val/T-mAP",
        "val/optimal-transport-distance",
        "val/next-item-mae",
        "val/next-item-accuracy",
    ]
    hist = mlflow_metric_history(conf["mlflow"], run_name, keys)
    if not hist:
        return None

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    quality = [
        ("val/T-mAP", "#2a78b8", "val T-mAP, %", False),
        ("val/optimal-transport-distance", "#d95f02", "val OTD", True),
        ("val/next-item-mae", "#7570b3", "val next-event time MAE", True),
        ("val/next-item-accuracy", "#1b9e77", "val next-item acc", True),
    ]
    for key, color, name, secondary in quality:
        if key not in hist:
            continue
        xs, ys = zip(*hist[key])
        if key == "val/T-mAP":
            ys = tuple(v * 100 for v in ys)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=name,
                                 line=dict(color=color, width=2.5), marker=dict(size=7)),
                      secondary_y=secondary)
    fig.update_layout(**paper_layout(
        title=dict(text=f"{dataset_title}: validation quality dynamics for Hybrid Diffusion + DeTPP", x=0.02, xanchor="left", font=dict(size=34)),
        width=1100,
        height=780,
        font=dict(family="Times New Roman", size=30, color="#1f1f1f"),
        legend=dict(orientation="h", yanchor="top", y=-0.24, xanchor="center", x=0.5,
                    font=dict(size=28)),
        margin=dict(l=130, r=120, t=120, b=200),
    ))
    fig.update_xaxes(title=dict(text="training step", font=dict(size=32)),
                     tickfont=dict(size=28))
    fig.update_yaxes(title=dict(text="T-mAP, %", font=dict(size=32)),
                     tickfont=dict(size=28), secondary_y=False)
    fig.update_yaxes(title=dict(text="OTD / next-item metrics", font=dict(size=32)),
                     tickfont=dict(size=28), secondary_y=True)
    apply_axis_style(fig)
    return fig


def plot_loss_curves(conf: dict, run_name: str, dataset_title: str) -> go.Figure | None:
    keys = [
        "val/loss_diffusion",
        "val/loss_decoder__presence",
        "val/loss_decoder_labels",
        "val/loss_decoder_timestamps",
    ]
    hist = mlflow_metric_history(conf["mlflow"], run_name, keys)
    if not hist:
        return None

    fig = go.Figure()
    losses = [
        ("val/loss_diffusion", "#7e57c2", "diffusion"),
        ("val/loss_decoder__presence", "#e08a3c", "presence"),
        ("val/loss_decoder_labels", "#2a78b8", "labels"),
        ("val/loss_decoder_timestamps", "#3fa37b", "time"),
    ]
    for key, color, name in losses:
        if key not in hist:
            continue
        xs, ys = zip(*hist[key])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=name,
                                 line=dict(color=color, width=2.5), marker=dict(size=7)),
                      )
    fig.update_layout(**paper_layout(
        title=dict(text=f"{dataset_title}: validation loss components for Hybrid Diffusion + DeTPP", x=0.02, xanchor="left", font=dict(size=34)),
        width=1100,
        height=780,
        font=dict(family="Times New Roman", size=30, color="#1f1f1f"),
        legend=dict(orientation="h", yanchor="top", y=-0.24, xanchor="center", x=0.5,
                    font=dict(size=28)),
        margin=dict(l=130, r=70, t=120, b=200),
    ))
    fig.update_xaxes(title=dict(text="training step", font=dict(size=32)),
                     tickfont=dict(size=28))
    fig.update_yaxes(title=dict(text="loss", font=dict(size=32)),
                     tickfont=dict(size=28))
    apply_axis_style(fig)
    return fig


def load_checkpoint_tensors(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    if not checkpoint_path.exists():
        return {}
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return {k: v for k, v in state.items() if torch.is_tensor(v)}


def plot_checkpoint_slot_stats(conf: dict, run_name: str, dataset_title: str) -> go.Figure | None:
    tensors = load_checkpoint_tensors(conf["checkpoints"] / f"{run_name}.ckpt")
    priors = tensors.get("_loss._next_item._matching_priors")
    thresholds = tensors.get("_loss._next_item._matching_thresholds")
    queries = tensors.get("_loss._decoder.queries")
    mask_embedding = tensors.get("_loss._mask_embedding")
    if priors is None and thresholds is None and queries is None:
        return None

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("Matching prior by head", "Presence threshold by head", "Query vector norm by head"),
    )
    if priors is not None:
        x = list(range(1, priors.numel() + 1))
        fig.add_trace(go.Bar(x=x, y=priors.numpy(), marker_color="#2a78b8", name="matching prior",
                             hovertemplate="head=%{x}<br>prior=%{y:.2f}<extra></extra>"),
                      row=1, col=1)
    if thresholds is not None:
        x = list(range(1, thresholds.numel() + 1))
        fig.add_trace(go.Scatter(x=x, y=thresholds.numpy(), mode="lines+markers",
                                 line=dict(color="#e08a3c", width=2.2),
                                 marker=dict(size=5),
                                 name="presence threshold",
                                 hovertemplate="head=%{x}<br>threshold=%{y:.2f}<extra></extra>"),
                      row=2, col=1)
    if queries is not None:
        norms = queries.norm(dim=1)
        x = list(range(1, norms.numel() + 1))
        fig.add_trace(go.Scatter(x=x, y=norms.numpy(), mode="lines+markers",
                                 line=dict(color="#3fa37b", width=2.2),
                                 marker=dict(size=5),
                                 name="query norm",
                                 hovertemplate="head=%{x}<br>norm=%{y:.2f}<extra></extra>"),
                      row=3, col=1)
        if mask_embedding is not None:
            fig.add_hline(y=float(mask_embedding.norm().item()), row=3, col=1,
                          line_color="#222", line_dash="dash", line_width=1.3,
                          annotation_text="mask embedding norm",
                          annotation_position="top right",
                          annotation_font=dict(size=28, color="#222"))
    fig.update_layout(**paper_layout(
        title=dict(text=f"{dataset_title}: checkpoint head statistics for Hybrid Diffusion + DeTPP", x=0.02, xanchor="left", font=dict(size=38)),
        width=1500,
        height=1400,
        font=dict(family="Times New Roman", size=34, color="#1f1f1f"),
        showlegend=False,
        margin=dict(l=170, r=100, t=170, b=150),
    ))
    fig.update_annotations(font_size=38)
    fig.update_xaxes(tickfont=dict(size=32))
    fig.update_yaxes(tickfont=dict(size=32))
    fig.update_xaxes(title=dict(text="head index", font=dict(size=36)), row=3, col=1)
    fig.update_yaxes(title=dict(text="prior", font=dict(size=36)), row=1, col=1)
    fig.update_yaxes(title=dict(text="logit", font=dict(size=36)), row=2, col=1)
    fig.update_yaxes(title=dict(text="L2 norm", font=dict(size=36)), row=3, col=1)
    apply_axis_style(fig)
    return fig


def plot_diffusion_reconstruction(capture_path: Path, dataset_title: str) -> go.Figure | None:
    if not capture_path.exists():
        return None
    cap = torch.load(capture_path, map_location="cpu", weights_only=False)
    embeddings = cap["embed"]["embeddings"]
    reconstructed = cap["diffusion"]["reconstructed"]
    presence = cap["embed"]["presence_after_slice"].bool()

    err = (reconstructed - embeddings).norm(dim=-1)
    err_real = err[presence].numpy()
    err_mask = err[~presence].numpy()
    per_slot_err = err.mean(dim=0).numpy()
    per_slot_pres = presence.float().mean(dim=0).numpy()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Distribution of reconstruction error",
            "Mean reconstruction error by head",
        ),
        horizontal_spacing=0.12,
    )

    fig.add_trace(go.Histogram(
        x=err_real, nbinsx=40, name="real-event heads",
        marker_color="#e08a3c", opacity=0.75,
        hovertemplate="error=%{x:.2f}<br>count=%{y}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Histogram(
        x=err_mask, nbinsx=40, name="mask-embedding heads",
        marker_color="#2a78b8", opacity=0.75,
        hovertemplate="error=%{x:.2f}<br>count=%{y}<extra></extra>",
    ), row=1, col=1)

    head_idx = list(range(1, per_slot_err.shape[0] + 1))
    fig.add_trace(go.Scatter(
        x=head_idx, y=per_slot_err, mode="lines+markers",
        name="mean error",
        line=dict(color="#3fa37b", width=2.5), marker=dict(size=8),
        hovertemplate="head=%{x}<br>mean err=%{y:.3f}<extra></extra>",
    ), row=1, col=2)
    scale = float(per_slot_err.max()) if per_slot_err.size else 1.0
    fig.add_trace(go.Scatter(
        x=head_idx, y=per_slot_pres * scale, mode="lines",
        name=f"presence rate × {scale:.2f}",
        line=dict(color="#888", dash="dot", width=2.0),
        customdata=per_slot_pres,
        hovertemplate="head=%{x}<br>presence rate=%{customdata:.3f}<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(**paper_layout(
        title=dict(
            text=f"{dataset_title}: diffusion reconstruction error for Hybrid Diffusion + DeTPP",
            x=0.02, xanchor="left", font=dict(size=42),
        ),
        width=1900,
        height=1100,
        font=dict(family="Times New Roman", size=38, color="#1f1f1f"),
        barmode="overlay",
        legend=dict(
            orientation="h", yanchor="top", y=-0.20, xanchor="center", x=0.5,
            font=dict(size=34),
        ),
        margin=dict(l=190, r=120, t=180, b=260),
    ))
    fig.update_annotations(font_size=40)
    fig.update_xaxes(title=dict(text="‖denoised − target‖", font=dict(size=38)),
                     tickfont=dict(size=34), row=1, col=1)
    fig.update_yaxes(title=dict(text="count", font=dict(size=38)),
                     tickfont=dict(size=34), row=1, col=1)
    fig.update_xaxes(title=dict(text="head index", font=dict(size=38)),
                     tickfont=dict(size=34), row=1, col=2)
    fig.update_yaxes(title=dict(text="mean error", font=dict(size=38)),
                     tickfont=dict(size=34), row=1, col=2)
    apply_axis_style(fig)
    return fig


def main() -> None:
    args = parse_args()
    conf = DATASETS[args.dataset]
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    rows = collect_results(conf)
    if not rows:
        raise RuntimeError(f"No result rows found for {args.dataset}.")

    dataset_title = conf["title"]
    out = args.output_dir
    prefix = args.dataset

    figures: list[tuple[str, go.Figure | None]] = [
        (f"{prefix}_ablation_tmap", plot_ablation_tmap(rows, dataset_title)),
        (f"{prefix}_metrics_overview", plot_metric_overview(rows, dataset_title)),
        (f"{prefix}_label_entropy", plot_label_entropy(rows, dataset_title)),
        (f"{prefix}_training_quality_{args.run}", plot_training_quality(conf, args.run, dataset_title)),
        (f"{prefix}_loss_curves_{args.run}", plot_loss_curves(conf, args.run, dataset_title)),
        (f"{prefix}_slot_checkpoint_stats_{args.run}", plot_checkpoint_slot_stats(conf, args.run, dataset_title)),
        (f"{prefix}_diffusion_reconstruction", plot_diffusion_reconstruction(
            ROOT / "scripts" / "debug" / "v3_debug.pt", dataset_title)),
    ]
    # Generic names used in text_final/text.tex. For now they point to the selected dataset/run.
    generic = {
        f"{prefix}_label_entropy": "hybrid_label_entropy",
        f"{prefix}_training_quality_{args.run}": "hybrid_training_tmap_next_mae",
        f"{prefix}_loss_curves_{args.run}": "hybrid_loss_curves",
        f"{prefix}_slot_checkpoint_stats_{args.run}": "hybrid_slot_statistics",
        f"{prefix}_diffusion_reconstruction": "hybrid_diffusion_reconstruction",
    }

    made = {}
    for stem, fig in figures:
        if fig is None:
            print(f"skip {stem}: no data")
            continue
        save_figure(fig, out, stem, formats)
        made[stem] = fig
        if stem in generic:
            save_figure(fig, out, generic[stem], formats)

    if args.combined:
        dataset_names = args.datasets if args.datasets is not None else [args.dataset]
        combined_rows = collect_combined_results(dataset_names)
        combined_suffix = "_".join(dataset_names)
        combined_figures = [
            (
                f"combined_tmap_{combined_suffix}",
                plot_combined_metric(
                    combined_rows,
                    "test_tmap",
                    "T-mAP, % across datasets",
                    "test T-mAP, %",
                ),
            ),
            (
                f"combined_otd_{combined_suffix}",
                plot_combined_metric(
                    combined_rows,
                    "otd",
                    "Optimal Transport Distance across datasets",
                    "test OTD",
                ),
            ),
            (
                f"combined_label_entropy_{combined_suffix}",
                plot_combined_metric(
                    combined_rows,
                    "entropy",
                    "Predicted label entropy across datasets",
                    "Entropy",
                    show_target_entropy=True,
                ),
            ),
            (
                f"combined_next_mae_{combined_suffix}",
                plot_combined_metric(
                    combined_rows,
                    "next_mae",
                    "Next-event time MAE across datasets",
                    "test next-event MAE",
                ),
            ),
            (
                f"combined_entropy_otd_tradeoff_{combined_suffix}",
                plot_entropy_otd_tradeoff(
                    combined_rows,
                    "OTD and predicted-label diversity trade-off",
                ),
            ),
        ]
        for stem, fig in combined_figures:
            save_figure(fig, out, stem, formats)


if __name__ == "__main__":
    main()
