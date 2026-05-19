"""Bar charts for diploma presentation: 5 models per metric, 3 datasets per chart.

Models per panel (last column is the best hybrid variant for that dataset):
  RMTPP       (autoregressive baseline)
  Diffusion   (DLT-PP baseline)
  DeTPP       (detection baseline)
  Hybrid                    (a6_v3, base hybrid)
  StackOverflow: Hybrid (far-target)   (a6_v12)
  Amazon / Retweet: Hybrid (+context)  (a6_v8)

Metrics rendered (one figure each):
  T-mAP, %             (higher is better)
  OTD                  (lower is better)
  Next-item accuracy   (higher is better)
  Label entropy        (closer to ground-truth dashed line is better)
"""
import os
import yaml
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "text_final" / "figures_actual"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["stackoverflow", "amazon", "retweet"]
DATASET_TITLES = {"stackoverflow": "StackOverflow", "amazon": "Amazon", "retweet": "Retweet"}

# Baseline values: T-mAP / OTD / entropy from HoTPP/DeTPP papers (canonical b1/b2 reproduce them
# within stochastic noise). Next-item accuracy values supplied directly by user (per-dataset
# canonical numbers we have on hand).
PAPER = {
    "stackoverflow": {
        "RMTPP":     {"tmap": 0.1335, "otd": 13.25, "next_acc": 0.4530, "entropy": 0.1592, "target_entropy": 1.4228},
        "Diffusion": {"tmap": 0.1507, "otd": 13.01, "next_acc": 0.3703, "entropy": 1.1695, "target_entropy": 1.4228},
        "DeTPP":     {"tmap": 0.2311, "otd": 12.06, "next_acc": None,   "entropy": 0.7275, "target_entropy": 1.4228},
    },
    "amazon": {
        "RMTPP":     {"tmap": 0.1999, "otd": 6.62,  "next_acc": 0.3585, "entropy": 0.2579, "target_entropy": 1.0760},
        "Diffusion": {"tmap": 0.3029, "otd": 6.52,  "next_acc": 0.3057, "entropy": 0.6508, "target_entropy": 1.0760},
        "DeTPP":     {"tmap": 0.3718, "otd": 5.98,  "next_acc": None,   "entropy": 0.9451, "target_entropy": 1.0760},
    },
    "retweet": {
        "RMTPP":     {"tmap": 0.4560, "otd": 168.0, "next_acc": 0.6001, "entropy": 0.0873, "target_entropy": 0.4827},
        "Diffusion": {"tmap": 0.5224, "otd": 158.0, "next_acc": 0.5772, "entropy": 0.3280, "target_entropy": 0.4827},
        "DeTPP":     {"tmap": 0.5737, "otd": 134.4, "next_acc": None,   "entropy": 0.4220, "target_entropy": 0.4827},
    },
}

# DeTPP next-item accuracy comes from our canonical b2 run (paper does not report it).
DETPP_NEXT_ACC_CANONICAL = {
    "stackoverflow": 0.44888535141944885,
    "amazon":        0.34671205282211304,
    "retweet":       0.5931285619735718,
}

# Manual overrides for specific (dataset, model, metric) cells when the
# value coming from canonical/results yaml is wrong or refers to a different run.
OVERRIDES = {
    ("amazon", "Hybrid",            "next_acc"): 0.3252,
    ("amazon", "Hybrid (+context)", "next_acc"): 0.3246,
}

# Per-dataset best hybrid variant (last column on each panel).
BEST_HYBRID = {
    "stackoverflow": ("a6_query_only_decoder_v12_far_target_paper_detection", "Hybrid (far-target)"),
    "amazon":        ("a6_query_only_decoder_v8_full_components_paper_detection", "Hybrid (+context)"),
    "retweet":       ("a6_query_only_decoder_v8_full_components_paper_detection", "Hybrid (+context)"),
}

HYBRID_BASE_NAME = "a6_query_only_decoder_v3_paper_detection"

COLORS = {
    "RMTPP":              "#117733",
    "Diffusion":          "#7e57c2",
    "DeTPP":              "#2a78b8",
    "Hybrid":             "#e08a3c",
    "Hybrid (far-target)": "#3fa37b",
    "Hybrid (+context)":   "#3fa37b",
}

OUR_LABELS = {"Hybrid", "Hybrid (far-target)", "Hybrid (+context)"}


def load_canonical(dataset, yaml_name):
    path = ROOT / "experiments" / dataset / "canonical" / "results" / f"{yaml_name}.yaml"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    return {
        "tmap":     d.get("test/T-mAP"),
        "otd":      d.get("test/optimal-transport-distance"),
        "next_acc": d.get("test/next-item-accuracy"),
        "entropy":  d.get("test/sequence-labels-entropy"),
    }


def get_metric(dataset, model_label, metric):
    override = OVERRIDES.get((dataset, model_label, metric))
    if override is not None:
        return override
    if model_label in PAPER[dataset]:
        v = PAPER[dataset][model_label].get(metric)
        if model_label == "DeTPP" and metric == "next_acc" and v is None:
            v = DETPP_NEXT_ACC_CANONICAL.get(dataset)
        return v
    if model_label == "Hybrid":
        data = load_canonical(dataset, HYBRID_BASE_NAME)
    else:  # best hybrid variant
        best_yaml, best_label = BEST_HYBRID[dataset]
        if model_label != best_label:
            return None
        data = load_canonical(dataset, best_yaml)
    if data is None:
        return None
    return data.get(metric)


def model_order_for(dataset):
    _, best_label = BEST_HYBRID[dataset]
    return ["RMTPP", "Diffusion", "DeTPP", "Hybrid", best_label]


def bold_label(label):
    return f"<b>{label}</b>" if label in OUR_LABELS else label


def build_bar_figure(metric_key, axis_title, scale=1.0, decimals=3,
                     add_target_line=False, title=None):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[DATASET_TITLES[d] for d in DATASETS],
        horizontal_spacing=0.08,
    )
    for col, ds in enumerate(DATASETS, start=1):
        models = model_order_for(ds)
        labels = [bold_label(m) for m in models]
        values = []
        colors = []
        for m in models:
            v = get_metric(ds, m, metric_key)
            values.append(v * scale if v is not None else None)
            colors.append(COLORS[m])

        fig.add_trace(
            go.Bar(
                x=labels,
                y=values,
                marker=dict(color=colors, line=dict(color="#333", width=0.6)),
                text=[(f"{v:.{decimals}f}" if v is not None else "—") for v in values],
                textposition="outside",
                textfont=dict(size=20),
                showlegend=False,
                hovertemplate="<b>%{x}</b><br>%{y:.4f}<extra></extra>",
            ),
            row=1, col=col,
        )

        finite = [v for v in values if v is not None]
        if finite:
            ymax = max(finite) * 1.22
            if add_target_line:
                te = PAPER[ds]["RMTPP"]["target_entropy"] * scale
                ymax = max(ymax, te * 1.18)
                fig.add_hline(
                    y=te, line_dash="dash", line_color="#000000", line_width=1.5,
                    row=1, col=col,
                )
            fig.update_yaxes(range=[0, ymax], row=1, col=col)

    fig.update_layout(
        title=dict(text=title or "", x=0.02, xanchor="left", font=dict(size=22)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=16, color="#222"),
        width=1500,
        height=560,
        margin=dict(l=80, r=40, t=110, b=120),
    )
    fig.update_annotations(font_size=22)  # subplot titles
    fig.update_xaxes(
        tickangle=-22, tickfont=dict(size=16),
        showline=True, linewidth=1, linecolor="#222", ticks="outside",
    )
    fig.update_yaxes(
        title_text=axis_title, title_font_size=18, tickfont=dict(size=14),
        showline=True, linewidth=1, linecolor="#222", ticks="outside",
        gridcolor="#e6e6e6", zeroline=False,
    )
    return fig


def save_fig(fig, basename):
    pdf_path = OUT_DIR / f"{basename}.pdf"
    png_path = OUT_DIR / f"{basename}.png"
    fig.write_image(str(pdf_path), format="pdf")
    fig.write_image(str(png_path), format="png", scale=2.0)
    print(f"  saved: {pdf_path.name} + {png_path.name}")


def main():
    print("Building slim 5-model charts (best hybrid per dataset)...")

    fig = build_bar_figure(
        "tmap", axis_title="test T-mAP, %", scale=100.0, decimals=2,
        title="T-mAP (выше — лучше)",
    )
    save_fig(fig, "slim5_tmap")

    fig = build_bar_figure(
        "otd", axis_title="test OTD", decimals=2,
        title="OTD (ниже — лучше)",
    )
    save_fig(fig, "slim5_otd")

    fig = build_bar_figure(
        "next_acc", axis_title="test next-item accuracy, %", scale=100.0, decimals=2,
        title="Next-item accuracy (выше — лучше)",
    )
    save_fig(fig, "slim5_next_acc")

    fig = build_bar_figure(
        "entropy", axis_title="entropy предсказанных меток", decimals=3,
        add_target_line=True,
        title="Энтропия предсказанных меток (ближе к истинной энтропии таргета — лучше)",
    )
    save_fig(fig, "slim5_entropy")

    print("Done.")


if __name__ == "__main__":
    main()
