"""Сравнение T-mAP всех канонических Amazon-экспериментов:
два существующих метода (DeTPP, Diffusion-GRU) + 9 a6-вариантов.
"""
from pathlib import Path

import plotly.graph_objects as go
import yaml


ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "amazon" / "canonical" / "results"
OUT = ROOT / "figures"

# (имя конфига, подпись, цвет)
SERIES = [
    ("b2_detpp_baseline_v1",                                                  "DeTPP",                       "#2a78b8"),
    ("b1_pure_diffusion_v1",                                                  "Diffusion-GRU",               "#7e57c2"),
    ("a6_query_only_decoder_v3_paper_detection",                              "Базовая модель",              "#e08a3c"),
    ("a6_query_only_decoder_v4_horizon_mask",                                 "Маска past-horizon",          "#888"),
    ("a6_query_only_decoder_v5_no_padding",                                   "Без mask-эмбеддинга",         "#888"),
    ("a6_query_only_decoder_v13_random_padding_paper_detection",              "Random padding",              "#888"),
    ("a6_query_only_decoder_v10_latent_only_paper_detection",                 "Только латент",               "#888"),
    ("a6_query_only_decoder_v9_latent_context_paper_detection",               "Латент + контекст",           "#888"),
    ("a6_query_only_decoder_v8_full_components_paper_detection",              "Латент + query + контекст",   "#888"),
    ("a6_query_only_decoder_v11_matching_padding_paper_detection",            "Matching padding",            "#888"),
    ("a6_query_only_decoder_v12_far_target_paper_detection",                  "Far-target",                  "#888"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels, vals, colors = [], [], []
    for cfg, label, color in SERIES:
        with open(RES / f"{cfg}.yaml", "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        labels.append(label)
        vals.append(float(data["test/T-mAP"]))
        colors.append(color)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=vals,
        marker=dict(color=colors, line=dict(color="#333", width=0.6)),
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
        textfont=dict(size=11),
        showlegend=False,
    ))
    fig.update_layout(
        xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(title=dict(text="test T-mAP", font=dict(size=12)),
                   range=[0.0, max(vals) * 1.15], showgrid=True, gridcolor="#e7e7e7"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=60, r=20, t=20, b=120),
        width=1200, height=540,
    )
    pdf_path = OUT / "method_amazon_a6_results.pdf"
    png_path = OUT / "method_amazon_a6_results.png"
    fig.write_image(str(pdf_path), format="pdf")
    fig.write_image(str(png_path), format="png", scale=2)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
