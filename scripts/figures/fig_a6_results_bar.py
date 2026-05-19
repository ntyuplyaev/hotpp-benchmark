"""Сравнение метрики T-mAP всех канонических a6-вариантов с
существующими методами (DeTPP, Diffusion-GRU).

Источник чисел: для существующих методов и a6-вариантов берётся test/T-mAP
из соответствующего yaml-файла результатов.
"""
from pathlib import Path

import plotly.graph_objects as go
import yaml


ROOT = Path(__file__).resolve().parents[2]
RESULTS_CANONICAL = ROOT / "experiments" / "stackoverflow" / "canonical" / "results"
RESULTS_LEGACY = ROOT / "experiments" / "stackoverflow" / "results"
OUT = ROOT / "figures"

# (источник, имя файла, подпись для оси, цвет)
SERIES = [
    (RESULTS_CANONICAL, "b2_detpp_baseline_v1",
     "DeTPP", "#2a78b8"),
    (RESULTS_LEGACY, "diffusion_gru",
     "Diffusion-GRU", "#7e57c2"),
    (RESULTS_CANONICAL, "a6_query_only_decoder_v3_paper_detection",
     "Базовая модель", "#e08a3c"),
    (RESULTS_CANONICAL, "a6_query_only_decoder_v4_horizon_mask",
     "Маска past-horizon", "#888"),
    (RESULTS_CANONICAL, "a6_query_only_decoder_v5_no_padding",
     "Без mask-эмбеддинга", "#888"),
    (RESULTS_CANONICAL, "a6_query_only_decoder_v13_random_padding_paper_detection",
     "Random padding", "#888"),
    (RESULTS_CANONICAL, "a6_query_only_decoder_v10_latent_only_paper_detection",
     "Только латент", "#888"),
    (RESULTS_CANONICAL, "a6_query_only_decoder_v9_latent_context_paper_detection",
     "Латент + контекст", "#888"),
    (RESULTS_CANONICAL, "a6_query_only_decoder_v8_full_components_paper_detection",
     "Латент + query + контекст", "#888"),
    (RESULTS_CANONICAL, "a6_query_only_decoder_v11_matching_padding_paper_detection",
     "Matching padding", "#888"),
    (RESULTS_CANONICAL, "a6_query_only_decoder_v12_far_target_paper_detection",
     "Far-target", "#888"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    labels, tmap_vals, colors = [], [], []
    for src, cfg_name, label, color in SERIES:
        path = src / f"{cfg_name}.yaml"
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        labels.append(label)
        tmap_vals.append(float(data["test/T-mAP"]))
        colors.append(color)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=tmap_vals,
        marker=dict(color=colors, line=dict(color="#333", width=0.6)),
        text=[f"{v:.3f}" for v in tmap_vals],
        textposition="outside",
        textfont=dict(size=11),
        showlegend=False,
    ))

    fig.update_layout(
        xaxis=dict(
            title=None,
            tickangle=-30,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title=dict(text="test T-mAP", font=dict(size=12)),
            tickfont=dict(size=10),
            range=[0.0, max(tmap_vals) * 1.15],
            showgrid=True, gridcolor="#e7e7e7",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=20, t=20, b=110),
        width=1100,
        height=520,
    )

    pdf_path = OUT / "method_a6_results.pdf"
    png_path = OUT / "method_a6_results.png"
    fig.write_image(str(pdf_path), format="pdf")
    fig.write_image(str(png_path), format="png", scale=2)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
