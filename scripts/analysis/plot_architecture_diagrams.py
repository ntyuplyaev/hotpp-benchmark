"""Generate Plotly architecture diagrams used in the diploma text.

The diagrams intentionally mimic the clean schematic style of DeTPP Figure 2:
thin black outlines, white background, muted gray/yellow/orange blocks, and
compact prediction-head callouts.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[2]

BLACK = "#111111"
GRAY = "#d9d9d9"
LIGHT_GRAY = "#f2f2f2"
YELLOW = "#f4ef63"
ORANGE = "#f2bd70"
BLUE = "#dcecff"
GREEN = "#dff1e8"
WHITE = "#ffffff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "text_final" / "figures")
    parser.add_argument("--formats", default="html,pdf")
    return parser.parse_args()


def base_figure(width: int = 1120, height: int = 460,
                yrange: tuple[float, float] = (0, 6)) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        width=width,
        height=height,
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        margin=dict(l=18, r=18, t=18, b=18),
        font=dict(family="Times New Roman", size=20, color=BLACK),
        showlegend=False,
    )
    fig.update_xaxes(visible=False, range=[0, 16], fixedrange=True)
    fig.update_yaxes(visible=False, range=list(yrange), fixedrange=True, scaleanchor="x", scaleratio=1)
    return fig


def add_rect(fig: go.Figure, x0: float, y0: float, x1: float, y1: float, *,
             fill: str = WHITE, line: str = BLACK, width: float = 1.6,
             radius: float | None = None, dash: str | None = None) -> None:
    fig.add_shape(
        type="rect",
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        line=dict(color=line, width=width, dash=dash),
        fillcolor=fill,
        layer="below",
    )


def add_circle(fig: go.Figure, x: float, y: float, r: float, *,
               fill: str = WHITE, line: str = BLACK, width: float = 1.6) -> None:
    fig.add_shape(
        type="circle",
        x0=x - r,
        y0=y - r,
        x1=x + r,
        y1=y + r,
        line=dict(color=line, width=width),
        fillcolor=fill,
        layer="below",
    )


def add_line(fig: go.Figure, x0: float, y0: float, x1: float, y1: float, *,
             dash: str | None = None, width: float = 1.6) -> None:
    fig.add_shape(
        type="line",
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        line=dict(color=BLACK, width=width, dash=dash),
        layer="below",
    )


def add_text(fig: go.Figure, x: float, y: float, text: str, *,
             size: int = 20, anchor: str = "middle center") -> None:
    fig.add_annotation(
        x=x,
        y=y,
        text=text,
        showarrow=False,
        font=dict(family="Times New Roman", size=size, color=BLACK),
        xanchor=anchor.split()[1],
        yanchor=anchor.split()[0],
        align="center",
    )


def add_arrow(fig: go.Figure, x0: float, y0: float, x1: float, y1: float, *,
              width: float = 1.5, dash: str | None = None) -> None:
    fig.add_annotation(
        x=x1,
        y=y1,
        ax=x0,
        ay=y0,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.0,
        arrowwidth=width,
        arrowcolor=BLACK,
        standoff=1,
        startstandoff=1,
    )
    if dash:
        add_line(fig, x0, y0, x1, y1, dash=dash, width=width)


def event_square(fig: go.Figure, x: float, y: float, color: str = GRAY, size: float = 0.38) -> None:
    add_rect(fig, x - size / 2, y - size / 2, x + size / 2, y + size / 2, fill=color)


def latent_bar(fig: go.Figure, x: float, y: float, *, color: str = YELLOW,
               w: float = 0.28, h: float = 0.70) -> None:
    add_rect(fig, x - w / 2, y - h / 2, x + w / 2, y + h / 2, fill=color)


def rounded_box(fig: go.Figure, x: float, y: float, w: float, h: float, text: str,
                *, fill: str = LIGHT_GRAY, size: int = 20) -> None:
    # Plotly shape corner radius is unavailable in older Kaleido; use a plain
    # rectangle for robust PDF export and keep the DeTPP-style thin outline.
    add_rect(fig, x - w / 2, y - h / 2, x + w / 2, y + h / 2, fill=fill)
    add_text(fig, x, y, text, size=size)


def head_box(fig: go.Figure, x: float, y: float, label: str) -> None:
    rounded_box(fig, x, y, 0.58, 0.52, label, fill=LIGHT_GRAY, size=18)


def prediction_head_callout(fig: go.Figure, *, x0: float = 11.35, y0: float = 0.68,
                            title: str = "Prediction head",
                            input_labels: tuple[str, ...] = ("ẑₖ",)) -> None:
    add_rect(fig, x0, y0, x0 + 4.05, y0 + 4.15, fill="#fbfbfb", width=1.7)
    add_text(fig, x0 + 2.0, y0 + 0.28, title, size=20)

    if len(input_labels) == 1:
        latent_bar(fig, x0 + 0.62, y0 + 2.20, color=YELLOW, h=1.15)
        add_text(fig, x0 + 0.26, y0 + 2.20, input_labels[0], size=14, anchor="middle right")
        source_points = [(x0 + 0.78, y0 + 2.20)]
    else:
        colors = [YELLOW, BLUE, GREEN]
        ys = [y0 + 2.80, y0 + 2.10, y0 + 1.40]
        source_points = []
        for label, color, yy in zip(input_labels, colors, ys):
            latent_bar(fig, x0 + 0.62, yy, color=color, w=0.23, h=0.55)
            add_text(fig, x0 + 0.26, yy, label, size=14, anchor="middle right")
            source_points.append((x0 + 0.78, yy))

    outputs = [
        (y0 + 3.25, "σ", "ôₖ"),
        (y0 + 2.15, "", "τ̂ₖ"),
        (y0 + 1.05, "SM", "p̂ₖ(l)"),
    ]
    for yy, circle_label, out_label in outputs:
        tri_x = x0 + 1.55
        add_rect(fig, tri_x - 0.17, yy - 0.35, tri_x + 0.17, yy + 0.35, fill=WHITE)
        for sx, sy in source_points:
            add_arrow(fig, sx, sy, tri_x - 0.18, yy, width=1.2)
        if circle_label:
            add_circle(fig, x0 + 2.35, yy, 0.31, fill=WHITE)
            add_text(fig, x0 + 2.35, yy, circle_label, size=16)
            add_arrow(fig, tri_x + 0.20, yy, x0 + 2.04, yy, width=1.2)
            add_arrow(fig, x0 + 2.66, yy, x0 + 3.25, yy, width=1.2)
        else:
            add_arrow(fig, tri_x + 0.20, yy, x0 + 3.25, yy, width=1.2)
        add_text(fig, x0 + 3.45, yy, out_label, size=19, anchor="middle left")


def add_common_left(fig: go.Figure, *, encoder_label: str = "History encoder",
                    output_label: str = "h_p") -> tuple[float, float]:
    for x in [0.75, 1.65, 2.35]:
        event_square(fig, x, 1.20, GRAY)
    add_text(fig, 1.55, 0.58, "Historical events", size=20)
    rounded_box(fig, 1.55, 2.28, 2.65, 0.70, encoder_label, fill=LIGHT_GRAY, size=20)
    for x in [0.75, 1.65, 2.35]:
        add_arrow(fig, x, 1.42, x, 1.92)
    latent_bar(fig, 3.05, 3.72, color=YELLOW, w=0.22, h=1.05)
    add_text(fig, 2.74, 3.72, output_label, size=18, anchor="middle right")
    add_arrow(fig, 2.40, 2.52, 3.05, 3.17)
    return 3.05, 3.72


def add_slots_and_loss(fig: go.Figure, *, denoiser_x: float = 4.75,
                       slot_input: str = "z", query: bool = False,
                       context: bool = False) -> None:
    xs = [6.15, 7.35, 8.55, 10.75]
    labels = ["H1", "H2", "H3", "HK"]
    add_line(fig, denoiser_x + 1.05, 4.88, 10.75, 4.88, width=1.3)
    add_arrow(fig, denoiser_x + 0.85, 3.72, denoiser_x + 1.05, 4.88, width=1.1)
    for x, label in zip(xs, labels):
        latent_bar(fig, x, 3.05, color=YELLOW)
        if query:
            latent_bar(fig, x, 3.72, color=BLUE, w=0.25, h=0.45)
        if context:
            latent_bar(fig, x, 4.22, color=GREEN, w=0.25, h=0.45)
        head_box(fig, x, 4.72, label)
        add_arrow(fig, x, 4.88, x, 3.40, width=1.0)
        add_arrow(fig, x, 3.40, x, 4.42, width=1.2)
    add_text(fig, 9.62, 3.76, "...", size=23)

    rounded_box(fig, 8.45, 1.78, 2.45, 0.65, "DeTPP matching loss", fill=WHITE, size=18)
    for x in xs:
        add_line(fig, x, 4.45, 8.45, 2.10, width=1.2)
    for x in [6.0, 7.15, 8.30, 9.45, 10.60]:
        event_square(fig, x, 0.88, ORANGE)
        add_line(fig, x, 1.10, 8.45, 1.45, width=1.2)
    add_text(fig, 8.30, 0.30, "Future events", size=20)


def diagram_detpp_architecture() -> go.Figure:
    fig = base_figure()
    hp_x, hp_y = add_common_left(fig, encoder_label="Backbone", output_label="Embedding")

    xs = [4.80, 5.95, 7.10, 10.20]
    labels = ["H1", "H2", "H3", "HK"]
    add_line(fig, hp_x, 4.88, 10.20, 4.88, width=1.3)
    add_arrow(fig, hp_x, hp_y + 0.55, hp_x, 4.88, width=1.1)
    for x, label in zip(xs, labels):
        head_box(fig, x, 4.32, label)
        latent_bar(fig, x, 3.05, color=YELLOW)
        add_arrow(fig, x, 4.88, x, 4.58, width=1.0)
        add_arrow(fig, x, 4.06, x, 3.40, width=1.2)
    add_text(fig, 8.45, 3.78, "...", size=23)

    rounded_box(fig, 7.25, 1.78, 2.45, 0.65, "Matching loss", fill=WHITE, size=18)
    for x in xs:
        add_line(fig, x, 2.78, 7.25, 2.10, width=1.2)
    for x in [5.00, 6.15, 7.30, 8.45, 9.60]:
        event_square(fig, x, 0.88, ORANGE)
        add_line(fig, x, 1.10, 7.25, 1.45, width=1.2)
    add_text(fig, 7.25, 0.30, "Future events", size=20)

    add_line(fig, 10.20, 4.32, 11.35, 4.83, dash="dash", width=1.4)
    add_line(fig, 10.20, 4.32, 11.35, 0.82, dash="dash", width=1.4)
    prediction_head_callout(fig, x0=11.35, y0=0.68, title="Prediction head",
                            input_labels=("h_p", "q_k"))
    return fig


def diagram_hybrid_pipeline() -> go.Figure:
    fig = base_figure()
    hp_x, hp_y = add_common_left(fig, encoder_label="History encoder")
    rounded_box(fig, 4.75, 3.72, 2.30, 0.75, "Diffusion denoiser", fill=LIGHT_GRAY, size=19)
    add_arrow(fig, hp_x + 0.18, hp_y, 3.60, 3.72)
    add_slots_and_loss(fig, denoiser_x=4.75, slot_input="ẑₖ")
    add_line(fig, 10.75, 4.72, 11.35, 4.83, dash="dash", width=1.4)
    add_line(fig, 10.75, 4.72, 11.35, 0.82, dash="dash", width=1.4)
    prediction_head_callout(fig, x0=11.35, y0=0.68, title="Prediction head", input_labels=("ẑₖ",))
    return fig


def diagram_hybrid_baseline() -> go.Figure:
    fig = base_figure()
    hp_x, hp_y = add_common_left(fig, encoder_label="GRU encoder")
    rounded_box(fig, 4.75, 3.72, 2.35, 0.88, "BiGRU diffusion<br>denoiser", fill=LIGHT_GRAY, size=18)
    add_arrow(fig, hp_x + 0.18, hp_y, 3.58, 3.72)
    add_slots_and_loss(fig, denoiser_x=4.75, slot_input="ẑₖ", query=True)
    add_line(fig, 10.75, 4.72, 11.35, 4.83, dash="dash", width=1.4)
    add_line(fig, 10.75, 4.72, 11.35, 0.82, dash="dash", width=1.4)
    prediction_head_callout(fig, x0=11.35, y0=0.68, title="Slot decoder",
                            input_labels=("ẑₖ", "qₖ"))
    return fig


def diagram_matching_padding() -> go.Figure:
    fig = base_figure(height=500, yrange=(0, 8))
    steps = [
        (2.10, 5.65, "Reverse diffusion<br><span style='font-size:15px'>no gradients</span>", LIGHT_GRAY),
        (5.05, 5.65, "Slot decoder<br><span style='font-size:15px'>K candidates</span>", LIGHT_GRAY),
        (8.00, 5.65, "Hungarian<br>matching", WHITE),
        (5.05, 3.25, "Rebuild target<br>X0 -> X0_tilde", BLUE),
        (8.00, 3.25, "Noise/denoise<br>training step", GREEN),
    ]
    for x, y, text, fill in steps:
        rounded_box(fig, x, y, 2.35, 0.82, text, fill=fill, size=18)
    add_arrow(fig, 3.30, 5.65, 3.85, 5.65)
    add_arrow(fig, 6.25, 5.65, 6.80, 5.65)
    add_arrow(fig, 8.00, 5.20, 5.25, 3.72)
    add_text(fig, 6.35, 4.18, "σ*", size=17)
    add_arrow(fig, 6.25, 3.25, 6.80, 3.25)

    for x in [4.25, 4.80, 5.35, 5.90]:
        latent_bar(fig, x, 6.75, color=YELLOW, w=0.22, h=0.45)
        head_box(fig, x, 7.35, "")
        add_arrow(fig, x, 6.98, x, 7.08, width=1.0)
    add_text(fig, 5.08, 7.85, "Predicted slots", size=18)

    for x in [7.10, 7.60, 8.10, 8.60, 9.10]:
        event_square(fig, x, 4.55, ORANGE, size=0.34)
    add_text(fig, 8.10, 4.05, "Future events in horizon", size=18)

    for x, color, label in [(4.15, YELLOW, "matched -> z_j"), (5.10, YELLOW, ""), (6.05, LIGHT_GRAY, "unmatched -> m")]:
        latent_bar(fig, x, 2.00, color=color, w=0.25, h=0.55)
    add_text(fig, 5.15, 1.35, "Realigned diffusion target", size=18)
    add_text(fig, 5.15, 0.86, "matched slots -> event embeddings", size=15)
    add_text(fig, 5.15, 0.55, "unmatched slots -> mask embedding", size=15)

    add_rect(fig, 10.05, 2.25, 15.25, 6.42, fill="#fbfbfb")
    add_text(fig, 12.65, 2.65, "Training target after matching", size=19)
    for i, (x, color, top) in enumerate([(11.0, YELLOW, "z_j"), (12.0, LIGHT_GRAY, "m"), (13.0, YELLOW, "z_j"), (14.0, LIGHT_GRAY, "m")]):
        latent_bar(fig, x, 4.65, color=color, w=0.28, h=0.72)
        head_box(fig, x, 5.55, f"H{i + 1}")
        add_arrow(fig, x, 4.98, x, 5.28, width=1.0)
        add_text(fig, x, 3.78, top.replace("_", ""), size=17)
    return fig


def diagram_far_target() -> go.Figure:
    fig = base_figure(height=420, yrange=(0, 5.5))
    add_arrow(fig, 0.70, 2.00, 15.30, 2.00, width=1.5)
    add_text(fig, 15.55, 2.00, "t", size=20)

    for x in [1.25, 2.25, 3.05, 4.35, 5.30]:
        add_circle(fig, x, 2.00, 0.16, fill=GRAY)
    add_text(fig, 3.25, 3.12, "Observed history", size=20)

    for x, label in [(6.15, "t<sub>p</sub>"), (9.80, "t<sub>p</sub> + H"), (13.90, "t<sub>p</sub> + H + H<sub>far</sub>")]:
        add_line(fig, x, 1.35, x, 4.80, dash="dash", width=1.4)
        add_text(fig, x, 5.05, label, size=18)

    add_rect(fig, 6.15, 2.45, 9.80, 3.40, fill="#dcecff", line="#2a78b8", width=1.4)
    add_text(fig, 7.98, 3.82, "Detection window", size=20)
    add_text(fig, 7.98, 2.92, "L<sub>DeTPP</sub>", size=19)
    for x in [6.75, 7.35, 8.30, 9.05]:
        event_square(fig, x, 2.02, ORANGE, size=0.32)

    add_rect(fig, 9.80, 2.45, 13.90, 3.40, fill="#fff1cf", line="#d18a22", width=1.4)
    add_text(fig, 11.85, 3.82, "Diffusion target window", size=20)
    add_text(fig, 11.85, 2.92, "L<sub>diff</sub>", size=19)
    for x in [10.45, 11.10, 12.25, 13.20]:
        latent_bar(fig, x, 2.02, color=YELLOW, w=0.24, h=0.55)

    rounded_box(fig, 3.25, 0.80, 2.40, 0.62, "Shared history encoder", fill=LIGHT_GRAY, size=18)
    add_arrow(fig, 4.45, 0.80, 7.95, 2.45, width=1.0)
    add_arrow(fig, 4.45, 0.80, 11.85, 2.45, width=1.0)
    return fig


# =====================================================================
# v2 diagrams: bigger components, terse labels, DeTPP-style layout.
# =====================================================================

V2_FONT = "Times New Roman"
V2_TEXT = "#1a1a1a"
V2_EDGE = "#222222"
V2_ENC = "#eef3f8"
V2_DIFF = "#fff3d6"
V2_DEC = "#e3eefc"
V2_LAT = "#fde6a4"
V2_QRY = "#cbdcf7"
V2_HEAD = "#ececec"
V2_HIST = "#cfcfcf"
V2_FUT = "#f6a866"
V2_DETW = "#cfe1ff"
V2_FARW = "#fcd9b0"
V2_STEP = "#f7f7f7"


def _rrect_path(x: float, y: float, w: float, h: float, r: float = 0.12) -> str:
    r = min(r, w / 2.0, h / 2.0)
    return (
        f"M {x + r},{y} L {x + w - r},{y} Q {x + w},{y} {x + w},{y + r} "
        f"L {x + w},{y + h - r} Q {x + w},{y + h} {x + w - r},{y + h} "
        f"L {x + r},{y + h} Q {x},{y + h} {x},{y + h - r} "
        f"L {x},{y + r} Q {x},{y} {x + r},{y} Z"
    )


def v2_rrect(fig, x, y, w, h, fc=V2_ENC, ec=V2_EDGE, lw=2.0, r=0.12):
    fig.add_shape(type="path", path=_rrect_path(x, y, w, h, r),
                  fillcolor=fc, line=dict(color=ec, width=lw))


def v2_rect(fig, x, y, w, h, fc=V2_LAT, ec=V2_EDGE, lw=1.6):
    fig.add_shape(type="rect", x0=x, y0=y, x1=x + w, y1=y + h,
                  fillcolor=fc, line=dict(color=ec, width=lw))


def v2_circle(fig, cx, cy, r, fc=V2_HIST, ec=V2_EDGE, lw=1.6):
    fig.add_shape(type="circle", x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r,
                  fillcolor=fc, line=dict(color=ec, width=lw))


def v2_text(fig, cx, cy, msg, size=24, color=V2_TEXT, weight=None,
            xanchor="center", yanchor="middle"):
    if weight == "bold":
        msg = f"<b>{msg}</b>"
    fig.add_annotation(x=cx, y=cy, text=msg, showarrow=False,
                      font=dict(family=V2_FONT, size=size, color=color),
                      xanchor=xanchor, yanchor=yanchor)


def v2_arrow(fig, x1, y1, x2, y2, lw=2.4, head=3, color=V2_EDGE):
    fig.add_annotation(x=x2, y=y2, ax=x1, ay=y1, xref="x", yref="y",
                      axref="x", ayref="y", showarrow=True, arrowhead=head,
                      arrowsize=1.1, arrowwidth=lw, arrowcolor=color, text="")


def v2_line(fig, x1, y1, x2, y2, lw=1.6, dash=None, color=V2_EDGE):
    fig.add_shape(type="line", x0=x1, y0=y1, x1=x2, y1=y2,
                  line=dict(color=color, width=lw, dash=dash or "solid"))


def v2_canvas(w_units: float, h_units: float, ppu: int = 110) -> go.Figure:
    fig = go.Figure()
    fig.update_xaxes(visible=False, range=[0, w_units])
    fig.update_yaxes(visible=False, range=[0, h_units], scaleanchor="x", scaleratio=1.0)
    fig.update_layout(
        paper_bgcolor=WHITE, plot_bgcolor=WHITE,
        margin=dict(l=10, r=10, t=10, b=10),
        width=int(w_units * ppu), height=int(h_units * ppu),
        showlegend=False,
        font=dict(family=V2_FONT, color=V2_TEXT, size=22),
    )
    return fig


def _v2_latent_column(fig, x, y, w, h, cells=4, fc=V2_LAT):
    """Stacked cells representing a vector. (x,y) is bottom-left."""
    ch = h / cells
    for i in range(cells):
        v2_rect(fig, x, y + i * ch, w, ch * 0.94, fc=fc, lw=1.2)


def _v2_history_block(fig, cx, cy, label="Historical events"):
    xs = [cx - 0.65, cx, cx + 0.65]
    for x in xs:
        v2_circle(fig, x, cy, 0.27, fc=V2_HIST)
    v2_text(fig, xs[-1] + 0.5, cy, "…", size=30)
    v2_text(fig, cx, cy - 0.85, label, size=22)


def _v2_future_block(fig, cx, cy, n=5, label="Future events"):
    half = (n - 1) / 2
    xs = [cx + (i - half) * 0.65 for i in range(n)]
    for x in xs:
        v2_circle(fig, x, cy, 0.27, fc=V2_FUT)
    v2_text(fig, xs[-1] + 0.5, cy, "…", size=30)
    v2_text(fig, cx, cy - 0.85, label, size=22)
    return xs


def _v2_prediction_head_inset(fig, x, y, w, h, *, title, inputs, lat_color=V2_LAT,
                              query_color=V2_QRY):
    """Right-side inset showing the per-head decoder/prediction-head detail."""
    v2_rrect(fig, x, y, w, h, fc="white", lw=1.5, r=0.18)
    v2_text(fig, x + w / 2, y + h - 0.45, title, size=23, weight="bold")
    # Inputs on the left of the inset
    in_x = x + 0.55
    in_w = 0.5
    if len(inputs) == 1:
        lab, col = inputs[0]
        col_h = h * 0.55
        col_y = y + (h - col_h) / 2 - 0.3
        _v2_latent_column(fig, in_x, col_y, in_w, col_h, cells=4, fc=col)
        v2_text(fig, in_x + in_w / 2, col_y - 0.4, lab, size=22)
        anchors = [(in_x + in_w, col_y + col_h / 2)]
    else:
        # two stacked inputs (ẑ_k on top, q_k below)
        gap = 0.25
        col_h_top = h * 0.30
        col_h_bot = h * 0.22
        top_y = y + h - 1.6 - col_h_top
        bot_y = y + 1.1
        _v2_latent_column(fig, in_x, top_y, in_w, col_h_top, cells=4, fc=inputs[0][1])
        v2_text(fig, in_x + in_w / 2, top_y - 0.35, inputs[0][0], size=21)
        _v2_latent_column(fig, in_x, bot_y, in_w, col_h_bot, cells=3, fc=inputs[1][1])
        v2_text(fig, in_x + in_w / 2, bot_y - 0.35, inputs[1][0], size=21)
        anchors = [(in_x + in_w, top_y + col_h_top / 2),
                   (in_x + in_w, bot_y + col_h_bot / 2)]

    # MLP / FC block in the middle
    mlp_x = x + 1.55
    mlp_w = 0.78
    mlp_y = y + 1.05
    mlp_h = h - 2.15
    v2_rrect(fig, mlp_x, mlp_y, mlp_w, mlp_h, fc=V2_DEC, r=0.10)
    v2_text(fig, mlp_x + mlp_w / 2, mlp_y + mlp_h / 2, "MLP", size=21)
    for ax, ay in anchors:
        v2_line(fig, ax, ay, mlp_x, ay, lw=1.5)

    # Three outputs from the right of the MLP
    out_specs = [
        (mlp_y + mlp_h - 0.6, "σ", "ô<sub>k</sub>"),
        (mlp_y + mlp_h / 2, None, "τ̂<sub>k</sub>"),
        (mlp_y + 0.6, "SM", "π̂<sub>k</sub>(l)"),
    ]
    out_x = mlp_x + mlp_w
    for cy, sym, out_lab in out_specs:
        if sym is not None:
            v2_circle(fig, out_x + 0.6, cy, 0.32, fc="white")
            v2_text(fig, out_x + 0.6, cy, sym, size=20)
            v2_line(fig, out_x, cy, out_x + 0.27, cy, lw=1.5)
            v2_line(fig, out_x + 0.92, cy, out_x + 1.25, cy, lw=1.5)
            v2_text(fig, out_x + 1.65, cy, out_lab, size=22, xanchor="left")
        else:
            v2_line(fig, out_x, cy, out_x + 1.25, cy, lw=1.5)
            v2_text(fig, out_x + 1.65, cy, out_lab, size=22, xanchor="left")


# ---------- v2.1: hybrid pipeline ----------

def diagram_hybrid_pipeline_v2() -> go.Figure:
    W, H = 20.0, 9.5
    fig = v2_canvas(W, H)

    # History on the far left
    _v2_history_block(fig, 1.35, 5.6)

    # Encoder
    v2_rrect(fig, 2.95, 4.95, 2.4, 1.35, fc=V2_ENC)
    v2_text(fig, 4.15, 5.62, "Encoder", size=27, weight="bold")
    v2_arrow(fig, 2.05, 5.6, 2.95, 5.6, lw=2.6)

    # h_p column
    hp_x, hp_y, hp_w, hp_h = 5.65, 4.65, 0.55, 1.95
    _v2_latent_column(fig, hp_x, hp_y, hp_w, hp_h, cells=5, fc="#d4dfee")
    v2_text(fig, hp_x + hp_w / 2, hp_y - 0.4, "h<sub>p</sub>", size=24)
    v2_arrow(fig, 5.35, 5.6, hp_x, 5.6, lw=2.4)

    # Diffusion
    v2_rrect(fig, 6.45, 4.95, 2.6, 1.35, fc=V2_DIFF)
    v2_text(fig, 7.75, 5.62, "Diffusion", size=27, weight="bold")
    v2_arrow(fig, hp_x + hp_w, 5.6, 6.45, 5.6, lw=2.4)

    # K latent vectors with heads above
    lat_xs = [9.7, 10.85, 12.0, 13.95]
    lat_w, lat_h = 0.55, 1.8
    lat_y = 4.75
    z_labels = ["ẑ<sub>1</sub>", "ẑ<sub>2</sub>", "ẑ<sub>3</sub>", "ẑ<sub>K</sub>"]
    head_y, head_h, head_w = 7.55, 1.0, 1.15
    head_labels = ["H<sub>1</sub>", "H<sub>2</sub>", "H<sub>3</sub>", "H<sub>K</sub>"]
    for x, zl, hl in zip(lat_xs, z_labels, head_labels):
        _v2_latent_column(fig, x, lat_y, lat_w, lat_h, cells=4, fc=V2_LAT)
        v2_text(fig, x + lat_w / 2, lat_y - 0.42, zl, size=22)
        v2_rrect(fig, x + lat_w / 2 - head_w / 2, head_y, head_w, head_h, fc=V2_HEAD, r=0.12)
        v2_text(fig, x + lat_w / 2, head_y + head_h / 2, hl, size=25, weight="bold")
        v2_arrow(fig, x + lat_w / 2, lat_y + lat_h + 0.05, x + lat_w / 2, head_y - 0.05, lw=2.0)
    # ellipsis between 3rd and 4th latents
    mid_x = (lat_xs[2] + lat_xs[3]) / 2 + lat_w / 2
    v2_text(fig, mid_x, lat_y + lat_h / 2, "⋯", size=36)
    v2_text(fig, mid_x, head_y + head_h / 2, "⋯", size=30)

    # Fan-out arrows from diffusion to each latent
    for x in lat_xs:
        v2_arrow(fig, 9.05, 5.6, x + lat_w / 2, lat_y + lat_h + 0.05, lw=1.6, head=3)

    # Matching loss
    ml_x, ml_y, ml_w, ml_h = 9.7, 2.8, 4.8, 1.0
    v2_rrect(fig, ml_x, ml_y, ml_w, ml_h, fc=V2_DETW, lw=2.2, r=0.10)
    v2_text(fig, ml_x + ml_w / 2, ml_y + ml_h / 2, "Matching loss", size=25, weight="bold")
    for x in lat_xs:
        v2_line(fig, x + lat_w / 2, head_y, x + lat_w / 2, ml_y + ml_h, lw=1.4, color="#666666")

    # Future events
    fut_xs = _v2_future_block(fig, ml_x + ml_w / 2, 1.55, n=5)
    for x in fut_xs:
        v2_line(fig, x, 1.82, x, ml_y, lw=1.4, color="#666666")

    # Inset on the right
    _v2_prediction_head_inset(fig, x=15.7, y=1.5, w=4.05, h=6.4,
                              title="Prediction head",
                              inputs=[("ẑ<sub>k</sub>", V2_LAT)])
    return fig


# ---------- v2.2: hybrid baseline (queries + GRU labels + decoder inset) ----------

def diagram_hybrid_baseline_v2() -> go.Figure:
    W, H = 20.0, 9.5
    fig = v2_canvas(W, H)

    _v2_history_block(fig, 1.35, 5.6)

    # Encoder (GRU)
    v2_rrect(fig, 2.95, 4.95, 2.75, 1.35, fc=V2_ENC)
    v2_text(fig, 4.32, 5.62, "Encoder (GRU)", size=25, weight="bold")
    v2_arrow(fig, 2.05, 5.6, 2.95, 5.6, lw=2.6)

    # h_p
    hp_x, hp_y, hp_w, hp_h = 5.95, 4.65, 0.55, 1.95
    _v2_latent_column(fig, hp_x, hp_y, hp_w, hp_h, cells=5, fc="#d4dfee")
    v2_text(fig, hp_x + hp_w / 2, hp_y - 0.4, "h<sub>p</sub>", size=23)
    v2_arrow(fig, 5.70, 5.6, hp_x, 5.6, lw=2.4)

    # Diffusion (GRU)
    v2_rrect(fig, 6.78, 4.95, 2.75, 1.35, fc=V2_DIFF)
    v2_text(fig, 8.15, 5.62, "Diffusion (GRU)", size=25, weight="bold")
    v2_arrow(fig, hp_x + hp_w, 5.6, 6.78, 5.6, lw=2.4)

    # K columns of stacked [ẑ_k; q_k]: yellow on top, blue below
    lat_xs = [10.20, 11.30, 12.40, 14.20]
    lat_w = 0.55
    z_h = 1.45
    q_h = 1.05
    z_y = 5.30
    q_y = z_y - q_h - 0.05  # small gap separating the two colour zones
    z_labels = ["ẑ<sub>1</sub>", "ẑ<sub>2</sub>", "ẑ<sub>3</sub>", "ẑ<sub>K</sub>"]
    head_y, head_h, head_w = 7.65, 1.0, 1.15
    head_labels = ["H<sub>1</sub>", "H<sub>2</sub>", "H<sub>3</sub>", "H<sub>K</sub>"]
    for x, zl, hl in zip(lat_xs, z_labels, head_labels):
        _v2_latent_column(fig, x, z_y, lat_w, z_h, cells=4, fc=V2_LAT)
        _v2_latent_column(fig, x, q_y, lat_w, q_h, cells=3, fc=V2_QRY)
        v2_text(fig, x + lat_w / 2, z_y + z_h + 0.32, zl, size=21)
        v2_rrect(fig, x + lat_w / 2 - head_w / 2, head_y, head_w, head_h, fc=V2_HEAD, r=0.12)
        v2_text(fig, x + lat_w / 2, head_y + head_h / 2, hl, size=25, weight="bold")
        v2_arrow(fig, x + lat_w / 2, z_y + z_h + 0.55, x + lat_w / 2, head_y - 0.05, lw=1.8)
    # ellipses
    mid_x = (lat_xs[2] + lat_xs[3]) / 2 + lat_w / 2
    v2_text(fig, mid_x, z_y + z_h / 2, "⋯", size=36)
    v2_text(fig, mid_x, q_y + q_h / 2, "⋯", size=32)
    v2_text(fig, mid_x, head_y + head_h / 2, "⋯", size=30)

    # Diffusion → each latent (fanout)
    for x in lat_xs:
        v2_arrow(fig, 9.53, 5.6, x + lat_w / 2, z_y + z_h + 0.05, lw=1.6, head=3)

    # Matching loss
    ml_x, ml_y, ml_w, ml_h = 10.2, 2.55, 4.7, 1.0
    v2_rrect(fig, ml_x, ml_y, ml_w, ml_h, fc=V2_DETW, lw=2.2, r=0.10)
    v2_text(fig, ml_x + ml_w / 2, ml_y + ml_h / 2, "Matching loss", size=25, weight="bold")
    for x in lat_xs:
        v2_line(fig, x + lat_w / 2, q_y, x + lat_w / 2, ml_y + ml_h, lw=1.4, color="#666666")

    # Future events
    fut_xs = _v2_future_block(fig, ml_x + ml_w / 2, 1.30, n=5)
    for x in fut_xs:
        v2_line(fig, x, 1.57, x, ml_y, lw=1.4, color="#666666")

    # Inset on the right — Decoder
    _v2_prediction_head_inset(fig, x=15.7, y=1.4, w=4.0, h=6.6,
                              title="Decoder",
                              inputs=[("ẑ<sub>k</sub>", V2_LAT),
                                      ("q<sub>k</sub>", V2_QRY)])
    return fig


# ---------- v2.3: matching padding ----------

def diagram_matching_padding_v2() -> go.Figure:
    W, H = 15.0, 7.0
    fig = v2_canvas(W, H)

    bw, bh = 3.8, 1.55
    rowtop = 4.4
    rowbot = 1.7

    # Top row: reverse diffusion → decode → matching
    top_boxes = [
        (0.5, "Reverse diffusion", "(no gradient)"),
        (5.3, "Decode", "K candidates"),
        (10.1, "Hungarian", "matching"),
    ]
    for x, l1, l2 in top_boxes:
        v2_rrect(fig, x, rowtop, bw, bh, fc=V2_STEP, r=0.14)
        v2_text(fig, x + bw / 2, rowtop + bh / 2 + 0.35, l1, size=23, weight="bold")
        v2_text(fig, x + bw / 2, rowtop + bh / 2 - 0.32, l2, size=21)

    # Bottom row: rebuild target → training step
    v2_rrect(fig, 5.3, rowbot, bw, bh, fc=V2_STEP, r=0.14)
    v2_text(fig, 5.3 + bw / 2, rowbot + bh / 2 + 0.35, "Rebuild target", size=23, weight="bold")
    v2_text(fig, 5.3 + bw / 2, rowbot + bh / 2 - 0.32, "X<sub>0</sub> → X̃<sub>0</sub>", size=22)

    v2_rrect(fig, 10.1, rowbot, bw, bh, fc=V2_DETW, r=0.14)
    v2_text(fig, 10.1 + bw / 2, rowbot + bh / 2 + 0.35, "Noise / denoise", size=23, weight="bold")
    v2_text(fig, 10.1 + bw / 2, rowbot + bh / 2 - 0.32, "training step", size=21)

    # Arrows in top row
    v2_arrow(fig, 0.5 + bw, rowtop + bh / 2, 5.3, rowtop + bh / 2, lw=2.6)
    v2_arrow(fig, 5.3 + bw, rowtop + bh / 2, 10.1, rowtop + bh / 2, lw=2.6)
    # Arrow from matching down to rebuild target (σ* label well above the line)
    v2_arrow(fig, 10.1 + bw / 2, rowtop, 5.3 + bw / 2, rowbot + bh, lw=2.6)
    v2_text(fig, 9.0, 4.25, "σ<sup>*</sup>", size=28, weight="bold")
    # Arrow rebuild → training
    v2_arrow(fig, 5.3 + bw, rowbot + bh / 2, 10.1, rowbot + bh / 2, lw=2.6)

    # Bottom note
    v2_text(fig, W / 2, 0.85,
            "matched heads → z<sub>j<sub>k</sub></sub>     unmatched heads → m",
            size=22)
    return fig


# ---------- v2.4: far-target time windows ----------

def diagram_far_target_v2() -> go.Figure:
    W, H = 16.5, 5.6
    fig = v2_canvas(W, H)

    axis_y = 1.45
    v2_arrow(fig, 0.35, axis_y, W - 0.35, axis_y, lw=2.4)
    v2_text(fig, W - 0.15, axis_y, "t", size=24, xanchor="left")

    # Observed history circles
    hist_xs = [0.85, 1.75, 2.65, 3.55, 4.45]
    for x in hist_xs:
        v2_circle(fig, x, axis_y, 0.27, fc=V2_HIST)
    v2_text(fig, sum(hist_xs) / len(hist_xs), axis_y + 1.05, "Observed history", size=22)

    # Detection window
    tp_x, tpH_x, tpHHf_x = 5.3, 9.9, 15.3
    v2_line(fig, tp_x, 0.55, tp_x, 4.5, lw=1.6, dash="dash", color="#555555")
    v2_text(fig, tp_x, 4.85, "t<sub>p</sub>", size=24)
    v2_rrect(fig, tp_x, 2.05, tpH_x - tp_x, 1.05, fc=V2_DETW, r=0.10)
    v2_text(fig, (tp_x + tpH_x) / 2, 2.58, "Detection window", size=22, weight="bold")
    v2_text(fig, (tp_x + tpH_x) / 2, 3.55, "ℒ<sub>DeTPP</sub>", size=24)

    v2_line(fig, tpH_x, 0.55, tpH_x, 4.5, lw=1.6, dash="dash", color="#555555")
    v2_text(fig, tpH_x, 4.85, "t<sub>p</sub> + H", size=24)
    v2_rrect(fig, tpH_x, 2.05, tpHHf_x - tpH_x, 1.05, fc=V2_FARW, r=0.10)
    v2_text(fig, (tpH_x + tpHHf_x) / 2, 2.58, "Diffusion target window", size=22, weight="bold")
    v2_text(fig, (tpH_x + tpHHf_x) / 2, 3.55, "ℒ<sub>diff</sub>", size=24)

    v2_line(fig, tpHHf_x, 0.55, tpHHf_x, 4.5, lw=1.6, dash="dash", color="#555555")
    v2_text(fig, tpHHf_x, 4.85, "t<sub>p</sub> + H + H<sub>far</sub>", size=23)

    return fig


def save(fig: go.Figure, out_dir: Path, stem: str, formats: Iterable[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        if fmt == "html":
            fig.write_html(str(path), include_plotlyjs="cdn")
        else:
            fig.write_image(str(path), format=fmt)
        print(path)


def main() -> None:
    args = parse_args()
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    diagrams = {
        "arch_hybrid_pipeline_v2": diagram_hybrid_pipeline_v2(),
        "arch_hybrid_baseline_v2": diagram_hybrid_baseline_v2(),
        "arch_matching_padding_v2": diagram_matching_padding_v2(),
        "arch_far_target_v2": diagram_far_target_v2(),
    }
    for stem, fig in diagrams.items():
        save(fig, args.output_dir, stem, formats)


if __name__ == "__main__":
    main()
