# -*- coding: utf-8 -*-
"""
Иллюстрация структуры одного объекта набора данных в задаче
прогнозирования последовательностей событий: наблюдаемая история до
момента t и горизонт длительности H, в который попадают события,
подлежащие предсказанию.

Реализовано на plotly. Экспорт в PDF/PNG через kaleido.

Outputs:
    figures/event_sequence_structure.pdf
    figures/event_sequence_structure.png
    figures/event_sequence_structure.html
"""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NAVY = "#0F2C68"
ACCENT = "#3B5FAA"
ACCENT_LIGHT = "#E1E9F6"
ORANGE = "#C25A12"
ORANGE_LIGHT = "#FBE9D6"
GRAY = "#9CA3AF"
DARK = "#1F2430"

HISTORY = [(0.6, 1), (1.6, 3), (2.6, 2), (3.6, 1), (4.4, 4)]
FUTURE = [(5.7, 2), (6.7, 4), (7.8, 1)]
CURRENT_T = 5.0
HORIZON_END = 8.5
T_MIN, T_MAX = 0.0, 9.4

ZONE_TOP = 1.4
ZONE_BOT = -1.6
TIMELINE_Y = 0.0
TIME_LABEL_Y = -0.55
EVENT_PX = 56


def _add_zone(fig, x0, x1, color, label, label_color, label_y):
    fig.add_shape(
        type="rect", x0=x0, x1=x1, y0=ZONE_BOT, y1=ZONE_TOP,
        line=dict(width=0), fillcolor=color, opacity=0.55, layer="below",
    )
    fig.add_annotation(
        x=(x0 + x1) / 2, y=label_y,
        text=label, showarrow=False,
        font=dict(size=18, color=label_color, family="Times New Roman"),
        bgcolor="white",
        bordercolor=label_color, borderwidth=1.2, borderpad=8,
    )


def _add_events(fig, points, fill_color, time_offset):
    xs = [p[0] for p in points]
    ys = [TIMELINE_Y] * len(points)
    type_text = [f"$k_{{{p[1]}}}$" for p in points]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(size=EVENT_PX, color=fill_color,
                    line=dict(color="white", width=2.5)),
        text=type_text, textposition="middle center",
        textfont=dict(size=16, color="white", family="Times New Roman"),
        hoverinfo="skip", showlegend=False,
    ))
    time_xs = xs
    time_ys = [TIME_LABEL_Y] * len(points)
    time_text = [f"$t_{{{i + time_offset}}}$" for i in range(1, len(points) + 1)]
    fig.add_trace(go.Scatter(
        x=time_xs, y=time_ys, mode="text",
        text=time_text, textposition="middle center",
        textfont=dict(size=16, color=DARK, family="Times New Roman"),
        hoverinfo="skip", showlegend=False,
    ))


def main():
    fig = go.Figure()

    _add_zone(fig, T_MIN, CURRENT_T, ACCENT_LIGHT,
              "наблюдаемая история", ACCENT, ZONE_TOP - 0.18)
    _add_zone(fig, CURRENT_T, HORIZON_END, ORANGE_LIGHT,
              r"$\text{горизонт длительности } H$", ORANGE,
              ZONE_TOP - 0.18)

    fig.add_shape(
        type="line", x0=T_MIN - 0.15, x1=T_MAX, y0=TIMELINE_Y, y1=TIMELINE_Y,
        line=dict(color=GRAY, width=1.6), layer="below",
    )
    fig.add_annotation(
        ax=T_MAX - 0.4, ay=TIMELINE_Y, x=T_MAX, y=TIMELINE_Y,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1.4, arrowwidth=1.6,
        arrowcolor=GRAY, text="",
    )
    fig.add_annotation(
        x=T_MAX + 0.18, y=TIMELINE_Y, text="<i>time</i>",
        showarrow=False, xanchor="left",
        font=dict(size=18, color=DARK, family="Times New Roman"),
    )

    fig.add_shape(
        type="line", x0=CURRENT_T, x1=CURRENT_T,
        y0=ZONE_BOT + 0.1, y1=ZONE_TOP - 0.05,
        line=dict(color=NAVY, width=2, dash="dash"),
    )
    fig.add_annotation(
        x=CURRENT_T, y=ZONE_BOT - 0.05, text="$t$",
        showarrow=False, yanchor="top",
        font=dict(size=20, color=NAVY, family="Times New Roman"),
    )
    fig.add_shape(
        type="line", x0=HORIZON_END, x1=HORIZON_END,
        y0=ZONE_BOT + 0.25, y1=ZONE_TOP - 0.25,
        line=dict(color=ORANGE, width=1.6, dash="dot"),
    )
    fig.add_annotation(
        x=HORIZON_END, y=ZONE_BOT - 0.05, text="$t + H$",
        showarrow=False, yanchor="top",
        font=dict(size=18, color=ORANGE, family="Times New Roman"),
    )

    _add_events(fig, HISTORY, ACCENT, time_offset=0)
    _add_events(fig, FUTURE, ORANGE, time_offset=len(HISTORY))

    fig.update_layout(
        width=1300, height=420,
        margin=dict(l=20, r=80, t=10, b=80),
        paper_bgcolor="white", plot_bgcolor="white",
        showlegend=False,
        xaxis=dict(
            visible=False, range=[T_MIN - 0.4, T_MAX + 0.7],
            fixedrange=True, constrain="domain",
        ),
        yaxis=dict(
            visible=False, range=[ZONE_BOT - 0.55, ZONE_TOP + 0.05],
            fixedrange=True,
        ),
        font=dict(family="Times New Roman", size=15, color=DARK),
    )

    pdf_path = OUT_DIR / "event_sequence_structure.pdf"
    png_path = OUT_DIR / "event_sequence_structure.png"
    html_path = OUT_DIR / "event_sequence_structure.html"
    fig.write_image(str(pdf_path), format="pdf", scale=2)
    fig.write_image(str(png_path), format="png", scale=2)
    fig.write_html(str(html_path), include_mathjax="cdn")
    print(f"saved {pdf_path}")
    print(f"saved {png_path}")
    print(f"saved {html_path}")


if __name__ == "__main__":
    main()
