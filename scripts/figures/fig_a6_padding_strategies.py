"""Сравнение четырёх стратегий построения целевой последовательности
диффузионной ветви на одном горизонте.

Для наглядности показано K = 12 слотов; на StackOverflow в работе K = 48.
"""
from pathlib import Path

import plotly.graph_objects as go


# Цветовая схема
C_REAL_NEAR  = "#7fb069"  # реальное событие, t ≤ H
C_REAL_FAR   = "#f6c570"  # реальное событие, t > H
C_OOS_RAW    = "#cccccc"  # out-of-sequence, сырой выход embedder'а
C_MASK       = "#cd5c5c"  # mask_embedding (обучаемое отсутствие события)


# Сценарий: 12 слотов. 4 события в горизонте, 2 за пределами горизонта,
# 6 out-of-sequence (последовательность короче чем K).
SLOT_TYPES = ["near", "near", "near", "near",
              "far",  "far",
              "oos",  "oos",  "oos",  "oos",  "oos",  "oos"]
N_SLOTS = len(SLOT_TYPES)


STRATEGIES = [
    # (заголовок, функция получения цвета слота k по типу)
    (
        "Опорная стратегия:<br>mask_emb только для out-of-sequence",
        {"near": C_REAL_NEAR, "far": C_REAL_FAR, "oos": C_MASK},
    ),
    (
        "Маска past-horizon:<br>замещение всех событий с t > H",
        {"near": C_REAL_NEAR, "far": C_MASK,     "oos": C_MASK},
    ),
    (
        "Без mask_emb:<br>out-of-sequence остаются сырыми",
        {"near": C_REAL_NEAR, "far": C_REAL_FAR, "oos": C_OOS_RAW},
    ),
    (
        "Random padding:<br>дополнительная случайная подмена",
        # near/far отображаются как mask с условной вероятностью; покажем 1/4 near как mask
        # и 1/2 far как mask, чтобы показать характер регуляризации.
        {"near": [C_REAL_NEAR, C_REAL_NEAR, C_REAL_NEAR, C_MASK],
         "far":  [C_REAL_FAR, C_MASK],
         "oos":  C_MASK},
    ),
]


def color_for(strategy_map, slot_type, idx_within_type):
    val = strategy_map[slot_type]
    if isinstance(val, list):
        return val[idx_within_type % len(val)]
    return val


def build_figure():
    shapes = []
    annotations = []

    box_w = 0.85
    box_h = 0.7
    row_gap = 0.6
    x_offset = 1.5

    # Подсчитываем индексы внутри типа для random-padding варианта
    row_y = []
    for row_idx, (title, strat) in enumerate(STRATEGIES):
        y_top = (len(STRATEGIES) - row_idx - 1) * (box_h + row_gap) + 0.4
        row_y.append(y_top)

        type_seen = {"near": 0, "far": 0, "oos": 0}
        for k, t in enumerate(SLOT_TYPES):
            color = color_for(strat, t, type_seen[t])
            type_seen[t] += 1
            x0 = x_offset + k * (box_w + 0.08)
            shapes.append(dict(
                type="rect",
                x0=x0, y0=y_top, x1=x0 + box_w, y1=y_top + box_h,
                line=dict(color="#333", width=0.6),
                fillcolor=color,
            ))
            annotations.append(dict(
                x=x0 + box_w / 2, y=y_top + box_h / 2,
                text=str(k + 1),
                showarrow=False,
                font=dict(size=10, color="#222"),
            ))

        # Заголовок строки
        annotations.append(dict(
            x=0.05, y=y_top + box_h / 2,
            text=title,
            showarrow=False,
            xanchor="left",
            font=dict(size=11.5, color="#222", family="Segoe UI, Arial"),
            align="left",
        ))

    # Подпись ось k = 1..K
    last_y = row_y[-1] - 0.55
    for k in range(N_SLOTS):
        x = x_offset + k * (box_w + 0.08) + box_w / 2
        annotations.append(dict(
            x=x, y=last_y,
            text=f"k = {k+1}",
            showarrow=False,
            font=dict(size=9, color="#666"),
        ))

    # Легенда — простыми прямоугольниками сверху
    legend_y = row_y[0] + box_h + 0.45
    legend_items = [
        (C_REAL_NEAR, "событие в горизонте (t ≤ H)"),
        (C_REAL_FAR,  "событие за горизонтом (t > H)"),
        (C_OOS_RAW,   "out-of-sequence, сырой выход"),
        (C_MASK,      "mask_embedding"),
    ]
    legend_x = x_offset
    for color, label in legend_items:
        shapes.append(dict(
            type="rect",
            x0=legend_x, y0=legend_y, x1=legend_x + 0.4, y1=legend_y + 0.4,
            line=dict(color="#333", width=0.6),
            fillcolor=color,
        ))
        annotations.append(dict(
            x=legend_x + 0.5, y=legend_y + 0.2,
            text=label,
            showarrow=False,
            xanchor="left",
            font=dict(size=10, color="#222"),
        ))
        legend_x += 3.0

    fig = go.Figure()
    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(visible=False, range=[0.0, 13.5]),
        yaxis=dict(visible=False, range=[-0.4, row_y[0] + box_h + 1.2], scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=1200,
        height=520,
    )
    return fig


def main():
    out_dir = Path(__file__).resolve().parents[2] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    pdf_path = out_dir / "method_a6_padding_strategies.pdf"
    png_path = out_dir / "method_a6_padding_strategies.png"
    fig.write_image(str(pdf_path), format="pdf")
    fig.write_image(str(png_path), format="png", scale=2)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
