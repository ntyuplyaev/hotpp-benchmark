"""Архитектурная схема опорной гибридной архитектуры (a6_v3_paper_detection).

Изображает:
  - кодировщик истории (GRU);
  - диффузионную ветвь: шум, денойзер, K восстановленных латентов;
  - декодер с обучаемыми запросами слотов: per-slot Linear на [z_k; q_k];
  - детекционную функцию потерь с венгерским сопоставлением.

Все числовые размеры тензоров проставлены для StackOverflow-конфигурации:
  D = 33 (event embedding), D_h = 64 (GRU hidden), K = 48 (DeTPP slots),
  d_q = 64 (slot query size), P_base = 24 (1 presence + 22 labels + 1 time).
"""
from pathlib import Path

import plotly.graph_objects as go


def make_box(x, y, w, h, text, fillcolor, font_size=11, text_color="#222"):
    return dict(
        type="rect",
        x0=x, y0=y, x1=x + w, y1=y + h,
        line=dict(color="#444", width=1.2),
        fillcolor=fillcolor,
        layer="below",
    ), dict(
        x=x + w / 2, y=y + h / 2,
        text=text,
        showarrow=False,
        font=dict(size=font_size, color=text_color, family="Segoe UI, Arial"),
        align="center",
    )


def make_arrow(x0, y0, x1, y1, label=None, label_dy=0.0):
    arr = dict(
        x=x1, y=y1, ax=x0, ay=y0,
        xref="x", yref="y", axref="x", ayref="y",
        arrowhead=3, arrowsize=1.1, arrowwidth=1.4,
        arrowcolor="#333",
        showarrow=True,
    )
    ann = None
    if label:
        ann = dict(
            x=(x0 + x1) / 2, y=(y0 + y1) / 2 + label_dy,
            text=label,
            showarrow=False,
            font=dict(size=9.5, color="#444", family="Consolas, monospace"),
            align="center",
            bgcolor="rgba(255,255,255,0.85)",
        )
    return arr, ann


def build_figure():
    shapes = []
    annotations = []
    arrows = []

    # Цвета по блокам
    C_ENC = "#cfe6f5"   # encoder: голубой
    C_DIFF = "#fde4b5"  # diffusion: жёлто-оранжевый
    C_DEC = "#d8ebd2"   # decoder: зелёный
    C_LOSS = "#eaeaea"  # loss: серый
    C_DATA = "#f4f4f4"  # data tensors: светло-серый

    # === Левый столбец: ввод и кодировщик
    s, a = make_box(0.0, 4.6, 2.5, 0.9, "Последовательность событий<br>(label, time)<br>(B, L)", C_DATA, font_size=10)
    shapes.append(s); annotations.append(a)

    s, a = make_box(0.0, 3.3, 2.5, 0.9, "Embedder<br>label_emb (22→32) ⊕ time<br>выход: (B, L, 33)", C_ENC, font_size=10)
    shapes.append(s); annotations.append(a)

    s, a = make_box(0.0, 2.0, 2.5, 0.9, "GRU<br>input_size = 33, hidden = 64<br>выход h_p ∈ (B, L, 64)", C_ENC, font_size=10)
    shapes.append(s); annotations.append(a)

    # === Центр: диффузия
    s, a = make_box(3.4, 5.5, 2.6, 0.85, "Гауссов шум x_T<br>(V, K=48, 33)", C_DATA, font_size=10)
    shapes.append(s); annotations.append(a)

    s, a = make_box(3.4, 3.6, 2.6, 1.5,
                    "Денойзер (биGRU)<br>условие: h_p<br>15 шагов реверс-процесса<br>выход ẑ ∈ (V, K, 33)",
                    C_DIFF, font_size=10)
    shapes.append(s); annotations.append(a)

    s, a = make_box(3.4, 2.0, 2.6, 0.9,
                    "Целевая последовательность<br>K+1 событий<br>(out-of-seq → mask_emb)",
                    C_DATA, font_size=10)
    shapes.append(s); annotations.append(a)

    # === Правый блок: декодер
    s, a = make_box(7.0, 5.5, 2.6, 0.85,
                    "Запросы слотов q_k<br>(K = 48, d_q = 64)",
                    C_DATA, font_size=10)
    shapes.append(s); annotations.append(a)

    s, a = make_box(7.0, 3.6, 2.6, 1.5,
                    "Per-slot декодер g_φ<br>[ẑ_k; q_k] ∈ ℝ^{33+64}<br>MLP [128, 256] + BN<br>выход P_base = 24",
                    C_DEC, font_size=10)
    shapes.append(s); annotations.append(a)

    s, a = make_box(7.0, 2.0, 2.6, 0.9,
                    "K кандидатов<br>(s_k, t̂_k, π̂_k)<br>(V, K, 24)",
                    C_DATA, font_size=10)
    shapes.append(s); annotations.append(a)

    # === Низ: матчинг и потери
    s, a = make_box(2.5, 0.4, 4.6, 0.9,
                    "Венгерское сопоставление K кандидатов с истинными событиями горизонта<br>стоимость: w_pres·BCE + w_t·MAE + w_y·CE + штраф past-horizon",
                    C_LOSS, font_size=9.5)
    shapes.append(s); annotations.append(a)

    s, a = make_box(7.4, 0.4, 2.2, 0.9,
                    "ℒ_dec (детекция)<br>+ ℒ_diff (MSE латентов)<br>= λ_dec·ℒ_dec + λ_diff·ℒ_diff",
                    C_LOSS, font_size=9.5)
    shapes.append(s); annotations.append(a)

    # === Стрелки потоков данных
    arrow_specs = [
        # Левый столбец (вертикальные)
        (1.25, 4.6, 1.25, 4.2, None),
        (1.25, 3.3, 1.25, 2.9, None),
        # h_p → денойзер (как условие)
        (2.5, 2.45, 3.4, 4.0, "условие h_p"),
        # h_p → нет в декодере (опорная архитектура): не рисуем
        # x_T → денойзер
        (4.7, 5.5, 4.7, 5.1, None),
        # денойзер → ẑ (выход уже в самом блоке; рисуем стрелку вниз к decoder)
        # Целевая последовательность → денойзер (для loss во время обучения)
        (4.7, 2.9, 4.7, 3.6, "ℒ_diff (MSE)"),
        # q_k → декодер
        (8.3, 5.5, 8.3, 5.1, None),
        # ẑ → декодер (горизонтально)
        (6.0, 4.35, 7.0, 4.35, "ẑ_k"),
        # декодер → кандидаты
        (8.3, 3.6, 8.3, 2.9, None),
        # кандидаты → matching
        (7.0, 2.0, 5.0, 1.3, None),
        # matching → loss
        (7.1, 0.85, 7.4, 0.85, None),
    ]
    for spec in arrow_specs:
        x0, y0, x1, y1, label = spec
        arr, ann = make_arrow(x0, y0, x1, y1, label=label)
        arrows.append(arr)
        if ann is not None:
            annotations.append(ann)

    fig = go.Figure()
    fig.update_layout(
        shapes=shapes,
        annotations=annotations + arrows,
        xaxis=dict(visible=False, range=[-0.2, 10.0], scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False, range=[0.0, 6.6]),
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=1200,
        height=720,
    )
    return fig


def main():
    out_dir = Path(__file__).resolve().parents[2] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    pdf_path = out_dir / "method_a6_base.pdf"
    png_path = out_dir / "method_a6_base.png"
    fig.write_image(str(pdf_path), format="pdf")
    fig.write_image(str(png_path), format="png", scale=2)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
