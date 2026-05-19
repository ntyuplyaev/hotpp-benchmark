"""Гипотеза 2: диффузия в пространстве предсказаний P_base.

Отличие от a6_v3 (HorizonDiffusionLossDetppQueryOnlyDecoder):
- Цель диффузии живёт не в пространстве event-embedding (D = 33), а
  напрямую в пространстве выхода inner DetectionLoss (P_base = 24).
- Каналы P_base = [presence (1)] + [label_logits (num_classes)] + [time (1)].
  Целевые значения: presence ∈ {0, 1}, label = one-hot, time = time_delta
  от текущего события до k-го в окне горизонта.
- Денойзер реконструирует тензор в P_base-пространстве размерности 24
  (не в event-embedding 33), на тех же conditions h_p (size 64).
- Decoder становится тонким: per-slot Linear([z_k; q_k]) → P_base, с
  residual-инициализацией (output ≈ z_k в начале обучения). q_k оставлены
  для сохранения механизма slot-identity и небольших per-slot поправок.
- z_k теперь *уже является* предсказанием (presence_logit, label_logits,
  time_pred). Decoder при необходимости может его уточнить, но не должен
  делать ничего радикального.

Цель: согласовать пространство, в котором живут латенты диффузии, с
пространством, в котором их ждёт inner DetectionLoss. Сейчас decoder
вынужден переводить event-embedding (33) → predictions (24); связь
слабая. После правки --- одно и то же пространство.
"""
import torch

from hotpp.data import PaddedBatch
from hotpp.nn import GRUDenoiser
from ..fields import PRESENCE
from .detection import DetectionLoss
from .horizon_diffusion_v42 import HorizonDiffusionLossV42


class _PBaseRefinementHead(torch.nn.Module):
    """Тонкий per-slot decoder в пространстве P_base.

    Вход: PaddedBatch с payload (V, K, P_base) --- выход денойзера.
    Forward: per-slot concat(z_k, q_k) → Linear → P_base.
    Инициализация: weight = [I_{P_base} | 0_{query_size}], bias = 0,
    так что в начале обучения forward(z, q) = z. Дальше decoder
    может научиться маленькой per-slot поправке через q_k.
    """

    def __init__(self, p_base, k, query_size=64, query_init_scale=0.02):
        super().__init__()
        self.k = k
        self.p_base = p_base
        self.queries = torch.nn.Parameter(
            torch.randn(k, query_size) * float(query_init_scale)
        )
        self.linear = torch.nn.Linear(p_base + query_size, p_base)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.weight[:, :p_base] = torch.eye(p_base)
            self.linear.bias.zero_()

    def forward(self, x):
        z = x.payload  # (V, K, P_base)
        v = z.shape[0]
        if z.shape[1] != self.k:
            raise ValueError(f"Expected K={self.k}, got {z.shape[1]}")
        q = self.queries.unsqueeze(0).expand(v, -1, -1)  # (V, K, query_size)
        combined = torch.cat([z, q], dim=-1)             # (V, K, P_base + query_size)
        out = self.linear(combined)
        return PaddedBatch(out, x.seq_lens)


class HorizonDiffusionLossDetppPredictionSpace(HorizonDiffusionLossV42):
    """V42-supervision диффузии в пространстве предсказаний P_base."""

    def __init__(
        self,
        *args,
        # принимаются для совместимости с конфигом a6_v3, не используются
        query_size=64,
        query_hidden_dims=(128, 256),  # noqa: ARG002, не используется
        query_use_batch_norm=True,     # noqa: ARG002, не используется
        query_init_scale=0.02,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if not isinstance(self._next_item, DetectionLoss):
            raise ValueError(
                "HorizonDiffusionLossDetppPredictionSpace ожидает DetectionLoss "
                "в качестве inner next_item_loss."
            )

        # Размер пространства предсказания.
        inner_next_item = self._next_item._next_item
        p_base = inner_next_item.input_size
        self._pbase_size = p_base

        # Извлекаем порядок и размеры компонент. NextItemLoss._order = sorted(losses),
        # для DeTPP это ['_presence', 'labels', 'timestamps'].
        self._pbase_field_order = list(inner_next_item._order)
        self._pbase_field_sizes = [
            inner_next_item._losses[name].input_size for name in self._pbase_field_order
        ]
        # Кеш индексов каналов.
        offsets = {}
        offset = 0
        for name, size in zip(self._pbase_field_order, self._pbase_field_sizes):
            offsets[name] = (offset, offset + size)
            offset += size
        assert offset == p_base
        self._pbase_offsets = offsets
        self._num_label_classes = inner_next_item._losses["labels"].input_size

        # Перестраиваем denoiser в P_base-пространстве с теми же
        # hidden_size / num_layers / bidirectional, что и у исходного.
        old = self._denoiser
        self._denoiser = GRUDenoiser(
            input_size=p_base,
            length=self._k,
            hidden_size=old._hidden_size,
            generation_steps=self._generation_steps,
            num_layers=old._model.num_layers,
            bidirectional=old._bidirectional,
        )

        # mask_embedding теперь в P_base-пространстве.
        self._mask_embedding = torch.nn.Parameter(torch.zeros(p_base))

        # Тонкий per-slot decoder с residual-инициализацией.
        self._decoder = _PBaseRefinementHead(
            p_base=p_base,
            k=self._k,
            query_size=query_size,
            query_init_scale=query_init_scale,
        )

    def _noise(self, batch_size, device, dtype):
        """Шум в пространстве P_base, не в event-embedding."""
        if (self._prediction == "sample") or self.training:
            x = torch.randn(
                batch_size, self._k, self._pbase_size,
                device=device, dtype=dtype,
            )
        elif self._prediction == "mode":
            x = torch.zeros(
                batch_size, self._k, self._pbase_size,
                device=device, dtype=dtype,
            )
        else:
            raise ValueError(f"Unknown prediction mode: {self._prediction}")
        return PaddedBatch(
            x,
            torch.full([batch_size], self._k, device=device, dtype=torch.long),
        )

    def _embed_targets(self, targets):
        """Построить целевой тензор диффузии в пространстве P_base.

        Цель: для каждого слота k задать (presence_real, label_one_hot, time_delta)
        в порядке [_presence, labels, timestamps], как ожидает NextItemLoss.

        Padded-слоты (presence=0): целевое значение замещается обучаемым
        mask_embedding, так же, как было в V42 (только теперь в P_base).
        """
        timestamps = targets.payload[self._timestamps_field]   # (V, K+1)
        labels_int = targets.payload["labels"].long()          # (V, K+1)
        presence = targets.payload[PRESENCE]                   # (V, K+1)

        # time_delta для слотов 1..K от слота 0 (anchor).
        anchor_time = timestamps[:, 0:1]
        time_delta = (timestamps[:, 1:] - anchor_time).clamp(
            min=0, max=self._max_time_delta
        )  # (V, K)

        labels_oh = torch.nn.functional.one_hot(
            labels_int[:, 1:], num_classes=self._num_label_classes
        ).to(timestamps.dtype)  # (V, K, num_classes)

        presence_real = presence[:, 1:].to(timestamps.dtype).unsqueeze(-1)  # (V, K, 1)
        time_real = time_delta.to(timestamps.dtype).unsqueeze(-1)           # (V, K, 1)

        # Сборка в порядке self._pbase_field_order = ['_presence', 'labels', 'timestamps'].
        target_pbase = torch.cat([presence_real, labels_oh, time_real], dim=-1)
        # (V, K, P_base)

        # Замещение mask_embedding для padded слотов.
        is_padding = presence[:, 1:] == 0  # (V, K)
        target_pbase = torch.where(
            is_padding.unsqueeze(-1),
            self._mask_embedding.view(1, 1, -1),
            target_pbase,
        )

        seq_lens = (targets.seq_lens - 1).clip(min=0)
        return PaddedBatch(target_pbase, seq_lens)
