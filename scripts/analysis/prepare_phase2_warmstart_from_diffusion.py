"""Подготовка warmstart-чекпоинта для фазы 2 двухфазного разогрева encoder'а.

Берёт чекпоинт фазы 1 (a6_query_only_decoder_v16_phase1_diffusion_only),
извлекает веса encoder'а (`_seq_encoder.*`), накладывает их поверх свежего
state_dict архитектуры b2_detpp_baseline_v1 (chistый DeTPP).

На выходе --- чекпоинт, с которого можно стартовать b2-конфиг через
`init_from_checkpoint`. Все DeTPP-специфичные веса (head, matching_priors)
остаются со случайной инициализацией; обновляется только encoder.
"""
import sys
from pathlib import Path
import torch
import hydra
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "stackoverflow"
CKPT_DIR = EXP_DIR / "canonical" / "checkpoints"

PHASE1_CKPT = CKPT_DIR / "a6_query_only_decoder_v16_phase1_diffusion_only.ckpt"
OUT_CKPT = CKPT_DIR / "b2_detpp_warmstart_from_diffusion_init.ckpt"


def build_fresh_b2_state_dict():
    # Указываем config_dir прямо в canonical/, чтобы избежать заворачивания
    # композированного конфига под ключ `canonical:` (которое ломает
    # интерполяции вида ${max_time_delta}).
    cfg_dir = EXP_DIR / "configs" / "canonical"
    with hydra.initialize_config_dir(config_dir=str(cfg_dir), version_base="1.2"):
        cfg = hydra.compose(config_name="b2_detpp_baseline_v1")
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    model = hydra.utils.instantiate(cfg.module)
    return model.state_dict()


def main():
    print(f"Загружаю чекпоинт фазы 1: {PHASE1_CKPT}")
    if not PHASE1_CKPT.exists():
        raise FileNotFoundError(
            f"Нет чекпоинта фазы 1 по пути {PHASE1_CKPT}. "
            f"Сначала запусти a6_query_only_decoder_v16_phase1_diffusion_only."
        )
    phase1_sd = torch.load(PHASE1_CKPT, map_location="cpu")
    if isinstance(phase1_sd, dict) and "state_dict" in phase1_sd:
        phase1_sd = phase1_sd["state_dict"]
    print(f"  всего ключей: {len(phase1_sd)}")

    encoder_keys = {k: v for k, v in phase1_sd.items() if k.startswith("_seq_encoder.")}
    print(f"  ключей encoder'а (_seq_encoder.*): {len(encoder_keys)}")

    print("Создаю свежий state_dict b2_detpp_baseline_v1...")
    b2_sd = build_fresh_b2_state_dict()
    print(f"  всего ключей: {len(b2_sd)}")

    merged = dict(b2_sd)
    matched = 0
    shape_mismatched = []
    not_in_b2 = []
    for k, v in encoder_keys.items():
        if k not in b2_sd:
            not_in_b2.append(k)
            continue
        if v.shape != b2_sd[k].shape:
            shape_mismatched.append((k, tuple(v.shape), tuple(b2_sd[k].shape)))
            continue
        merged[k] = v
        matched += 1

    print(f"\nРезультат:")
    print(f"  скопировано: {matched}")
    print(f"  не нашлось в b2: {len(not_in_b2)} -> {not_in_b2[:5]}")
    print(f"  размерность не совпала: {len(shape_mismatched)} -> {shape_mismatched[:5]}")
    print(f"  свежей инициализации в b2 (head + DetectionLoss): {len(b2_sd) - matched}")

    print(f"\nСохраняю warmstart-чекпоинт: {OUT_CKPT}")
    torch.save(merged, OUT_CKPT)
    print("Готово.")


if __name__ == "__main__":
    main()
