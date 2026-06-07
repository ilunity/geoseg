"""
pipeline.py — адаптивный ансамблевый пайплайн.

Логика авто-режима:
  1. Всегда: LULC (общая сегментация) + Terrain (классификация)
  2. Если water  > 3% в LULC → запускаем WaterModel    (IoU 0.81)
  3. Если forest > 3% в LULC → запускаем ForestModel   (IoU 0.82)
  4. Если urban  > 3% в LULC → запускаем BuildingModel (IoU 0.78)
  5. Дороги — только в ручном режиме (нет в LULC классах)
  6. Кастомные модели — только в ручном режиме (подключаются явно)

Ручной режим:
  Запускаются только модели из selected_models, без coverage-порога.
  Кастомные модели (ключи вида "custom_*") запускаются отдельно от
  основного combined_mask и возвращаются в custom_results.
"""
from __future__ import annotations
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
import time

# ── Единые ID классов в итоговой combined_mask ──────────────────────────────
CLASS_ID = {
    "background": 0,
    "water":      1,
    "forest":     2,
    "building":   3,
    "road":       4,
    "agriculture":5,
    "rangeland":  6,
    "barren":     7,
    "urban":      3,   # urban → building (один цвет)
}

CLASS_COLOR_BGR = {
    0: ( 20,  20,  20),   # background
    1: (200,  90,   0),   # water      — синий
    2: ( 34, 139,  34),   # forest     — зелёный
    3: ( 40,  40, 220),   # building   — красный
    4: ( 20, 160, 255),   # road       — жёлтый
    5: ( 60, 200,  60),   # agriculture— светлозелёный
    6: (100, 180,  80),   # rangeland
    7: ( 60, 130, 200),   # barren
}

CLASS_LABEL_RU = {
    0: "Фон",        1: "Вода",       2: "Лес",
    3: "Здания",     4: "Дороги",     5: "С/х угодья",
    6: "Кустарник",  7: "Пустошь",
}

COVERAGE_THRESHOLD = 0.03   # 3 %

MODEL_IOU = {
    "lulc":0.43,"water":0.81,"forest":0.82,
    "building":0.78,"road":0.74,"terrain":0.91,
}


@dataclass
class CustomModelResult:
    """Результат работы одной кастомной модели."""
    key:         str
    name:        str
    coverage:    dict[str, float]   # class_name → 0..1
    classes:     list[dict]         # [{id, name, name_ru, color}]
    model_type:  str                # "binary" | "multiclass"


@dataclass
class PipelineResult:
    combined_mask:   np.ndarray
    coverage:        dict[str, float]
    terrain:         dict | None
    models_used:     list[str]              = field(default_factory=list)
    log:             list[str]              = field(default_factory=list)
    ms:              float                  = 0.0
    custom_results:  list[dict]             = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────

class Pipeline:
    def __init__(self, models: dict):
        self.m = models

    # ─── Вспомогательные ─────────────────────────────────────────────────────

    def _is_custom(self, key: str) -> bool:
        return key.startswith("custom_")

    def _standard_models(self) -> dict:
        return {k: v for k, v in self.m.items() if not self._is_custom(k)}

    def _custom_models(self) -> dict:
        return {k: v for k, v in self.m.items() if self._is_custom(k)}

    # ─── run ─────────────────────────────────────────────────────────────────

    def run(
        self,
        image: Image.Image,
        mode: str = "auto",
        selected: list[str] | None = None,
    ) -> PipelineResult:

        t0   = time.perf_counter()
        log  = []
        used = []
        h, w = np.array(image).shape[:2]
        combined = np.zeros((h, w), dtype=np.uint8)
        custom_results: list[dict] = []

        def want(key: str) -> bool:
            if mode == "auto":
                return True
            return selected is not None and key in selected

        def apply_binary(mask: np.ndarray, class_id: int):
            combined[mask == 1] = class_id
            combined[(mask == 0) & (combined == class_id)] = 0

        # ── 1. LULC ──────────────────────────────────────────────────────────
        lulc_coverage: dict[str, float] = {}

        if "lulc" in self.m and want("lulc"):
            log.append("▶ lulc — общая сегментация...")
            raw_lulc     = self.m["lulc"].predict_mask(image)
            lulc_coverage = self.m["lulc"].coverage(image)
            used.append("lulc")

            from models.segmentation import LULC_IDX_TO_NAME
            for idx, name in LULC_IDX_TO_NAME.items():
                cid = CLASS_ID.get(name, 0)
                combined[raw_lulc == idx] = cid

            detected = [n for n, v in lulc_coverage.items()
                        if v > COVERAGE_THRESHOLD and n != "background"]
            log.append(f"  классы: {detected}")
        else:
            log.append("  — lulc пропущен")

        # ── 2. Специализированные стандартные модели ──────────────────────────
        spec = [
            ("building", "building", CLASS_ID["building"]),
            ("forest",   "forest",   CLASS_ID["forest"]),
            ("road",     "road",     CLASS_ID["road"]),
            ("water",    "water",    CLASS_ID["water"]),

        ]

        for lulc_name, model_key, class_id in spec:
            if model_key not in self.m:
                continue

            if mode == "auto":
                if model_key in ("road", "building"):
                    log.append(f"  — {model_key} пропущен в авто-режиме")
                    continue
                cov = lulc_coverage.get(lulc_name, 0)
                if lulc_name == "building":
                    cov = lulc_coverage.get("urban", 0)
                if cov <= COVERAGE_THRESHOLD:
                    log.append(f"  — {model_key}: {cov*100:.1f}% < порог, пропущен")
                    continue
            else:
                if not (selected and model_key in selected):
                    combined[combined == class_id] = 0
                    log.append(f"  — {model_key} отключён")
                    continue

            log.append(f"▶ {model_key} — сегментация...")
            spec_mask = self.m[model_key].predict_mask(image)
            apply_binary(spec_mask, class_id)
            used.append(model_key)
            cov_pct = float(spec_mask.mean()) * 100
            log.append(f"  ✓ {model_key}: {cov_pct:.1f}% · IoU {MODEL_IOU.get(model_key,'?')}")

        # ── 3. Terrain ───────────────────────────────────────────────────────
        terrain = None
        if "terrain" in self.m:
            log.append("▶ terrain — классификация рельефа...")
            terrain = self.m["terrain"].classify(image)
            used.append("terrain")
            log.append(f"  ✓ {terrain['icon']} {terrain['label_ru']} {terrain['confidence']*100:.0f}%")

        # ── 4. Кастомные модели ───────────────────────────────────────────────────
        for key, model in self._custom_models().items():
            if mode == "auto":
                log.append(f"  — {key} пропущен в авто-режиме (кастомный)")
                continue
            if not (selected and key in selected):
                log.append(f"  — {key} отключён")
                continue

            log.append(f"▶ {key} ({model.meta['name']}) — кастомная сегментация...")
            try:
                mask = model.predict_mask(image)
                cov  = model.coverage(mask)
                used.append(key)

                if model.num_classes == 1 and model.classes:
                    dyn_id = self._get_dynamic_class_id(key)
                    apply_binary(mask, dyn_id)
                    classes_with_dyn = [{**cls, "id": dyn_id} for cls in model.classes]
                    log.append(f"  ✓ {key}: {mask.mean()*100:.1f}% покрытие")
                else:
                    nonzero = mask > 0
                    combined[nonzero] = mask[nonzero]
                    classes_with_dyn = model.classes
                    log.append(f"  ✓ {key}: многоклассовая, {len(model.classes)} кл.")

                custom_results.append({
                    "key":        key,
                    "name":       model.meta["name"],
                    "coverage":   {k: round(v * 100, 2) for k, v in cov.items()},
                    "classes":    classes_with_dyn,   # ← id теперь совпадает с маской
                    "model_type": model.meta.get("model_type", "binary"),
                })
            except Exception as e:
                log.append(f"  ✗ {key}: ошибка — {e}")

        # ── 5. Итоговое покрытие ─────────────────────────────────────────────
        total = combined.size
        coverage = {
            name: float(np.sum(combined == cid)) / total
            for name, cid in CLASS_ID.items()
            if name not in ("urban",) and cid != 0
        }

        ms = (time.perf_counter() - t0) * 1000
        log.append(f"✓ Готово за {ms:.0f} мс")

        return PipelineResult(
            combined_mask  = combined,
            coverage       = coverage,
            terrain        = terrain,
            models_used    = used,
            log            = log,
            ms             = ms,
            custom_results = custom_results,
        )

    def _get_dynamic_class_id(self, key: str) -> int:
        """
        Возвращает динамический class_id для кастомной бинарной модели.
        Начинаем с 20, каждая следующая +1 (по алфавиту ключа).
        """
        custom_keys = sorted(k for k in self.m if k.startswith("custom_"))
        idx = custom_keys.index(key) if key in custom_keys else 0
        return 20 + idx
