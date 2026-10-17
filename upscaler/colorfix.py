"""Коррекция цвета: нейросети (особенно GAN, напр. Real-ESRGAN) слегка «уводят» цвет и яркость, а при нескольких
проходах ×4 сдвиг накапливается (на тёплых кадрах — красно-пурпурный оттенок).

Решение — обратная проекция на грубой сетке: крупномасштабный цвет/свет берётся из источника, мелкие детали —
от нейросети. Разница считается на сетке block×block исходных пикселей (по умолчанию 4), поэтому JPEG-артефакты
(блоки 8 px) и шум источника НЕ возвращаются обратно, а цвет и освещение совпадают с оригиналом.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def tile_color_fix(x, o, block: int = 4):
    """x: тензор входа [1,3,h,w] (0..1), o: выход модели [1,3,H,W]. Всё на CPU. Возвращает исправленный o."""
    import torch.nn.functional as F
    h, w = x.shape[-2:]
    hc, wc = max(1, round(h / block)), max(1, round(w / block))
    d = F.adaptive_avg_pool2d(x, (hc, wc)) - F.adaptive_avg_pool2d(o, (hc, wc))
    return o + F.interpolate(d, size=o.shape[-2:], mode="bicubic", align_corners=False)


def _area_down_bands(arr: np.ndarray, wc: int, hc: int, band: int = 16) -> np.ndarray:
    """Точное усреднение по площади (BOX) большого HxWx3 массива до hc×wc — полосами, без загрузки целиком."""
    H, W, _ = arr.shape
    ky = H / hc
    out = np.empty((hc, wc, 3), np.float32)
    for ja in range(0, hc, band):
        jb = min(ja + band, hc)
        y0, y1 = int(math.floor(ja * ky)), min(H, int(math.ceil(jb * ky)))
        chunk = np.ascontiguousarray(arr[y0:y1]).astype(np.float32)
        for c in range(3):
            im = Image.fromarray(chunk[..., c], "F")
            out[ja:jb, :, c] = np.asarray(im.resize((wc, jb - ja), Image.BOX, box=(0, ja * ky - y0, W, jb * ky - y0)))
    return out


def anchor_to_original(arr: np.ndarray, orig: Image.Image, block: int = 4, band: int = 512,
                       cancel: Callable[[], bool] = lambda: False, progress: Callable[[float], None] = lambda f: None):
    """Привязывает крупный цвет/свет итогового изображения (memmap HxWx3 uint8, изменяется на месте) к оригиналу.
    Возвращает (среднее |Δ| до, после) на грубой сетке — для лога."""
    H, W, _ = arr.shape
    ow, oh = orig.size
    wc, hc = max(1, round(ow / block)), max(1, round(oh / block))
    ref = np.asarray(orig.convert("RGB"), np.float32)
    ref_c = np.stack([np.asarray(Image.fromarray(ref[..., c], "F").resize((wc, hc), Image.BOX)) for c in range(3)], -1)
    res_c = _area_down_bands(arr, wc, hc)
    D = ref_c - res_c
    before = float(np.abs(D).mean())
    planes = [Image.fromarray(np.ascontiguousarray(D[..., c]), "F") for c in range(3)]
    ky = hc / H
    for a in range(0, H, band):
        if cancel():
            from .engine import Cancelled
            raise Cancelled()
        b = min(a + band, H)
        box = (0, a * ky, wc, b * ky)
        up = np.stack([np.asarray(p.resize((W, b - a), Image.BICUBIC, box=box)) for p in planes], -1)
        blk = arr[a:b].astype(np.float32) + up
        arr[a:b] = np.clip(blk + 0.5, 0, 255).astype(np.uint8)
        progress(b / H)
    if hasattr(arr, "flush"):
        arr.flush()
    after = float(np.abs(ref_c - _area_down_bands(arr, wc, hc)).mean())
    return before, after
