"""Сравнение на 100%-кропе: слева — оригинал, увеличенный обычным Lanczos до тех же размеров (то же поле зрения),
справа — результат нейросети. Оба фрагмента показаны 1:1 (по пикселям 16K-холста)."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .engine import load_image

Image.MAX_IMAGE_PIXELS = None


def _detail_center(gray: np.ndarray, win: int) -> tuple[int, int]:
    """Центр окна win×win с максимумом деталей (дисперсия лапласиана), с лёгким предпочтением центру кадра."""
    h, w = gray.shape
    lap = np.zeros_like(gray)
    lap[1:-1, 1:-1] = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1])
    win = max(8, min(win, h, w))
    gy, gx = max(1, h // win), max(1, w // win)
    best, bc = -1.0, (w // 2, h // 2)
    for iy in range(gy):
        for ix in range(gx):
            y0, x0 = iy * win, ix * win
            blk = lap[y0:y0 + win, x0:x0 + win]
            if blk.size == 0:
                continue
            cy, cx = (y0 + win / 2) / h - 0.5, (x0 + win / 2) / w - 0.5
            score = float(blk.var()) * float(np.exp(-(cx * cx + cy * cy) * 3.0))
            if score > best:
                best, bc = score, (int(x0 + win / 2), int(y0 + win / 2))
    return bc


def make_comparison(orig_path: str, result_path: str, out_path: str, crop: int = 800) -> str:
    orig, _, _, _ = load_image(orig_path)
    res = Image.open(result_path)
    W, H = res.size
    ow, oh = orig.size
    crop = min(crop, W, H)
    sx = W / ow
    # поиск «самого детального» места на оригинале
    g = np.asarray(orig.convert("L"), dtype=np.float32)
    win_src = max(8, int(round(crop / sx)))
    cx_o, cy_o = _detail_center(g, win_src)
    cx, cy = int(cx_o * sx), int(cy_o * (H / oh))
    x0 = int(np.clip(cx - crop // 2, 0, W - crop))
    y0 = int(np.clip(cy - crop // 2, 0, H - crop))

    right = res.crop((x0, y0, x0 + crop, y0 + crop)).convert("RGB")
    # то же поле зрения у оригинала, увеличенное Lanczos
    box = (x0 * ow / W, y0 * oh / H, (x0 + crop) * ow / W, (y0 + crop) * oh / H)
    left = orig.resize((crop, crop), Image.LANCZOS, box=box)

    pad, bar = 12, 40
    canvas = Image.new("RGB", (crop * 2 + pad * 3, crop + bar + pad * 2), (24, 24, 27))
    canvas.paste(left, (pad, bar + pad))
    canvas.paste(right, (crop + pad * 2, bar + pad))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:
        font = ImageFont.load_default()
    d.text((pad, 10), f"ORIGINAL + Lanczos (100%, {ow}x{oh} -> {W}x{H})", fill=(220, 220, 225), font=font)
    d.text((crop + pad * 2, 10), f"NEURAL NET RESULT (100%, crop {crop}x{crop} @ {x0},{y0})", fill=(220, 220, 225), font=font)
    canvas.save(out_path, "PNG")
    return out_path
