"""Проверка логики тайлинга/блендинга без нейросети: «модель» = nearest x4 (не зависит от контекста),
поэтому результат после смешивания тайлов должен совпасть с эталоном побитово."""
import sys, tempfile, time
from pathlib import Path
import numpy as np
from upscaler.engine import upscale_pass, resize_bands

def nearest(t): return np.repeat(np.repeat(t, 4, 0), 4, 1).astype(np.float32) / 255.0

rng = np.random.default_rng(1)
tmp = Path(tempfile.mkdtemp())
for (h, w, tile, ov) in [(100, 130, 64, 16), (257, 301, 96, 24), (50, 40, 128, 16), (333, 211, 64, 8), (640, 480, 100, 90)]:
    src = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    out = upscale_pass(src, nearest, 4, tile, ov, tmp / "o.dat", lambda a, b: None, lambda: False)
    ref = np.repeat(np.repeat(src, 4, 0), 4, 1)
    d = np.abs(out.astype(int) - ref.astype(int)).max()
    print(f"{w}x{h} tile={tile} ov={ov}: max diff={d}")
    assert d <= 1, "seam/coverage bug"
# ресайз полосами против PIL целиком
from PIL import Image
src = rng.integers(0, 256, (400, 700, 3), dtype=np.uint8)
dst = resize_bands(src, tmp / "r.dat", 500, 321)
ref = np.asarray(Image.fromarray(src).resize((500, 321), Image.LANCZOS))
print("resize_bands max diff:", np.abs(dst.astype(int) - ref.astype(int)).max())
print("OK")
