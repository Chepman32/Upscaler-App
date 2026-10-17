#!/usr/bin/env python3
"""Исправляет цвет уже готового результата апскейла по оригиналу (без повторного прогона нейросети).

    python fix_colors.py result_16k.png original.jpg [-o out.png] [--block 4]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from upscaler.colorfix import anchor_to_original
from upscaler.engine import human_size, load_image

Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result")
    ap.add_argument("original")
    ap.add_argument("-o", "--out")
    ap.add_argument("--block", type=int, default=4, help="размер сетки привязки в пикселях оригинала (по умолч. 4)")
    a = ap.parse_args()
    out = a.out or str(Path(a.result).with_name(Path(a.result).stem + "_colorfix.png"))
    orig, _, icc, dpi = load_image(a.original)
    res = Image.open(a.result)
    alpha = res.getchannel("A") if res.mode in ("RGBA", "LA") else None
    arr = np.array(res.convert("RGB"))
    del res
    before, after = anchor_to_original(arr, orig, block=a.block)
    print(f"Средний цветовой сдвиг (по сетке {a.block}px оригинала): {before:.2f} → {after:.2f}")
    im = Image.fromarray(arr)
    if alpha is not None:
        im.putalpha(alpha)
    kw = {"icc_profile": icc} if icc else {}
    im.save(out, "PNG", compress_level=6, **kw)
    print(f"Сохранено: {out}  {im.size[0]}×{im.size[1]} px, {human_size(Path(out).stat().st_size)}")


if __name__ == "__main__":
    sys.exit(main())
