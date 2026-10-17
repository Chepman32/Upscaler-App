#!/usr/bin/env python3
"""Консольная версия: python upscale_cli.py photo.jpg [--target 15360] ..."""
import argparse
import sys

from upscaler.engine import PRESETS, Settings, human_size, run_upscale


def main():
    ap = argparse.ArgumentParser(description="Локальный нейросетевой апскейл (Real-ESRGAN / SwinIR)")
    ap.add_argument("image", help="путь к изображению (любой формат, который читает Pillow)")
    ap.add_argument("--target", default="16K", help="16K | 8K | 4K | число пикселей длинной стороны")
    ap.add_argument("--model", default="realesrgan", choices=["realesrgan", "swinir"])
    ap.add_argument("--fidelity", type=float, default=0.8, help="0..1 (0 — верность, 1 — максимум деталей)")
    ap.add_argument("--tile", type=int, default=384)
    ap.add_argument("--overlap", type=int, default=48)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--format", default="png", choices=["png", "tiff"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("--no-color-fix", action="store_true", help="отключить привязку цвета к оригиналу")
    a = ap.parse_args()

    tgt = PRESETS.get(a.target.upper()) or int(a.target)
    st = Settings(target_long=tgt, tile=a.tile, overlap=a.overlap, device=a.device, model=a.model,
                  fidelity=a.fidelity, fmt=a.format, make_compare=not a.no_compare, out_dir=a.out_dir,
                  color_fix=not a.no_color_fix)
    last = [""]

    def prog(f, s):
        line = f"[{f * 100:5.1f}%] {s}"
        if line != last[0]:
            print("\r" + line.ljust(90), end="", flush=True)
            last[0] = line

    info = run_upscale(a.image, st, progress=prog, log=lambda m: print("\n" + m, flush=True))
    print(f"\n\nГотово: {info['name']}\n  разрешение: {info['size_px'][0]}×{info['size_px'][1]} px"
          f"\n  размер файла: {human_size(info['size_bytes'])}\n  время: {info['seconds']:.0f} с")
    if info.get("compare"):
        print("  сравнение 100%:", info["compare"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
