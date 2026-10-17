"""Движок: тайловый нейросетевой апскейл до целевого разрешения с плавным блендингом швов.

Память: результат пишется потоком на диск (np.memmap), в ОЗУ держится только полоса высотой в один ряд тайлов,
поэтому 16K (15360×8640) обрабатывается даже на машине с небольшим объёмом памяти.
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None
try:  # HEIC/HEIF, если установлен pillow-heif
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:  # noqa: BLE001
    pass

PRESETS = {"16K": 15360, "8K": 7680, "4K": 3840}


class Cancelled(Exception):
    pass


@dataclass
class Settings:
    target_long: int = 15360
    tile: int = 384
    overlap: int = 48
    device: str = "auto"          # auto | mps | cuda | cpu
    model: str = "realesrgan"     # realesrgan | swinir
    fidelity: float = 0.8         # 0 = верность оригиналу (мягче), 1 = максимум деталей (GAN)
    fmt: str = "png"              # png | tiff
    make_compare: bool = True
    color_fix: bool = True        # держать цвет/свет как в оригинале (против дрейфа цвета GAN-моделей)
    color_block: int = 4          # сетка привязки цвета, в пикселях источника
    out_dir: Optional[str] = None
    suffix: Optional[str] = None  # по умолчанию _16k / _8k / _<px>px


# ---------------------------------------------------------------- вспомогательное

def suffix_for(target_long: int) -> str:
    for name, px in PRESETS.items():
        if px == target_long:
            return "_" + name.lower()
    return f"_{target_long}px"


def plan_target(w: int, h: int, target_long: int) -> tuple[int, int]:
    """Длинная сторона = target_long, пропорции сохраняются."""
    if w >= h:
        return target_long, max(1, round(h * target_long / w))
    return max(1, round(w * target_long / h)), target_long


def pass_inputs(longest: int, target_long: int, model_scale: int = 4) -> list[int]:
    """Длинные стороны входов каждого прохода. Для 2-го и далее проходов вход (уже синтезированный результат)
    ужимается до target/scale — модель дорисовывает ровно до цели, без лишних пикселей и гигантских буферов."""
    ins = [longest]
    cur = longest
    while cur * model_scale < target_long:
        cur = min(cur * model_scale, math.ceil(target_long / model_scale))
        ins.append(cur)
    return ins


def plan_passes(w: int, h: int, target_long: int, model_scale: int = 4) -> int:
    return len(pass_inputs(max(w, h), target_long, model_scale))


def load_image(path: str):
    """-> (rgb PIL, alpha PIL|None, icc bytes|None, dpi|None)"""
    im = Image.open(path)
    try:
        im.seek(0)
    except Exception:
        pass
    im = ImageOps.exif_transpose(im)
    icc = im.info.get("icc_profile")
    dpi = im.info.get("dpi")
    if im.mode in ("I;16", "I;16L", "I;16B", "I", "F"):
        a = np.asarray(im).astype(np.float32)
        a = a / (65535.0 if a.max() > 255 else 255.0)
        a8 = np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8)
        im = Image.fromarray(a8).convert("RGB")
    alpha = None
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        alpha = im.getchannel("A")
    rgb = im.convert("RGB")
    return rgb, alpha, icc, dpi


def tile_starts(n: int, t: int, ov: int) -> tuple[list[int], int]:
    if n <= t:
        return [0], n
    step = max(1, t - ov)
    s = list(range(0, n - t + 1, step))
    if s[-1] + t < n:
        s.append(n - t)
    return s, t


def _ramp(o: int) -> np.ndarray:
    x = (np.arange(o, dtype=np.float32) + 0.5) / o
    r = x * x * (3 - 2 * x)  # smoothstep
    return np.maximum(r, 1e-3)


def axis_weights(starts: list[int], t: int, scale: int) -> list[np.ndarray]:
    ws = []
    for k, s in enumerate(starts):
        w = np.ones(t * scale, np.float32)
        if k > 0:
            o = (starts[k - 1] + t - s) * scale
            if o > 0:
                w[:o] = np.minimum(w[:o], _ramp(o))
        if k < len(starts) - 1:
            o = (s + t - starts[k + 1]) * scale
            if o > 0:
                w[-o:] = np.minimum(w[-o:], _ramp(o)[::-1])
        ws.append(w)
    return ws


def _is_oom(e: Exception) -> bool:
    m = str(e).lower()
    return any(s in m for s in ("out of memory", "invalid buffer size", "mps backend", "alloc", "not enough memory"))


# ---------------------------------------------------------------- один проход модели

def upscale_pass(src: np.ndarray, run_tile: Callable[[np.ndarray], np.ndarray], scale: int, tile: int,
                 overlap: int, out_path: Path, on_tile: Callable[[int, int], None],
                 cancel: Callable[[], bool]) -> np.memmap:
    """src: HxWx3 uint8 (ndarray/memmap). run_tile: uint8 tile -> float32 (0..1) tile*scale.
    Возвращает memmap (H*scale, W*scale, 3) uint8 с бесшовно смешанными тайлами."""
    H, W, _ = src.shape
    ys, th = tile_starts(H, tile, overlap)
    xs, tw = tile_starts(W, tile, overlap)
    wy = axis_weights(ys, th, scale)
    wx = axis_weights(xs, tw, scale)
    OH, OW = H * scale, W * scale
    out = np.memmap(out_path, dtype=np.uint8, mode="w+", shape=(OH, OW, 3))
    Hb = th * scale
    acc = np.zeros((Hb, OW, 3), np.float32)
    wacc = np.zeros((Hb, OW), np.float32)
    flushed = 0
    total = len(ys) * len(xs)
    done = 0

    def flush(n: int):
        nonlocal flushed, acc, wacc
        if n <= 0:
            return
        blk = acc[:n] / wacc[:n, :, None]
        out[flushed:flushed + n] = np.clip(blk * 255.0 + 0.5, 0, 255).astype(np.uint8)
        if n < Hb:
            acc[:Hb - n] = acc[n:].copy()
            wacc[:Hb - n] = wacc[n:].copy()
        acc[max(Hb - n, 0):] = 0
        wacc[max(Hb - n, 0):] = 0
        flushed += n

    for iy, y in enumerate(ys):
        flush(min(y * scale - flushed, Hb))
        for ix, x in enumerate(xs):
            if cancel():
                raise Cancelled()
            t = np.ascontiguousarray(src[y:y + th, x:x + tw])
            o = run_tile(t)
            w2 = wy[iy][:, None] * wx[ix][None, :]
            r0 = y * scale - flushed
            c0 = x * scale
            acc[r0:r0 + th * scale, c0:c0 + tw * scale] += o * w2[..., None]
            wacc[r0:r0 + th * scale, c0:c0 + tw * scale] += w2
            done += 1
            on_tile(done, total)
    flush(OH - flushed)
    out.flush()
    return out


# ---------------------------------------------------------------- финальный ресайз полосами

def resize_bands(src: np.ndarray, dst_path: Path, dst_w: int, dst_h: int, band: int = 256,
                 cancel: Callable[[], bool] = lambda: False) -> np.memmap:
    """Точный ресайз до цели LANCZOS-ом по горизонтальным полосам (без загрузки всей картинки в память)."""
    sh, sw, _ = src.shape
    dst = np.memmap(dst_path, dtype=np.uint8, mode="w+", shape=(dst_h, dst_w, 3))
    ky = sh / dst_h
    m = 6
    for a in range(0, dst_h, band):
        if cancel():
            raise Cancelled()
        b = min(a + band, dst_h)
        y0 = max(0, int(math.floor(a * ky)) - m)
        y1 = min(sh, int(math.ceil(b * ky)) + m)
        chunk = Image.fromarray(np.ascontiguousarray(src[y0:y1]))
        box = (0, a * ky - y0, sw, b * ky - y0)
        r = chunk.resize((dst_w, b - a), Image.LANCZOS, box=box)
        dst[a:b] = np.asarray(r)
    dst.flush()
    return dst


# ---------------------------------------------------------------- полный пайплайн

def make_tile_runner(desc, device, log, color_fix: bool = True, color_block: int = 4):
    import torch
    from .colorfix import tile_color_fix

    def run(t: np.ndarray) -> np.ndarray:
        xc = torch.from_numpy(np.array(t, copy=True)).permute(2, 0, 1).unsqueeze(0).float().div(255.0).contiguous()
        with torch.inference_mode():
            o = desc(xc.to(device)).float().cpu()
            if color_fix:  # обратная проекция на грубой сетке: цвет/свет — от входа, детали — от сети (на CPU)
                o = tile_color_fix(xc, o, color_block)
            o = o.squeeze(0).permute(1, 2, 0).clamp(0, 1).contiguous().numpy()
        return o
    return run


def run_upscale(path: str, st: Settings, progress: Callable[[float, str], None] = lambda f, s: None,
                log: Callable[[str], None] = print, cancel: Callable[[], bool] = lambda: False,
                runner_override: Optional[Callable] = None, model_scale: int = 4) -> dict:
    t_start = time.time()
    p = Path(path)
    out_dir = Path(st.out_dir) if st.out_dir else p.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".png" if st.fmt == "png" else ".tif"
    out_path = out_dir / f"{p.stem}{st.suffix or suffix_for(st.target_long)}{ext}"

    progress(0.0, "Читаю изображение…")
    rgb, alpha, icc, dpi = load_image(str(p))
    w, h = rgb.size
    tw_, th_ = plan_target(w, h, st.target_long)
    passes = plan_passes(w, h, st.target_long, model_scale)
    log(f"Исходник: {p.name}  {w}×{h}px → цель {tw_}×{th_}px, проходов модели: {max(passes, 1)}")
    if max(w, h) >= st.target_long:
        log("⚠ Исходник уже не меньше цели: прогоняю модель и аккуратно уменьшаю (supersampling).")
        passes = 1

    if runner_override is None:
        from .models import load_model, pick_device
        device = pick_device(st.device)
        log(f"Устройство: {device}. Загружаю модель {st.model}…")
        progress(0.0, "Загружаю модель…")
        desc = load_model(st.model, st.fidelity, device, log)
        model_scale = int(getattr(desc, "scale", 4))
        runner = make_tile_runner(desc, device, log, st.color_fix, st.color_block)
        if st.color_fix:
            log("Коррекция цвета включена: цвет и свет привязываются к оригиналу.")
    else:
        runner = runner_override

    tmp = Path(tempfile.mkdtemp(prefix="upscaler16k_"))
    try:
        cur = np.asarray(rgb)
        tile = st.tile
        ins = pass_inputs(max(w, h), st.target_long, model_scale)
        passes = len(ins)
        weights = [float(L) * L for L in ins]
        tot_w = float(sum(weights)) or 1.0
        t0 = time.time()
        done_w = 0.0
        for pass_idx in range(passes):
            ch, cw, _ = cur.shape
            if pass_idx > 0 and max(ch, cw) > ins[pass_idx]:
                k = ins[pass_idx] / max(ch, cw)
                progress(done_w / tot_w * 0.93, f"Подготовка входа прохода {pass_idx + 1} ({round(cw * k)}×{round(ch * k)})…")
                cur = resize_bands(cur, tmp / f"pre{pass_idx}.dat", max(1, round(cw * k)), max(1, round(ch * k)),
                                   cancel=cancel)
            pw = weights[pass_idx]
            while True:
                def on_tile(d, t_, pi=pass_idx, pw=pw, dw=done_w):
                    frac = (dw + pw * d / t_) / tot_w
                    el = time.time() - t0
                    eta = el * (1 - frac) / max(frac, 1e-6)
                    progress(min(frac * 0.93, 0.93),
                             f"Проход {pi + 1}/{passes}: тайл {d}/{t_} · осталось ≈ {_fmt_t(eta)}")
                try:
                    nxt = upscale_pass(cur, runner, model_scale, tile, st.overlap,
                                       tmp / f"pass{pass_idx}.dat", on_tile, cancel)
                    break
                except Cancelled:
                    raise
                except Exception as e:  # noqa: BLE001
                    if _is_oom(e) and tile > 128:
                        tile = max(128, (tile // 2) // 8 * 8)
                        log(f"⚠ Не хватило памяти устройства — уменьшаю тайл до {tile}px и повторяю проход.")
                    else:
                        raise
            done_w += pw
            cur = nxt

        H2, W2, _ = cur.shape
        if (W2, H2) != (tw_, th_):
            progress(0.94, f"Точная подгонка {W2}×{H2} → {tw_}×{th_} (Lanczos)…")
            cur = resize_bands(cur, tmp / "final.dat", tw_, th_, cancel=cancel)

        if st.color_fix:
            from .colorfix import anchor_to_original
            progress(0.945, "Привязка цвета к оригиналу…")
            b_, a_ = anchor_to_original(cur, rgb, block=st.color_block, cancel=cancel)
            log(f"Цвет: средний сдвиг относительно оригинала {b_:.2f} → {a_:.2f} (из 255)")

        progress(0.96, "Сохраняю файл без потерь…")
        res = Image.fromarray(np.asarray(cur))
        if alpha is not None:
            res.putalpha(alpha.resize((tw_, th_), Image.LANCZOS))
        kw = {}
        if icc:
            kw["icc_profile"] = icc
        if dpi:
            kw["dpi"] = tuple(float(x) for x in dpi) if isinstance(dpi, tuple) else (dpi, dpi)
        if st.fmt == "png":
            res.save(out_path, "PNG", compress_level=6, **kw)
        else:
            res.save(out_path, "TIFF", compression="tiff_adobe_deflate", **kw)
        del res, cur, nxt
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    info = {
        "path": str(out_path), "name": out_path.name, "size_px": (tw_, th_),
        "size_bytes": out_path.stat().st_size, "seconds": time.time() - t_start,
        "tile_used": tile, "passes": passes,
    }
    if st.make_compare:
        progress(0.98, "Готовлю сравнение 100%…")
        from .compare import make_comparison
        # сравнение — во временную папку, чтобы не засорять папку результата
        cmp_dir = Path(tempfile.gettempdir()) / "Upscaler16K"
        cmp_dir.mkdir(parents=True, exist_ok=True)
        cmp_path = cmp_dir / f"{p.stem}{st.suffix or suffix_for(st.target_long)}_compare.png"
        make_comparison(str(p), str(out_path), str(cmp_path))
        info["compare"] = str(cmp_path)
    progress(1.0, "Готово")
    return info


def _fmt_t(s: float) -> str:
    s = int(max(s, 0))
    if s >= 3600:
        return f"{s // 3600} ч {s % 3600 // 60} мин"
    if s >= 60:
        return f"{s // 60} мин {s % 60} с"
    return f"{s} с"


def human_size(n: float) -> str:
    for u in ("Б", "КБ", "МБ", "ГБ"):
        if n < 1024 or u == "ГБ":
            return f"{n:.1f} {u}" if u != "Б" else f"{int(n)} Б"
        n /= 1024
    return str(n)
