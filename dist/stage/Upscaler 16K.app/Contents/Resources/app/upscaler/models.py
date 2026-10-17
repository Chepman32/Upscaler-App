"""Загрузка и кеширование весов моделей. Интернет нужен ТОЛЬКО один раз — для скачивания весов
(их же скачивает setup.command). После этого вся обработка идёт полностью локально."""
from __future__ import annotations

import os
import ssl
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(os.environ.get("UPSCALER_MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))

MODELS = {
    "realesrgan": {
        "title": "Real-ESRGAN x4plus — фото (рекомендуется)",
        "files": {
            "gan": ("RealESRGAN_x4plus.pth",
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"),
            "net": ("RealESRNet_x4plus.pth",
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"),
        },
        "has_fidelity": True,
    },
    "swinir": {
        "title": "SwinIR-M real-world GAN — фото (медленнее)",
        "files": {
            "gan": ("SwinIR_realSR_x4_GAN.pth",
                    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
                    "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"),
        },
        "has_fidelity": False,
    },
}


def _ssl_context():
    if os.environ.get("SSL_CERT_FILE"):  # корпоративные прокси / нестандартные CA
        return ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"])
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def ensure_file(fname: str, url: str, log=print) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dst = MODELS_DIR / fname
    if dst.exists() and dst.stat().st_size > 1_000_000:
        return dst
    log(f"Скачиваю веса модели {fname} (один раз)…")
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "upscaler/1.0"})
    with urllib.request.urlopen(req, context=_ssl_context(), timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        last = -1
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = int(done * 100 / total)
                if pct // 10 != last // 10:
                    last = pct
                    log(f"  {pct}%")
    tmp.replace(dst)
    return dst


def ensure_model_files(key: str, log=print) -> dict:
    spec = MODELS[key]
    return {k: ensure_file(fn, url, log) for k, (fn, url) in spec["files"].items()}


def _unwrap(sd: dict) -> dict:
    for k in ("params_ema", "params"):
        if k in sd and isinstance(sd[k], dict):
            return sd[k]
    return sd


def load_model(key: str, fidelity: float, device, log=print):
    """Возвращает spandrel-дескриптор модели (callable: тензор [1,3,H,W] 0..1 -> [1,3,sH,sW])."""
    import torch
    from spandrel import ModelLoader

    files = ensure_model_files(key, log)
    gan = _unwrap(torch.load(files["gan"], map_location="cpu", weights_only=True))
    sd = gan
    if MODELS[key]["has_fidelity"] and "net" in files and fidelity < 0.999:
        net = _unwrap(torch.load(files["net"], map_location="cpu", weights_only=True))
        a = float(fidelity)
        # интерполяция весов сети (приём из Real-ESRGAN): 0 = ESRNet (верность, мягче), 1 = ESRGAN (детали)
        sd = {k: (a * v + (1 - a) * net[k]) if (v.is_floating_point() and k in net) else v for k, v in gan.items()}
    desc = ModelLoader().load_from_state_dict(sd)
    desc.to(device).eval()
    return desc


def pick_device(pref: str = "auto"):
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if pref == "mps" or (pref == "auto" and getattr(torch.backends, "mps", None)
                         and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    for k in MODELS:
        ensure_model_files(k, print)
    print("Готово: веса в", MODELS_DIR, file=sys.stderr)
