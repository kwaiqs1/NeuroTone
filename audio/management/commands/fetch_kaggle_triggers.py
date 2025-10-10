# audio/management/commands/fetch_kaggle_triggers.py
import os, sys, re, shutil, subprocess, tempfile
from pathlib import Path
from typing import List
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
from django.core.management.base import BaseCommand

EAT_DS   = "mashijie/eating-sound-collection"
NOISE_DS = "moazabdeljalil/back-ground-noise"

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _has_kaggle_cli() -> bool:
    try:
        subprocess.run(["kaggle", "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except Exception:
        return False

def _kaggle_download(dataset: str, out_dir: Path, only_wav: bool | None = True):
    _ensure_dir(out_dir)
    if only_wav:
        print(f"[kaggle] listing files for {dataset} ...")
        lst = subprocess.run(
            ["kaggle", "datasets", "files", "-d", dataset],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True
        ).stdout.splitlines()
        wavs = []
        for line in lst:
            cols = re.split(r"\s{2,}", line.strip())
            if cols and cols[0].lower().endswith(".wav"):
                wavs.append(cols[0])
        if not wavs:
            print("[kaggle] no explicit .wav listing found, downloading whole dataset...")
            subprocess.run(["kaggle", "datasets", "download", "-d", dataset, "-p", str(out_dir), "--unzip", "--force"], check=True)
            return

        for i, fn in enumerate(wavs, 1):
            try:
                subprocess.run(
                    ["kaggle", "datasets", "download", "-d", dataset, "-f", fn, "-p", str(out_dir), "--force"],
                    check=True
                )
            except subprocess.CalledProcessError:
                continue
        for z in out_dir.glob("*.zip"):
            subprocess.run(["python", "-m", "zipfile", "-e", str(z), str(out_dir)], check=True)
            z.unlink(missing_ok=True)
    else:
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset, "-p", str(out_dir), "--unzip", "--force"], check=True)

def _iter_wavs(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() == ".wav"]

def _read_mono_float(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), always_2d=False, dtype="float32")
    if y.ndim == 2:
        y = y.mean(axis=1)
    return y.astype(np.float32), int(sr)

def _resample(y: np.ndarray, sr: int, target: int = 16000) -> np.ndarray:
    if sr == target:
        return y.astype(np.float32)
    g = np.gcd(int(sr), int(target))
    return resample_poly(y, target // g, sr // g).astype(np.float32)

def _save_wav(path: Path, y: np.ndarray, sr: int = 16000):
    sf.write(str(path), y, sr, subtype="PCM_16")

class Command(BaseCommand):
    help = "Fetch Kaggle audio and prepare data/triggers/{chewing,other}"

    def add_arguments(self, parser):
        parser.add_argument("--max_pos", type=int, default=1500, help="Max chewing files")
        parser.add_argument("--max_neg", type=int, default=2000, help="Max other files")
        parser.add_argument("--kaggle_dir", type=str, default="data/kaggle_cache")
        parser.add_argument("--out_dir", type=str, default="data/triggers")
        parser.add_argument("--target_sr", type=int, default=16000)

    def handle(self, *args, **opts):
        if not _has_kaggle_cli():
            self.stderr.write(self.style.ERROR(
                "Не найден kaggle CLI. В Colab/локально сделай: pip install kaggle "
                "и добавь kaggle.json в ~/.kaggle/"
            ))
            return

        cache = Path(opts["kaggle_dir"])
        trg   = Path(opts["out_dir"])
        sr_t  = int(opts["target_sr"])
        _ensure_dir(cache); _ensure_dir(trg)

        pos_dir = trg / "chewing"
        neg_dir = trg / "other"
        shutil.rmtree(pos_dir, ignore_errors=True)
        shutil.rmtree(neg_dir, ignore_errors=True)
        _ensure_dir(pos_dir); _ensure_dir(neg_dir)


        tmp_pos = cache / "chewing_raw"
        if tmp_pos.exists():
            print("[kaggle] reuse cache:", tmp_pos)
        else:
            _kaggle_download(EAT_DS, tmp_pos, only_wav=True)

        pos_files = _iter_wavs(tmp_pos)[: int(opts["max_pos"])]
        self.stdout.write(self.style.SUCCESS(f"chewing files: {len(pos_files)}"))
        c = 0
        for p in pos_files:
            try:
                y, sr = _read_mono_float(p)
                y = _resample(y, sr, sr_t)
                need = int(1.2 * sr_t)
                if y.size < need:
                    y = np.pad(y, (0, need - y.size))
                else:
                    i0 = (y.size - need) // 2
                    y = y[i0:i0+need]
                _save_wav(pos_dir / f"chew_{c:06d}.wav", y, sr_t); c += 1
            except Exception:
                continue


        tmp_neg = cache / "bg_raw"
        if tmp_neg.exists():
            print("[kaggle] reuse cache:", tmp_neg)
        else:
            _kaggle_download(NOISE_DS, tmp_neg, only_wav=True)

        neg_files = _iter_wavs(tmp_neg)[: int(opts["max_neg"])]
        self.stdout.write(self.style.SUCCESS(f"other files: {len(neg_files)}"))
        c = 0
        for p in neg_files:
            try:
                y, sr = _read_mono_float(p)
                y = _resample(y, sr, sr_t)
                need = int(1.2 * sr_t)
                if y.size < need:
                    y = np.pad(y, (0, need - y.size))
                else:
                    i0 = (y.size - need) // 2
                    y = y[i0:i0+need]
                _save_wav(neg_dir / f"other_{c:06d}.wav", y, sr_t); c += 1
            except Exception:
                continue

        self.stdout.write(self.style.SUCCESS(f"Prepared at {trg}"))
        self.stdout.write("Классы: ['chewing', 'other']")
