# audio/management/commands/train_triggers.py
import os, json, random
import subprocess, shutil
from typing import List, Tuple
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import tensorflow as tf
from tensorflow import keras
from django.core.management.base import BaseCommand

# ---- Константы ----
DATA_DIR   = "data/triggers"
MODEL_DIR  = "models"
N_MELS     = 64
TARGET_SR  = 16000
HOP        = 160           # 10 мс при 16 кГц
N_FFT      = 1024
FMIN, FMAX = 40.0, 8000.0
WIN_SEC    = 1.2           # длина окна для обучения/валидации
FIX_T      = 120           # целевое число временных кадров (≈1.2 c / 10 мс)

def _scan_files() -> Tuple[List[Tuple[str,int]], List[str]]:
    classes = []
    files: List[Tuple[str,int]] = []
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError(f"Put your data under {DATA_DIR}/<class>/*.mp3|*.wav")
    for name in sorted(os.listdir(DATA_DIR)):
        p = os.path.join(DATA_DIR, name)
        if not os.path.isdir(p): continue
        wavs = []
        for fn in os.listdir(p):
            if fn.lower().endswith((".wav",".mp3",".flac",".ogg",".m4a",".aac")):
                wavs.append(os.path.join(p, fn))
        if wavs:
            cid = len(classes)
            classes.append(name)
            for f in wavs:
                files.append((f, cid))
    if len(classes) < 2:
        raise RuntimeError("Need at least 2 classes to train.")
    random.shuffle(files)
    return files, classes

def _read_mono_32f(path: str) -> Tuple[np.ndarray, int]:
    try:
        y, sr = sf.read(path, always_2d=False, dtype="float32")
        if y.ndim == 2:
            y = y.mean(axis=1)
        return y.astype(np.float32), int(sr)
    except Exception:
        # Надёжный путь на любые форматы/битые теги
        y, sr = _ffmpeg_read_mono_32f(path, TARGET_SR)
        return y, sr


def _resample_if_needed(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return y.astype(np.float32)
    # устойчивый и быстрый ресемплинг без numba
    g = np.gcd(int(sr), int(target_sr))
    up, down = target_sr // g, sr // g
    return resample_poly(y, up, down).astype(np.float32)


def _power_to_db(S: np.ndarray, ref: float | np.ndarray = 1.0, amin: float = 1e-10, top_db: float = 80.0) -> np.ndarray:
    S = np.maximum(S, amin)
    log_spec = 10.0 * np.log10(S)
    if np.isscalar(ref):
        # ВАЖНО: защититься от ref=0
        ref = max(float(ref), amin)
        log_spec -= 10.0 * np.log10(ref)
    else:
        log_spec -= 10.0 * np.log10(np.maximum(ref, amin))
    if top_db is not None:
        log_spec = np.maximum(log_spec, log_spec.max() - float(top_db))
    return log_spec


def _mel_from_wav(y16: np.ndarray) -> np.ndarray:
    # центрирование как в librosa(center=True): паддинг половины окна
    pad = N_FFT // 2
    ypad = np.pad(y16, (pad, pad), mode="reflect").astype(np.float32)

    frames = tf.signal.stft(
        tf.convert_to_tensor(ypad, dtype=tf.float32),
        frame_length=N_FFT, frame_step=HOP,
        window_fn=tf.signal.hann_window, pad_end=False
    )  # [T, F]
    power = tf.math.square(tf.abs(frames))  # [T, F]
    mel_w = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=N_FFT // 2 + 1,
        sample_rate=TARGET_SR,
        lower_edge_hertz=FMIN,
        upper_edge_hertz=FMAX
    )  # [F, M]
    mel = tf.matmul(power, mel_w)  # [T, M]
    mel = mel.numpy().astype(np.float32).T  # -> [M, T]

    S_db = _power_to_db(mel, ref=np.max(mel))
    mu, std = S_db.mean(), S_db.std() + 1e-6
    S_n = (S_db - mu) / std

    # по времени до FIX_T
    if S_n.shape[1] < FIX_T:
        pad_t = FIX_T - S_n.shape[1]
        S_n = np.pad(S_n, ((0,0),(0,pad_t)), mode="constant")
    elif S_n.shape[1] > FIX_T:
        i0 = (S_n.shape[1] - FIX_T)//2
        S_n = S_n[:, i0:i0+FIX_T]
    return S_n.astype(np.float32)  # [M,T]

def _load_clip(path: str, deterministic: bool) -> np.ndarray:
    y, sr = _read_mono_32f(path)
    y = _resample_if_needed(y, sr, TARGET_SR)
    need = int(WIN_SEC * TARGET_SR)
    if y.shape[0] < need:
        pad = need - y.shape[0]
        y = np.pad(y, (0, pad))
    else:
        i0 = (y.shape[0] - need)//2 if deterministic else random.randint(0, y.shape[0] - need)
        y = y[i0:i0+need]
    return y

def build_model(n_mels=N_MELS, t_frames=FIX_T, n_classes=2) -> keras.Model:
    inp = keras.Input(shape=(n_mels, t_frames, 1))
    x = keras.layers.Conv2D(32, 3, padding="same")(inp)
    x = keras.layers.BatchNormalization()(x); x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPool2D(pool_size=(2,2))(x)
    x = keras.layers.Conv2D(64, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x); x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPool2D(pool_size=(2,2))(x)
    x = keras.layers.Conv2D(64, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x); x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    out = keras.layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

class Command(BaseCommand):
    help = "Train tiny local trigger classifier (TensorFlow, no librosa) from data/triggers/<class>/*.mp3|*.wav"

    def add_arguments(self, parser):
        parser.add_argument("--epochs", type=int, default=18)
        parser.add_argument("--batch",  type=int, default=32)
        parser.add_argument("--val_split", type=float, default=0.15)
        parser.add_argument("--aug_per_file", type=int, default=2,
                            help="сколько случайных окон генерировать с одного файла (train)")
        parser.add_argument("--resume", action="store_true",
                            help="Продолжить обучение с уже сохранённой модели, если найдена")


    def handle(self, *args, **opts):
        os.makedirs(MODEL_DIR, exist_ok=True)
        files, classes = _scan_files()
        n = len(files)
        n_val = max(1, int(n * float(opts["val_split"])))
        val = files[:n_val]; tr = files[n_val:]
        self.stdout.write(self.style.SUCCESS(
            f"Classes: {classes} | train={len(tr)} val={len(val)}"
        ))

        # --- готовим данные ---
        X_tr, y_tr = [], []
        for path, cid in tr:
            for _ in range(int(opts["aug_per_file"])):
                y = _load_clip(path, deterministic=False)
                S = _mel_from_wav(y)
                X_tr.append(S[..., None]); y_tr.append(cid)

        X_val, y_val = [], []
        for path, cid in val:
            y = _load_clip(path, deterministic=True)
            S = _mel_from_wav(y)
            X_val.append(S[..., None]); y_val.append(cid)

        X_tr = np.stack(X_tr, axis=0).astype(np.float32)
        y_tr = np.array(y_tr, dtype=np.int64)
        X_val = np.stack(X_val, axis=0).astype(np.float32)
        y_val = np.array(y_val, dtype=np.int64)



        model = None
        if opts.get("resume"):
            for cand in [os.path.join(MODEL_DIR, "trigger_cls_keras.h5"),
                         os.path.join(MODEL_DIR, "trigger_cls_keras"),
                         os.path.join(MODEL_DIR, "trigger_cls_keras.keras")]:
                if os.path.exists(cand):
                    try:
                        model = keras.models.load_model(cand)
                        break
                    except Exception:
                        pass
        if model is None:
            model = build_model(n_mels=N_MELS, t_frames=FIX_T, n_classes=len(classes))
        else:
            # на случай изменения числа классов — простой safety-чек
            if model.output_shape[-1] != len(classes):
                model = build_model(n_mels=N_MELS, t_frames=FIX_T, n_classes=len(classes))




        model = build_model(n_mels=N_MELS, t_frames=FIX_T, n_classes=len(classes))
        cb = [
            keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
            keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1),
        ]
        hist = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=int(opts["epochs"]),
            batch_size=int(opts["batch"]),
            verbose=2,
            callbacks=cb,
            shuffle=True,
        )
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        self.stdout.write(f"Final val_acc={val_acc:.3f}")

        # сохраняем
        model_path_h5 = os.path.join(MODEL_DIR, "trigger_cls_keras.h5")
        model_dir_tf  = os.path.join(MODEL_DIR, "trigger_cls_keras")  # каталог SavedModel

        saved_to = None
        try:
            # если h5py установлен — сохраняем в .h5
            import h5py  # noqa: F401
            model.save(model_path_h5)
            saved_to = model_path_h5
        except Exception as e:
            # иначе сохраняем как SavedModel (каталог), h5py не нужен
            self.stdout.write(self.style.WARNING(
                f"Не получилось сохранить в .h5 ({e}). Сохраняю как SavedModel каталог: {model_dir_tf}"
            ))
            try:
                if os.path.isdir(model_dir_tf):
                    shutil.rmtree(model_dir_tf)
            except Exception:
                pass
            # Для tf.keras (TF 2.x) сохранение в путь-без-расширения -> SavedModel
            model.save(model_dir_tf)
            saved_to = model_dir_tf

        with open(os.path.join(MODEL_DIR, "trigger_classes.json"), "w", encoding="utf-8") as f:
            json.dump(classes, f, ensure_ascii=False, indent=2)

        self.stdout.write(self.style.SUCCESS(f"Saved model to {saved_to}"))



def _ffmpeg_read_mono_32f(path: str, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """
    Надежное чтение через ffmpeg:
    - добавляем WindowsApps в PATH (winget-алиасы),
    - пробуем просто "ffmpeg",
    - при неудаче перебираем типичные пути,
    - жестко выдаем mono float32 @ target_sr в stdout.
    """
    import os

    # 1) На Windows добавим WindowsApps в PATH (там лежат алиасы winget)
    if os.name == "nt":
        wa = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WindowsApps")
        if os.path.isdir(wa):
            cur = os.environ.get("PATH", "")
            if wa not in cur:
                os.environ["PATH"] = wa + os.pathsep + cur

    # 2) Базовая команда
    base_cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-i", path,
        "-map_metadata", "-1",
        "-vn",
        "-ac", "1",
        "-ar", str(target_sr),
        "-f", "f32le",
        "pipe:1",
    ]

    def _try_run(cmd0: str) -> np.ndarray:
        cmd = [cmd0] + base_cmd[1:]
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
        if not out:
            raise RuntimeError("ffmpeg вернул пустой вывод для файла: " + path)
        return np.frombuffer(out, dtype=np.float32)

    # 3) Пытаемся обычным путем
    try:
        y = _try_run("ffmpeg")
        return y, target_sr
    except FileNotFoundError:
        pass  # попробуем найти по другим путям
    except subprocess.CalledProcessError as e:
        # ffmpeg найден, но файл/кодек сбоит — пробуем еще пути на случай конфликтов alias
        last_err = e

    # 4) Типичные пути в Windows
    candidates = []
    if os.name == "nt":
        la = os.environ.get("LOCALAPPDATA", "")
        candidates += [
            os.path.join(la, "Microsoft", "WindowsApps", "ffmpeg.exe"),
            r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\ffmpeg\bin\ffmpeg.exe",
        ]

    for exe in candidates:
        if os.path.exists(exe):
            try:
                y = _try_run(exe)
                return y, target_sr
            except Exception:
                continue

    # 5) Последняя попытка: переменная окружения FFMPEG_BIN (если задашь вручную)
    ffenv = os.environ.get("FFMPEG_BIN")
    if ffenv and os.path.exists(ffenv):
        y = _try_run(ffenv)
        return y, target_sr

    # 6) Если дошли сюда — реально не нашли или не смогли декодировать
    raise RuntimeError(
        "ffmpeg не найден или недоступен из Python. "
        "Решения: "
        "а) перезапусти PowerShell/IDE после установки winget ffmpeg; "
        "б) проверь, что в PATH есть %LOCALAPPDATA%\\Microsoft\\WindowsApps; "
        "в) задай FFMPEG_BIN=полный_путь_к_ffmpeg.exe; "
        "г) либо конвертируй аудио в WAV вручную."
    )

