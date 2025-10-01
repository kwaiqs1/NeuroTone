# audio/management/commands/train_triggers.py
import os, json, random
from typing import List, Tuple
import numpy as np
import librosa
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
FIX_T      = 120           # целевое число временных кадров (примерно 1.2с / 10мс)

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

def _mel_from_wav(y: np.ndarray) -> np.ndarray:
    # mel power
    S = librosa.feature.melspectrogram(
        y=y, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0, center=True
    )  # [mels, frames]
    S_db = librosa.power_to_db(S, ref=np.max)
    # нормировка
    mu, std = S_db.mean(), S_db.std() + 1e-6
    S_n = (S_db - mu) / std
    # по времени до FIX_T
    if S_n.shape[1] < FIX_T:
        pad = FIX_T - S_n.shape[1]
        S_n = np.pad(S_n, ((0,0),(0,pad)), mode="constant")
    elif S_n.shape[1] > FIX_T:
        i0 = (S_n.shape[1] - FIX_T)//2
        S_n = S_n[:, i0:i0+FIX_T]
    return S_n.astype(np.float32)  # [M,T]

def _load_clip(path: str, deterministic: bool) -> np.ndarray:
    y, _sr = librosa.load(path, sr=TARGET_SR, mono=True)
    need = int(WIN_SEC * TARGET_SR)
    if y.shape[0] < need:
        pad = need - y.shape[0]
        y = np.pad(y, (0, pad))
    else:
        if deterministic:
            i0 = (y.shape[0] - need)//2
        else:
            i0 = random.randint(0, y.shape[0] - need)
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
    help = "Train tiny local trigger classifier (TensorFlow) from data/triggers/<class>/*.mp3|*.wav"

    def add_arguments(self, parser):
        parser.add_argument("--epochs", type=int, default=18)
        parser.add_argument("--batch",  type=int, default=32)
        parser.add_argument("--val_split", type=float, default=0.15)
        parser.add_argument("--aug_per_file", type=int, default=1,
                            help="сколько случайных окон генерировать с одного файла (train)")

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
            # несколько случайных окон из одного файла
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
        model_path = os.path.join(MODEL_DIR, "trigger_cls_keras.h5")
        model.save(model_path)
        with open(os.path.join(MODEL_DIR, "trigger_classes.json"), "w", encoding="utf-8") as f:
            json.dump(classes, f, ensure_ascii=False, indent=2)
        self.stdout.write(self.style.SUCCESS(
            f"Saved model to {model_path}"
        ))

