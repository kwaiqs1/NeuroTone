import threading
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import sounddevice as sd

from .realtime import RealTimeProcessor, RTConfig


__all__ = ["rt_manager", "RTManager", "RTState"]



@dataclass
class RTState:
    running: bool = False
    frames: int = 0
    samplerate: Optional[int] = None
    blocksize: Optional[int] = None
    channels: Optional[int] = None
    in_dev: Optional[int] = None
    out_dev: Optional[int] = None
    last_error: Optional[str] = None


class RTManager:
    """
    Один-единственный менеджер реального времени.
    Содержит sounddevice.Stream, процессор и текущее состояние.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.stream: Optional[sd.Stream] = None
        self.processor: Optional[RealTimeProcessor] = None
        self.state = RTState()
        self._cfg: Optional[RTConfig] = None

    # --------- внутренний callback PortAudio ----------
    def _callback(self, indata, outdata, frames, time, status):
        try:
            if status:
                # Можно залогировать предупреждения PortAudio
                # print("PortAudio status:", status)
                pass

            # indata shape: (frames, channels), float32
            y = self.processor.process_block(indata)
            # гарантируем корректную форму/тип
            if not isinstance(y, np.ndarray):
                y = np.asarray(y, dtype=np.float32)

            if y.dtype != np.float32:
                y = y.astype(np.float32, copy=False)

            # Если процессор вернул моно при stereo-выводе — повторим канал
            if y.ndim == 1:
                y = y[:, None]

            if y.shape[1] != self.state.channels:
                if y.shape[1] == 1 and self.state.channels == 2:
                    y = np.repeat(y, 2, axis=1)
                elif y.shape[1] == 2 and self.state.channels == 1:
                    y = y[:, :1]
                else:
                    # на всякий — принудительно привести размерность
                    y = np.resize(y, (frames, self.state.channels))

            outdata[:] = y
            self.state.frames += frames
        except Exception:
            # В случае ошибки не роняем поток, а отдаем тишину
            outdata.fill(0.0)
            self.state.last_error = traceback.format_exc()

    # --------- публичные методы для views_rt ----------
    def list_devices(self) -> Dict[str, Any]:
        """Вернуть список устройств для UI."""
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        # Сформируем удобные подписи, как в UI
        items = []
        for idx, d in enumerate(devices):
            io = []
            if d['max_input_channels'] > 0:
                io.append(f"in:{d['max_input_channels']}")
            if d['max_output_channels'] > 0:
                io.append(f"out:{d['max_output_channels']}")
            io_str = ", ".join(io) if io else "—"
            host = hostapis[d['hostapi']]['name']
            items.append({
                "index": idx,
                "label": f"[{idx}] {d['name']} — {host} ({io_str})",
                "max_input": d['max_input_channels'],
                "max_output": d['max_output_channels'],
                "default_samplerate": int(d['default_samplerate']) if d['default_samplerate'] else None,
            })
        return {"ok": True, "devices": items}

    def start(self,
              in_index: int,
              out_index: int,
              samplerate: Optional[int],
              blocksize: Optional[int],
              channels: Optional[int],
              cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Создать Stream с callback, запустить. Главное: callback передаём В КОНСТРУКТОР.
        """
        with self._lock:
            # если уже работает — останавливаем
            if self.state.running:
                self._stop_locked()

            # подобрать параметры по умолчанию
            if not samplerate or int(samplerate) <= 0:
                sr = int(sd.query_devices(in_index, 'input')['default_samplerate'])
            else:
                sr = int(samplerate)

            bs = int(blocksize) if (blocksize and int(blocksize) > 0) else 480
            ch = int(channels) if (channels and int(channels) in (1, 2)) else 1

            # подготовить процессор и конфиг
            self._cfg = RTConfig(**cfg_dict)
            self.processor = RealTimeProcessor(
                sr=sr,
                block=bs,
                channels=ch,
                config=self._cfg
            )

            # ВАЖНО: callback передаем сюда, а не в .start()
            try:
                self.stream = sd.Stream(
                    device=(in_index, out_index),
                    samplerate=sr,
                    blocksize=bs,
                    dtype='float32',
                    channels=ch,
                    callback=self._callback,   # <-- ключевое изменение
                )
            except TypeError:
                # Фолбэк для старых sounddevice (иногда channels нельзя задавать единым числом)
                self.stream = sd.Stream(
                    device=(in_index, out_index),
                    samplerate=sr,
                    blocksize=bs,
                    dtype='float32',
                    # попробуем без channels — sd сам возьмёт максимально допустимое,
                    # а processor приведёт форму
                    callback=self._callback,
                )

            # запуск без аргументов
            self.stream.start()

            self.state = RTState(
                running=True,
                frames=0,
                samplerate=sr,
                blocksize=bs,
                channels=ch,
                in_dev=in_index,
                out_dev=out_index,
                last_error=None
            )
            return {
                "ok": True,
                "sr": sr,
                "block": bs,
                "channels": ch
            }

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            self._stop_locked()
            return {"ok": True}

    def _stop_locked(self):
        if self.stream is not None:
            try:
                self.stream.stop()
            finally:
                try:
                    self.stream.close()
                finally:
                    self.stream = None
        self.processor = None
        self.state = RTState(running=False)

    def stats(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "frames": self.state.frames,
            "samplerate": self.state.samplerate,
            "blocksize": self.state.blocksize,
            "channels": self.state.channels,
            "in_dev": self.state.in_dev,
            "out_dev": self.state.out_dev,
            "last_error": self.state.last_error,
        }


# Синглтон для импорта во views_rt.py
rt_manager = RTManager()
