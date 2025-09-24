# audio/management/commands/rt_audio.py
import sys, time
import numpy as np
import sounddevice as sd
from django.core.management.base import BaseCommand

# ленивый импорт realtime после выбора sample rate

class Command(BaseCommand):
    help = "Real-time audio pipeline: mic -> filters -> headphones"

    def add_arguments(self, parser):
        parser.add_argument("--in", dest="input_index", type=int, default=None,
                            help="Input device index (microphone)")
        parser.add_argument("--out", dest="output_index", type=int, default=None,
                            help="Output device index (headphones/speakers)")
        parser.add_argument("--list", action="store_true", help="List audio devices and exit")
        parser.add_argument("--sr", type=int, default=None, help="Stream sample rate (try device default; fallback auto)")
        parser.add_argument("--sensitivity", type=float, default=0.12, help="Trigger sensitivity (lower = stronger)")
        parser.add_argument("--vacuum", type=float, default=0.9, help="Vacuum strength 0..1")
        parser.add_argument("--no-protect-speech", action="store_true", help="Do not protect speech")
        parser.add_argument("--no-triggers", action="store_true", help="Disable trigger detection")
        parser.add_argument("--no-vacuum", action="store_true", help="Disable vacuum")
        parser.add_argument("--no-denoise", action="store_true", help="Disable denoise")
        parser.add_argument("--block", type=int, default=256, help="Block size in stream samples")

    def handle(self, *args, **opts):
        if opts["list"]:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            default_in, default_out = sd.default.device
            for i, d in enumerate(devices):
                mark = ">"
                if i == default_in:   mark = ">"
                elif i == default_out: mark = "<"
                name = d["name"]
                hostapi = hostapis[d["hostapi"]]["name"]
                print(f"{mark:2}{i:3d} {name}, {hostapi} ({d['max_input_channels']} in, {d['max_output_channels']} out)  "
                      f"default_sr={d.get('default_samplerate', 'n/a')}")
            return

        in_idx  = opts["input_index"]
        out_idx = opts["output_index"]
        if in_idx is None or out_idx is None:
            self.stderr.write(self.style.ERROR(
                "Укажи индексы устройств: python manage.py rt_audio --list  (посмотреть), "
                "затем python manage.py rt_audio --in <in_idx> --out <out_idx>"
            ))
            return

        in_info  = sd.query_devices(in_idx)
        out_info = sd.query_devices(out_idx)
        if in_info["max_input_channels"] < 1:
            self.stderr.write(self.style.ERROR(f"Устройство ввода {in_idx} не поддерживает запись")); return
        if out_info["max_output_channels"] < 1:
            self.stderr.write(self.style.ERROR(f"Устройство вывода {out_idx} не поддерживает воспроизведение")); return

        in_ch  = 2 if in_info["max_input_channels"]  >= 2 else 1
        out_ch = 2 if out_info["max_output_channels"] >= 2 else 1

        # Выбираем рабочую частоту потока
        candidates = []
        if opts["sr"]: candidates.append(int(opts["sr"]))
        if out_info.get("default_samplerate"): candidates.append(int(out_info["default_samplerate"]))
        if in_info.get("default_samplerate"):  candidates.append(int(in_info["default_samplerate"]))
        candidates += [48000, 44100, 32000, 16000]
        # Убираем дубликаты, сохраняя порядок
        seen = set(); sr_list = []
        for s in candidates:
            if s not in seen:
                seen.add(s); sr_list.append(s)

        from audio.realtime import RealTimeProcessor, RTConfig  # ленивый импорт

        cfg = RTConfig(
            trigger_sensitivity=opts["sensitivity"],
            vacuum_strength=opts["vacuum"],
            protect_speech=not opts["no_protect_speech"],
            enable_triggers=not opts["no_triggers"],
            enable_vacuum=not opts["no_vacuum"],
            enable_denoise=not opts["no_denoise"],
        )

        stream = None
        chosen_sr = None
        last_err = None

        # Пробуем открыть поток на первом подходящем sr
        for sr in sr_list:
            try:
                proc = RealTimeProcessor(cfg, stream_sr=sr, blocksize=opts["block"])

                def callback(indata, outdata, frames, time_info, status):
                    if status:
                        sys.stderr.write(str(status) + "\n")
                    x = indata.mean(axis=1).astype(np.float32) if indata.ndim == 2 else indata[:, 0].astype(np.float32)
                    y = proc.process_block(x)
                    if outdata.ndim == 2 and outdata.shape[1] > 1:
                        outdata[:] = np.tile(y.reshape(-1, 1), (1, outdata.shape[1]))
                    else:
                        outdata[:, 0] = y.reshape(-1,)

                stream = sd.Stream(
                    samplerate=sr,
                    blocksize=opts["block"],
                    dtype='float32',
                    channels=(in_ch, out_ch),
                    callback=callback,
                    device=(in_idx, out_idx),
                )
                chosen_sr = sr
                break
            except Exception as e:
                last_err = e
                continue

        if stream is None:
            raise last_err

        self.stdout.write(self.style.SUCCESS(
            f"Real-time started @ {chosen_sr} Hz, block {opts['block']} samples.\n"
            f"Input device index: {in_idx} ({in_ch} ch), Output device index: {out_idx} ({out_ch} ch)\n"
            f"Press Ctrl+C to stop."
        ))
        try:
            with stream:
                while True:
                    time.sleep(1.0)
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING("Stopped."))
