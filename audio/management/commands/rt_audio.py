# audio/management/commands/rt_audio.py
import sys, time
import numpy as np
import sounddevice as sd
from django.core.management.base import BaseCommand

from audio.realtime import RealTimeProcessor, RTConfig, SR, BLOCK

class Command(BaseCommand):
    help = "Real-time audio pipeline: mic -> filters -> headphones"

    def add_arguments(self, parser):
        parser.add_argument("--in", dest="input_index", type=int, default=None,
                            help="Input device index (microphone)")
        parser.add_argument("--out", dest="output_index", type=int, default=None,
                            help="Output device index (headphones/speakers)")
        parser.add_argument("--list", action="store_true", help="List audio devices and exit")
        parser.add_argument("--sensitivity", type=float, default=0.12, help="Trigger sensitivity (lower = stronger)")
        parser.add_argument("--vacuum", type=float, default=0.9, help="Vacuum strength 0..1")
        parser.add_argument("--no-protect-speech", action="store_true", help="Do not protect speech")
        parser.add_argument("--no-triggers", action="store_true", help="Disable trigger detection")
        parser.add_argument("--no-vacuum", action="store_true", help="Disable vacuum")
        parser.add_argument("--no-denoise", action="store_true", help="Disable denoise")

    def handle(self, *args, **opts):
        if opts["list"]:
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                mark = ">" if i == sd.default.device[0] else ("<" if i == sd.default.device[1] else " ")
                name = d["name"]
                hostapi = sd.query_hostapis()[d["hostapi"]]["name"]
                print(f"{mark:2}{i:3d} {name}, {hostapi} ({d['max_input_channels']} in, {d['max_output_channels']} out)")
            return

        in_idx  = opts["input_index"]
        out_idx = opts["output_index"]
        if in_idx is None or out_idx is None:
            self.stderr.write(self.style.ERROR(
                "Укажи индексы устройств: python manage.py rt_audio --list  (посмотреть), "
                "затем python manage.py rt_audio --in <in_idx> --out <out_idx>"
            ))
            return

        # Автоматический выбор числа каналов
        in_info  = sd.query_devices(in_idx)
        out_info = sd.query_devices(out_idx)

        if in_info["max_input_channels"] < 1:
            self.stderr.write(self.style.ERROR(f"Устройство ввода {in_idx} не поддерживает запись"))
            return
        if out_info["max_output_channels"] < 1:
            self.stderr.write(self.style.ERROR(f"Устройство вывода {out_idx} не поддерживает воспроизведение"))
            return

        # Берём 2 канала, если устройство того требует/любит стерео; иначе 1
        input_channels  = 2 if in_info["max_input_channels"]  >= 2 else 1
        output_channels = 2 if out_info["max_output_channels"] >= 2 else 1

        cfg = RTConfig(
            trigger_sensitivity=opts["sensitivity"],
            vacuum_strength=opts["vacuum"],
            protect_speech=not opts["no_protect_speech"],
            enable_triggers=not opts["no_triggers"],
            enable_vacuum=not opts["no_vacuum"],
            enable_denoise=not opts["no_denoise"],
        )
        proc = RealTimeProcessor(cfg)

        def callback(indata, outdata, frames, time_info, status):
            if status:
                sys.stderr.write(str(status) + "\n")

            # in: сводим к моно
            if indata.ndim == 2 and indata.shape[1] > 1:
                x = indata.mean(axis=1).copy()
            else:
                x = indata[:, 0].copy()

            y = proc.process_block(x)

            # out: моно -> нужное число каналов
            if outdata.ndim == 2 and outdata.shape[1] > 1:
                outdata[:] = np.tile(y.reshape(-1, 1), (1, outdata.shape[1]))
            else:
                outdata[:, 0] = y.reshape(-1,)

        stream = sd.Stream(
            samplerate=SR,
            blocksize=BLOCK,
            dtype='float32',
            channels=None,                # не задаём одно число
            input_channels=input_channels,
            output_channels=output_channels,
            callback=callback,
            device=(in_idx, out_idx),
        )

        self.stdout.write(self.style.SUCCESS(
            f"Real-time started @ {SR} Hz, block {BLOCK} samples.\n"
            f"Input device index: {in_idx} ({input_channels} ch), "
            f"Output device index: {out_idx} ({output_channels} ch)\n"
            f"Press Ctrl+C to stop."
        ))
        try:
            with stream:
                while True:
                    time.sleep(1.0)
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING("Stopped."))
