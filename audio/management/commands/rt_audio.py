# audio/management/commands/rt_audio.py
import argparse, sys, time
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
            print(sd.query_devices())
            return

        cfg = RTConfig(
            trigger_sensitivity=opts["sensitivity"],
            vacuum_strength=opts["vacuum"],
            protect_speech=not opts["no_protect_speech"],
            enable_triggers=not opts["no_triggers"],
            enable_vacuum=not opts["no_vacuum"],
            enable_denoise=not opts["no_denoise"],
        )
        proc = RealTimeProcessor(cfg)

        # Проверка устройств
        in_idx  = opts["input_index"]
        out_idx = opts["output_index"]
        if in_idx is None or out_idx is None:
            self.stderr.write(self.style.ERROR(
                "Укажи индексы устройств: python manage.py rt_audio --list  (посмотреть), "
                "затем python manage.py rt_audio --in <in_idx> --out <out_idx>"
            ))
            return

        # Открываем full-duplex поток (моно float32)
        def callback(indata, outdata, frames, time_info, status):
            if status:
                # печатаем, но продолжаем
                sys.stderr.write(str(status) + "\n")
            x = indata[:, 0].copy()
            y = proc.process_block(x)
            outdata[:, 0] = y.reshape(-1,)

        stream = sd.Stream(
            samplerate=SR,
            blocksize=BLOCK,
            dtype='float32',
            channels=1,
            callback=callback,
            device=(in_idx, out_idx)
        )

        self.stdout.write(self.style.SUCCESS(
            f"Real-time started @ {SR} Hz, block {BLOCK} samples.\n"
            f"Input device index: {in_idx}, Output device index: {out_idx}\n"
            f"Press Ctrl+C to stop."
        ))
        try:
            with stream:
                while True:
                    time.sleep(1.0)
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING("Stopped."))
