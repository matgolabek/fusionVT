import matplotlib.pyplot as plt
import time
import numpy as np


class TimeMeasurement:
    def __init__(self, context_name: str, frames: int) -> None:
        self.context_name: str = context_name
        self.frames: int = frames
        self.begin: float = None
        self.end: float = None

    def __enter__(self):
        self.begin = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()

    @property
    def time(self) -> float:
        if self.begin is None or self.end is None:
            raise RuntimeError()
        return self.end - self.begin

    @property
    def fps(self):
        return self.frames / self.time

    def __str__(self) -> str:
        t = self.time
        h = t // 60
        min = (t - h*60) // 60
        s = int(t - h*60 - min*60)
        ms = int((t - np.floor(t))*1000)

        return f"Execution time: {h}:{min}:{s}:{ms}, processed {self.frames} frames, throughput: {self.fps} fps."

    def __repr__(self) -> str:
        t = self.time
        h = t // 60
        min = (t - h*60) // 60
        s = int(t - h*60 - min*60)
        ms = int((t - np.floor(t))*1000)

        return f'TimeMeasurement(context="{self.context_name}","{h}:{min}:{s}:{ms}", frames={self.frames}, throughput={self.fps})'
