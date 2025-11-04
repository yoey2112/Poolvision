# pv/camera.py
from __future__ import annotations
import time
import threading
from typing import Optional, Tuple, Union

import cv2
import numpy as np

_BACKENDS = {
    "any": 0,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "v4l2": cv2.CAP_V4L2,
    "avfoundation": cv2.CAP_AVFOUNDATION,
}

class Camera:
    """
    Non-threaded camera (baseline). Use ThreadedCamera for low-latency capture.
    Also measures runtime FPS using EMA of inter-frame times.
    """
    def __init__(
        self,
        index: Union[int, str] = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        backend_name: str = "dshow",
        flip: bool = False,
        prefer_mjpg: bool = True,
        set_buffer_sz: int = 1,  # ask driver for small buffer where supported
    ):
        self.index = 0 if index in (None, "auto") else int(index)
        self.req_width = int(width)
        self.req_height = int(height)
        self.req_fps = int(fps)
        self.flip = bool(flip)
        self.prefer_mjpg = bool(prefer_mjpg)
        self.set_buffer_sz = int(set_buffer_sz)

        self.backend = _BACKENDS.get(backend_name.lower(), 0)
        self.cap: Optional[cv2.VideoCapture] = None

        self._last_ts: Optional[float] = None
        self._fps_ema: float = 0.0
        self._ema_alpha = 0.15

        self.actual_width = None
        self.actual_height = None
        self.actual_fps_reported = None

    def open(self):
        self.cap = cv2.VideoCapture(self.index, self.backend)

        # Prefer MJPG for USB cams
        if self.prefer_mjpg:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        # Try to keep driver buffer tiny (some backends ignore this)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, float(self.set_buffer_sz))
        except Exception:
            pass

        # Request properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.req_width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.req_height))
        self.cap.set(cv2.CAP_PROP_FPS, float(self.req_fps))

        # Prime stream
        ok, _ = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read from camera")

        # Read back actuals
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.actual_fps_reported = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)

        backend_name = next((k for k, v in _BACKENDS.items() if v == self.backend), "any")
        print(
            f"Requested: {self.req_width}x{self.req_height}@{self.req_fps} ({backend_name})\n"
            f"Actual:    {self.actual_width}x{self.actual_height}@{self.actual_fps_reported:.1f}"
        )

        self._last_ts = time.perf_counter()
        self._fps_ema = 0.0

    def read(self) -> Tuple[bool, np.ndarray]:
        if self.cap is None:
            return False, None
        ok, frame = self.cap.read()
        now = time.perf_counter()
        if ok and frame is not None:
            if self.flip:
                frame = cv2.flip(frame, 1)
            if self._last_ts is not None:
                dt = now - self._last_ts
                if dt > 0:
                    inst_fps = 1.0 / dt
                    self._fps_ema = inst_fps if self._fps_ema <= 0 else (
                        self._ema_alpha * inst_fps + (1 - self._ema_alpha) * self._fps_ema
                    )
            self._last_ts = now
        return ok, frame

    @property
    def fps(self) -> float:
        if self._fps_ema > 0:
            return float(self._fps_ema)
        if self.actual_fps_reported and self.actual_fps_reported > 0:
            return float(self.actual_fps_reported)
        return 0.0

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


class ThreadedCamera(Camera):
    """
    Background thread continuously reads frames to keep the buffer fresh.
    The main thread always gets the MOST RECENT frame (older frames are dropped).
    This eliminates latency buildup and stabilizes FPS in long runs.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None

    def open(self):
        super().open()
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self):
        while not self._stop.is_set() and self.cap is not None:
            ok, frame = self.cap.read()
            now = time.perf_counter()
            if not ok or frame is None:
                time.sleep(0.001)
                continue
            if self.flip:
                frame = cv2.flip(frame, 1)
            # update fps ema here (reader loop timing)
            if self._last_ts is not None:
                dt = now - self._last_ts
                if dt > 0:
                    inst = 1.0 / dt
                    self._fps_ema = inst if self._fps_ema == 0 else (
                        self._ema_alpha * inst + (1 - self._ema_alpha) * self._fps_ema
                    )
            self._last_ts = now
            # keep only the newest frame
            with self._lock:
                self._latest = frame

    def read(self) -> Tuple[bool, np.ndarray]:
        # return the newest available frame (may be the same as last call if processing slower)
        with self._lock:
            if self._latest is None:
                return False, None
            frame = self._latest.copy()
        return True, frame

    def release(self):
        self._stop.set()
        if self._thread is not None:
            try:
                self._thread.join(timeout=0.5)
            except Exception:
                pass
            self._thread = None
        super().release()
