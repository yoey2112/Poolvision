from __future__ import annotations
import time
from typing import Optional, Tuple, Union
import cv2
import numpy as np

_BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "any": 0
}

def _backend_from_name(name: str) -> int:
    return _BACKENDS.get(name.lower(), cv2.CAP_DSHOW)

def find_working_camera(max_index: int = 5, backend: int = cv2.CAP_DSHOW) -> Optional[int]:
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, backend)
        if cap is not None and cap.isOpened():
            cap.release()
            return i
    return None

class Camera:
    """
    Simple camera wrapper that:
      - picks an index (or auto-discovers)
      - applies desired width/height/fps (best-effort)
      - provides read() frames and optional horizontal flip
      - tracks FPS for overlay
      - exposes actual capture properties
    """
    def __init__(
        self,
        index: Union[int, str] = "auto",
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        backend_name: str = "dshow",
        flip: bool = False
    ):
        self.backend = _backend_from_name(backend_name)
        if isinstance(index, str) and index.lower() == "auto":
            idx = find_working_camera(backend=self.backend)
            if idx is None:
                raise RuntimeError("No camera found (auto).")
            self.index = idx
        else:
            self.index = int(index)

        self.width = int(width)
        self.height = int(height)
        self.fps_req = int(fps)
        self.flip = bool(flip)

        self.cap: Optional[cv2.VideoCapture] = None
        self._last_time = None
        self._fps = 0.0

    def open(self):
        self.cap = cv2.VideoCapture(self.index, self.backend)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera at index {self.index}.")

        # Best-effort property set; some cams won’t honor all.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps_req)

        # Initialize FPS timing
        self._last_time = time.time()
        self._fps = 0.0

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.cap:
            return False, None
        ok, frame = self.cap.read()
        if not ok:
            return False, None
        if self.flip:
            frame = cv2.flip(frame, 1)
        self._update_fps()
        return True, frame

    def _update_fps(self):
        now = time.time()
        dt = now - (self._last_time or now)
        self._last_time = now
        if dt > 0:
            # EMA (smooth)
            instant = 1.0 / dt
            self._fps = (self._fps * 0.9) + (instant * 0.1) if self._fps > 0 else instant

    @property
    def fps(self) -> float:
        return float(self._fps)

    def actual_props(self) -> Tuple[int, int, float]:
        """Return (width, height, fps) actually provided by the camera."""
        if not self.cap:
            return 0, 0, 0.0
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = float(self.cap.get(cv2.CAP_PROP_FPS))
        return w, h, f

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
