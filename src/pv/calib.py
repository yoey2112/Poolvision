from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np

CALIB_FILENAME = "calibration.json"
CONFIG_DIRNAME = "config"

def _find_calibration_path(start: Path) -> Path:
    candidates = [start, *start.parents[:4]]
    for base in candidates:
        p = base / CONFIG_DIRNAME / CALIB_FILENAME
        if p.exists():
            return p
    # fallback: project root (../.. from pv/)
    return Path(__file__).resolve().parents[2] / CONFIG_DIRNAME / CALIB_FILENAME

def load_calibration() -> Tuple[Dict[str, Any], np.ndarray, Tuple[int,int], np.ndarray, Tuple[int,int]]:
    """
    Returns:
      raw (dict),
      H_full (3x3 float32), (full_w, full_h),
      H_play (3x3 float32), (play_w, play_h)
    """
    start = Path(__file__).resolve().parent
    path = _find_calibration_path(start)
    if not path.exists():
        raise FileNotFoundError(f"Calibration not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    H_full = np.array(data["homography_full"], dtype=np.float32)
    H_play = np.array(data["homography_play"], dtype=np.float32)
    fw = int(data["full_size"]["width"])
    fh = int(data["full_size"]["height"])
    pw = int(data["play_size"]["width"])
    ph = int(data["play_size"]["height"])

    return data, H_full, (fw, fh), H_play, (pw, ph)
