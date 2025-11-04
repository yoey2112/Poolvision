"""
Basic ball detection utilities.

This module provides a simple example of detecting billiard balls in an image.
The current implementation uses color thresholding on a green table to find
circular shapes.  It is *not* production ready—proper ball tracking should
include homography calibration, background subtraction, and tracking across
frames.
"""

from __future__ import annotations

import cv2  # type: ignore
import numpy as np  # type: ignore
from typing import List, Dict, Any


def detect_balls(image_path: str) -> List[Dict[str, Any]]:
    """
    Detect circular objects (balls) in the provided image.

    Args:
        image_path: Path to the image file (JPEG/PNG) to process.

    Returns:
        A list of dictionaries representing detected balls.  Each dictionary
        contains keys `id`, `x`, `y`, and `radius`.  Currently the `id`
        field is `None` as no tracking is implemented.
    """
    # Read image from disk
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Convert to HSV color space for easier masking of the table surface.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define a mask for the green felt of a pool table (adjust values as needed)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Invert mask to focus on non‑table areas (i.e., the balls)
    inverted = cv2.bitwise_not(mask)
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(inverted, (9, 9), 2)

    # Use HoughCircles to detect circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=30,
    )

    detected: List[Dict[str, Any]] = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for i, (x, y, r) in enumerate(circles):
            detected.append({"id": None, "x": int(x), "y": int(y), "radius": int(r)})

    return detected
