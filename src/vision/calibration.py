"""
Calibration utilities for PoolVision.

This module defines a basic `Calibration` class used to store the
relationship between camera coordinates and table coordinates.  In a
complete implementation, this class would include methods to compute
homographies using calibration images and provide functions to transform
points between coordinate systems.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Calibration:
    """
    Represents the calibration parameters for a pool table.

    Attributes:
        homography: 3×3 matrix mapping image coordinates to table coordinates.
        inverse_homography: 3×3 matrix mapping table coordinates to image coordinates.
    """

    homography: Optional[list[list[float]]] = None
    inverse_homography: Optional[list[list[float]]] = None

    def is_calibrated(self) -> bool:
        """Return True if calibration matrices have been computed."""
        return self.homography is not None and self.inverse_homography is not None

    def calibrate(self, image_points: list[tuple[float, float]], table_points: list[tuple[float, float]]) -> None:
        """
        Compute homography given corresponding points in image and table space.

        This stub simply stores placeholder matrices.  You should implement
        actual homography computation using OpenCV (e.g., cv2.findHomography).
        """
        if len(image_points) != len(table_points):
            raise ValueError("image_points and table_points must have the same length")

        # TODO: compute real homography using cv2.findHomography
        self.homography = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        self.inverse_homography = self.homography
