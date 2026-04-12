"""
Shared image processing utilities for basketball analytics.
"""

import cv2
import numpy as np


def calculate_jersey_brightness(crop: np.ndarray) -> float:
    """Calculate average brightness of the jersey area of a player crop.

    Extracts the upper-middle portion of the crop (jersey region) and
    returns mean grayscale intensity.

    Args:
        crop: BGR image crop of a player.

    Returns:
        Mean brightness (0-255). Returns 128.0 for invalid input.
    """
    if crop is None or crop.size == 0:
        return 128.0

    h, w = crop.shape[:2]
    # Upper-middle portion: rows h//6..h//2, cols w//4..3w//4
    jersey_area = crop[h // 6:h // 2, w // 4:3 * w // 4]

    if jersey_area.size == 0:
        jersey_area = crop

    if len(jersey_area.shape) == 3:
        gray = cv2.cvtColor(jersey_area, cv2.COLOR_BGR2GRAY)
    else:
        gray = jersey_area

    return float(np.mean(gray))
