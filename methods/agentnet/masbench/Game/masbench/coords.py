"""
Coordinate system utilities.

We keep internal grid indexing as (row, col) with origin at the top-left
for array storage. Externally, we expose positions in a math-style system
with origin at the bottom-left, x increasing to the right and y increasing up.

Use these helpers to convert between representations.
"""
from __future__ import annotations

from typing import Tuple


def to_math_xy(x_img: int, y_img: int, height: int) -> Tuple[int, int]:
    """Convert image/grid coords (x to right, y down from top) to math coords
    (x to right, y up from bottom). Requires total grid height.

    Examples:
    - top-left (0, 0) in image -> (0, height-1) in math
    - bottom-left (0, height-1) in image -> (0, 0) in math
    """
    return int(x_img), int(height - 1 - y_img)


def from_math_xy(x_math: int, y_math: int, height: int) -> Tuple[int, int]:
    """Convert math coords (origin bottom-left) back to image/grid coords
    (origin top-left). Requires total grid height.
    """
    return int(x_math), int(height - 1 - y_math)
