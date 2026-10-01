import numpy as np


def calculate_angle(a, b, c):
    """Return the 2D angle in degrees (0-180) at point ``b`` formed by a-b-c.

    Each point is an (x, y) pair, e.g. shoulder, elbow, wrist.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = abs(np.degrees(radians))

    if angle > 180.0:
        angle = 360.0 - angle

    return float(angle)
