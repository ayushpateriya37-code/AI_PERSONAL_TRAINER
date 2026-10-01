"""Angle-based rep counters for several exercises.

Every exercise is a small subclass of :class:`ExerciseTracker` that only
declares *which joint angle(s) to measure* and the angle range that
corresponds to 0% (start position) and 100% (peak of the movement).
A rep is counted when the movement goes 0% -> 100% -> 0%.
"""
import cv2
import numpy as np

from src.utils import calculate_angle

# MediaPipe Pose landmark indices
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# Landmarks less visible than this are ignored (e.g. an arm hidden in a side view)
MIN_VISIBILITY = 0.5


class ExerciseTracker:
    """Base class. Subclasses set the class attributes below."""

    name = "Exercise"
    # (a, b, c) landmark-index triples; the angle is measured at ``b``.
    # Typically one triple per body side.
    joints = ()
    start_angle = 160   # angle (degrees) at 0%   - starting position
    peak_angle = 50     # angle (degrees) at 100% - peak of the movement
    aggregate = "mean"  # how to combine sides: "mean" or "min"
    up_threshold = 95   # percent that counts as having reached the peak
    down_threshold = 5  # percent that counts as being back at the start

    def __init__(self):
        self.count = 0.0
        self.stage = "start"  # "start" or "peak"

    # ---- measurement -------------------------------------------------
    def _joint_angles(self, lm_list):
        """Angles of every joint triple whose landmarks are all visible."""
        angles = []
        for a, b, c in self.joints:
            if max(a, b, c) >= len(lm_list):
                continue
            pts = [lm_list[i] for i in (a, b, c)]
            # lm entries are [id, x, y] or [id, x, y, visibility]
            if any(len(p) > 3 and p[3] < MIN_VISIBILITY for p in pts):
                continue
            angles.append(calculate_angle(pts[0][1:3], pts[1][1:3], pts[2][1:3]))
        return angles

    def _percent(self, angle):
        if self.start_angle > self.peak_angle:
            xp, fp = (self.peak_angle, self.start_angle), (100, 0)
        else:
            xp, fp = (self.start_angle, self.peak_angle), (0, 100)
        return float(np.interp(angle, xp, fp))

    # ---- rep logic ---------------------------------------------------
    def update(self, lm_list):
        """Update the rep count. Returns (angle, percent), or None if not visible."""
        angles = self._joint_angles(lm_list)
        if not angles:
            return None

        angle = min(angles) if self.aggregate == "min" else float(np.mean(angles))
        per = self._percent(angle)

        if per >= self.up_threshold and self.stage == "start":
            self.stage = "peak"
            self.count += 0.5
        elif per <= self.down_threshold and self.stage == "peak":
            self.stage = "start"
            self.count += 0.5

        return angle, per

    def reset(self):
        self.count = 0.0
        self.stage = "start"

    # ---- drawing -----------------------------------------------------
    def track(self, img, lm_list):
        result = self.update(lm_list)
        if result is None:
            cv2.putText(img, f"{self.name}: pose not visible", (50, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return img
        _, per = result

        # Bar fills from the bottom (y=400) up to the top (y=200) as percent rises
        bar_y = int(np.interp(per, (0, 100), (400, 200)))

        cv2.putText(img, f"Reps: {int(self.count)}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(img, self.name, (50, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(img, (50, 200), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, bar_y), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f"{int(per)}%", (45, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        return img


# ---------------------------------------------------------------------
# Exercises
# ---------------------------------------------------------------------
class BicepCurlTracker(ExerciseTracker):
    """Left-arm bicep curl: elbow angle (shoulder-elbow-wrist)."""
    name = "Bicep Curl (left)"
    joints = ((L_SHOULDER, L_ELBOW, L_WRIST),)
    start_angle, peak_angle = 160, 50


class RightBicepCurlTracker(ExerciseTracker):
    """Right-arm bicep curl."""
    name = "Bicep Curl (right)"
    joints = ((R_SHOULDER, R_ELBOW, R_WRIST),)
    start_angle, peak_angle = 160, 50


class SquatTracker(ExerciseTracker):
    """Squat: knee angle (hip-knee-ankle). Best filmed from the side or front."""
    name = "Squat"
    joints = ((L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE))
    start_angle, peak_angle = 165, 100


class PushUpTracker(ExerciseTracker):
    """Push-up: elbow angle on the visible side(s). Film from the side."""
    name = "Push-up"
    joints = ((L_SHOULDER, L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW, R_WRIST))
    start_angle, peak_angle = 160, 95


class ShoulderPressTracker(ExerciseTracker):
    """Overhead press: elbow angle, starting with elbows bent at shoulder height."""
    name = "Shoulder Press"
    joints = ((L_SHOULDER, L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW, R_WRIST))
    start_angle, peak_angle = 85, 160


class LateralRaiseTracker(ExerciseTracker):
    """Lateral raise: arm-to-torso angle (hip-shoulder-elbow). Film from the front."""
    name = "Lateral Raise"
    joints = ((L_HIP, L_SHOULDER, L_ELBOW), (R_HIP, R_SHOULDER, R_ELBOW))
    start_angle, peak_angle = 20, 80


class LungeTracker(ExerciseTracker):
    """Lunge: the more-bent knee is used, so either leg counts. Film from the side."""
    name = "Lunge"
    joints = ((L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE))
    aggregate = "min"
    start_angle, peak_angle = 165, 100


# Name -> class. The first entry is the default exercise.
EXERCISES = {
    "bicep_curl": BicepCurlTracker,
    "bicep_curl_right": RightBicepCurlTracker,
    "squat": SquatTracker,
    "push_up": PushUpTracker,
    "shoulder_press": ShoulderPressTracker,
    "lateral_raise": LateralRaiseTracker,
    "lunge": LungeTracker,
}


def create_tracker(name):
    """Create a tracker by its registry name (see ``EXERCISES``)."""
    try:
        return EXERCISES[name]()
    except KeyError:
        raise ValueError(
            f"Unknown exercise '{name}'. Choose from: {', '.join(EXERCISES)}"
        ) from None
