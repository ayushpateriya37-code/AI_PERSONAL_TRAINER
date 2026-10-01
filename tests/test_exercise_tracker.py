import math

import numpy as np
import pytest

from src.exercise_tracker import (
    EXERCISES, BicepCurlTracker, LungeTracker, create_tracker,
)


def set_angle(lm, triple, angle_deg, origin=(300, 300)):
    """Place landmarks so the angle at the middle joint of ``triple`` is ``angle_deg``."""
    a, b, c = triple
    bx, by = origin
    lm[b] = [b, bx, by]
    lm[a] = [a, bx, by - 100]  # first point straight above the joint
    rad = math.radians(angle_deg)
    # second point = first point's direction rotated by angle_deg around the joint
    lm[c] = [c, int(round(bx + 100 * math.sin(rad))),
             int(round(by - 100 * math.cos(rad)))]


def pose(tracker, angle):
    """33 landmarks with every joint of ``tracker`` at ``angle`` degrees."""
    lm = [[i, 0, 0] for i in range(33)]
    for triple in tracker.joints:
        set_angle(lm, triple, angle)
    return lm


def make_landmarks(elbow_angle_deg):
    """Left-arm pose (kept for the original bicep curl tests)."""
    return pose(BicepCurlTracker(), elbow_angle_deg)


# ---- original bicep curl behaviour -----------------------------------
def test_full_rep_counts_once():
    t = BicepCurlTracker()
    for angle in (170, 40, 170):
        t.update(make_landmarks(angle))
    assert int(t.count) == 1


def test_half_rep_does_not_count():
    t = BicepCurlTracker()
    t.update(make_landmarks(170))
    t.update(make_landmarks(40))
    assert int(t.count) == 0


def test_no_landmarks_is_safe():
    assert BicepCurlTracker().update([]) is None


def test_percent_matches_angle():
    t = BicepCurlTracker()
    _, per_curled = t.update(make_landmarks(50))
    _, per_ext = t.update(make_landmarks(160))
    assert per_curled > 95 and per_ext < 5


# ---- every registered exercise ---------------------------------------
@pytest.mark.parametrize("name", list(EXERCISES))
def test_each_exercise_counts_a_full_rep(name):
    t = create_tracker(name)
    for angle in (t.start_angle, t.peak_angle, t.start_angle):
        t.update(pose(t, angle))
    assert int(t.count) == 1


@pytest.mark.parametrize("name", list(EXERCISES))
def test_each_exercise_counts_three_reps(name):
    t = create_tracker(name)
    t.update(pose(t, t.start_angle))
    for _ in range(3):
        t.update(pose(t, t.peak_angle))
        t.update(pose(t, t.start_angle))
    assert int(t.count) == 3


@pytest.mark.parametrize("name", list(EXERCISES))
def test_each_exercise_draws_without_error(name):
    t = create_tracker(name)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    t.track(img, pose(t, t.peak_angle))
    t.track(img, [])  # pose not visible


@pytest.mark.parametrize("name", list(EXERCISES))
def test_percent_direction(name):
    t = create_tracker(name)
    assert t._percent(t.start_angle) == pytest.approx(0)
    assert t._percent(t.peak_angle) == pytest.approx(100)


# ---- visibility, aggregation, registry, reset ------------------------
def test_hidden_side_is_ignored():
    t = create_tracker("push_up")
    lm = pose(t, 160)
    # Left arm hidden (low visibility) and at a wrong angle; right arm is correct
    set_angle(lm, t.joints[0], 30)
    for i in t.joints[0]:
        lm[i] = lm[i] + [0.1]
    for i in t.joints[1]:
        lm[i] = lm[i] + [0.99]
    angle, _ = t.update(lm)
    assert angle == pytest.approx(160, abs=2)


def test_all_sides_hidden_returns_none():
    t = create_tracker("squat")
    lm = pose(t, 160)
    lm = [p + [0.0] for p in lm]
    assert t.update(lm) is None


def test_lunge_uses_the_more_bent_knee():
    t = LungeTracker()
    lm = [[i, 0, 0] for i in range(33)]
    set_angle(lm, t.joints[0], 170, origin=(100, 300))  # straight back leg
    set_angle(lm, t.joints[1], 90, origin=(400, 300))   # bent front leg
    angle, _ = t.update(lm)
    assert angle == pytest.approx(90, abs=2)


def test_reset_clears_count():
    t = BicepCurlTracker()
    for angle in (170, 40, 170):
        t.update(make_landmarks(angle))
    t.reset()
    assert t.count == 0 and t.stage == "start"


def test_create_tracker_unknown_name():
    with pytest.raises(ValueError):
        create_tracker("jumping_unicorn")
