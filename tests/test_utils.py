import pytest

from src.utils import calculate_angle


def test_right_angle():
    assert calculate_angle((0, 1), (0, 0), (1, 0)) == pytest.approx(90)


def test_straight_line():
    assert calculate_angle((-1, 0), (0, 0), (1, 0)) == pytest.approx(180)


def test_folded_back():
    assert calculate_angle((1, 0), (0, 0), (2, 0)) == pytest.approx(0)


def test_angle_never_exceeds_180():
    assert 0 <= calculate_angle((0, -1), (0, 0), (-1, 0.1)) <= 180
