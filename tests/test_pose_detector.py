import numpy as np

from src.pose_detector import PoseDetector


def test_blank_frame_returns_no_landmarks():
    detector = PoseDetector(static_mode=True)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detector.find_pose(frame)
    assert detector.find_position(frame) == []
    detector.close()


def test_find_position_before_find_pose_is_safe():
    detector = PoseDetector()
    assert detector.find_position(np.zeros((10, 10, 3), dtype=np.uint8)) == []
    detector.close()
