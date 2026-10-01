import cv2
import mediapipe as mp


class PoseDetector:
    """Thin wrapper around MediaPipe Pose (legacy ``mp.solutions`` API)."""

    def __init__(self, static_mode=False, complexity=1, smooth=True,
                 detection_con=0.5, track_con=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_mode,
            model_complexity=complexity,
            smooth_landmarks=smooth,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None  # set by find_pose(); avoids AttributeError

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)

        if draw and self.results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
        return img

    def find_position(self, img):
        """Return [[landmark_id, x_px, y_px, visibility], ...] (empty if no pose found)."""
        lm_list = []
        if self.results is not None and self.results.pose_landmarks:
            h, w = img.shape[:2]
            for lm_id, lm in enumerate(self.results.pose_landmarks.landmark):
                lm_list.append([lm_id, int(lm.x * w), int(lm.y * h), lm.visibility])
        return lm_list

    def close(self):
        self.pose.close()
