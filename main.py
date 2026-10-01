import argparse
import sys
import time

import cv2

from src.exercise_tracker import EXERCISES, create_tracker
from src.pose_detector import PoseDetector


def parse_args():
    parser = argparse.ArgumentParser(description="AI Personal Trainer - exercise rep counter")
    parser.add_argument(
        "--exercise", default="bicep_curl", choices=list(EXERCISES),
        help="Exercise to track (default: bicep_curl)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available exercises and exit",
    )
    parser.add_argument(
        "--source", default="0",
        help="Webcam index (e.g. 0) or path to a video file (default: 0)",
    )
    parser.add_argument(
        "--no-flip", action="store_true", help="Disable mirroring of webcam frames",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("Available exercises:")
        for key, cls in EXERCISES.items():
            print(f"  {key:18} {cls.name}")
        return

    is_webcam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_webcam else args.source)
    if not cap.isOpened():
        sys.exit(f"Error: could not open video source '{args.source}'.")

    names = list(EXERCISES)
    current = names.index(args.exercise)
    detector = PoseDetector()
    tracker = create_tracker(names[current])
    p_time = 0.0

    try:
        while True:
            success, img = cap.read()
            if not success:
                print("Video ended or frame could not be read.")
                break

            if is_webcam and not args.no_flip:
                img = cv2.flip(img, 1)

            img = detector.find_pose(img, draw=True)
            lm_list = detector.find_position(img)
            img = tracker.track(img, lm_list)

            c_time = time.time()
            fps = 1 / (c_time - p_time) if p_time and c_time > p_time else 0
            p_time = c_time
            cv2.putText(img, f"FPS: {int(fps)}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(img, "n: next exercise  r: reset  q: quit",
                        (10, img.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("AI Personal Trainer", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("n"):  # cycle to the next exercise
                current = (current + 1) % len(names)
                tracker = create_tracker(names[current])
            elif key == ord("r"):  # reset the rep counter
                tracker.reset()
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
