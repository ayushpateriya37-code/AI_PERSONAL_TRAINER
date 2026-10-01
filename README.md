# AI Personal Trainer

A real-time exercise rep counter built with [MediaPipe Pose](https://developers.google.com/mediapipe) and OpenCV.
It tracks your body through a webcam (or video file), measures joint angles, counts reps, and shows a progress bar.

## Features
- Real-time pose detection with skeleton overlay
- 7 exercises, switchable on the fly with a key press
- Joint-angle based rep counting (full range of motion = 1 rep)
- Ignores body parts the camera can't see (e.g. the hidden arm in a side view)
- On-screen rep count, progress bar, percentage, and FPS
- Works with a webcam or a video file

## Supported exercises
| `--exercise` | Exercise | Joint measured | Best camera view |
|---|---|---|---|
| `bicep_curl` (default) | Bicep curl, left arm | Left elbow | Front |
| `bicep_curl_right` | Bicep curl, right arm | Right elbow | Front |
| `squat` | Squat | Knees (average) | Side or front |
| `push_up` | Push-up | Visible elbow(s) | Side |
| `shoulder_press` | Overhead press | Elbows (average) | Front |
| `lateral_raise` | Lateral raise | Arm-to-torso (shoulders) | Front |
| `lunge` | Lunge (either leg) | More-bent knee | Side |

## Requirements
- **Python 3.9 - 3.12** (the pinned MediaPipe version has no wheels for 3.13+)
- A webcam or a sample video

## Installation
```bash
git clone https://github.com/<your-username>/ai-personal-trainer.git
cd ai-personal-trainer

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
python main.py                              # bicep curls on the default webcam
python main.py --exercise squat             # pick an exercise
python main.py --list                       # show all exercises
python main.py --exercise push_up --source data/pushups.mp4   # video file
python main.py --source 1                   # second webcam
python main.py --no-flip                    # disable webcam mirroring
```
**Keys while running:** `n` next exercise, `r` reset reps, `q` quit.
Make sure the body parts for your exercise are fully visible to the camera.

## Project structure
```
ai-personal-trainer/
├── main.py                  # entry point (webcam/video loop)
├── src/
│   ├── pose_detector.py     # MediaPipe Pose wrapper
│   ├── exercise_tracker.py  # exercise definitions, rep counting + HUD drawing
│   └── utils.py             # angle calculation
├── tests/                   # pytest unit tests
├── data/                    # put sample videos here (git-ignored)
├── requirements.txt
└── requirements-dev.txt
```

## Running tests
```bash
pip install -r requirements-dev.txt
pytest
```

## How it works
1. `PoseDetector` finds 33 body landmarks per frame.
2. An `ExerciseTracker` measures the joint angle(s) for its exercise (e.g. shoulder-elbow-wrist for curls).
3. The angle is mapped to 0% (start position) .. 100% (peak of the movement). A rep is counted when it goes 0% -> 100% -> 0%.

Each exercise's angle range (`start_angle` / `peak_angle`) is a starting point; tune it in `src/exercise_tracker.py` to match your camera angle and range of motion.

## Adding your own exercise
Subclass `ExerciseTracker`, declare the joints and angle range, and register it:
```python
class SitUpTracker(ExerciseTracker):
    name = "Sit-up"
    joints = ((L_SHOULDER, L_HIP, L_KNEE),)   # (a, b, c): angle measured at b
    start_angle, peak_angle = 140, 70

EXERCISES["sit_up"] = SitUpTracker
```
Landmark indices are in the [MediaPipe Pose docs](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker).

## Known limitations
- Angles are measured in 2D, so accuracy depends on filming from the recommended view.
- Default angle ranges are reasonable estimates and may need tuning for your setup.
- Uses MediaPipe's legacy `mp.solutions` API, which was removed in MediaPipe 0.10.30+, hence the version pin.

## License
MIT - see [LICENSE](LICENSE).
