# Multi-Object Tracker

Real-time multi-object tracking for traffic and urban scenes using **YOLOv8** for detection and a custom **Kalman Filter + Hungarian Algorithm** pipeline for tracking.

Each detected object receives a persistent ID across frames with a visible trajectory trail.

---

## How It Works

```
Video frame
    │
    ▼
YOLOv8 Detection  →  bounding boxes + class labels
    │
    ▼
Kalman Filter     →  predicts each track's next position
    │
    ▼
Hungarian Algorithm →  optimal detection-to-track assignment (IoU cost matrix)
    │
    ▼
Track Lifecycle   →  TENTATIVE → CONFIRMED → LOST → removed
    │
    ▼
Annotated output  →  tracked video + JSON log
```

## Kalman Filter State

The filter uses a **constant-velocity motion model**:

```
State:       [cx, cy, w, h, vcx, vcy, vw, vh]
Measurement: [cx, cy, w, h]
```

- Predict step: propagates state forward using the transition matrix
- Update step: corrects prediction using the new detection measurement
- Gain (K): balances trust between prediction and measurement

---

## Quickstart

```bash
git clone https://github.com/usha-jujjuru/multi-object-tracker.git
cd multi-object-tracker
pip install -r requirements.txt

python detect_and_track.py --video sample_video/traffic.mp4 --save-json
```

Output saved to `output/tracked.mp4` and `output/tracked.json`.

---

## Usage

```bash
python detect_and_track.py \
  --video    path/to/video.mp4 \
  --output   output/tracked.mp4 \
  --model    yolov8n \
  --conf     0.3 \
  --iou      0.3 \
  --max-lost 10 \
  --min-hits 3 \
  --save-json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | required | Input video path |
| `--model` | yolov8n | YOLOv8 variant (n/s/m/l/x) |
| `--conf` | 0.3 | Detection confidence threshold |
| `--iou` | 0.3 | IoU threshold for track association |
| `--max-lost` | 10 | Frames before a lost track is removed |
| `--min-hits` | 3 | Detections before a track is confirmed |
| `--save-json` | off | Save per-frame JSON tracking log |

---

## Output

**Tracked video** — each object has:
- Unique color-coded bounding box
- Label: `ID:3 car 0.87`
- Trajectory trail showing movement path

**JSON log** (per frame):
```json
{
  "frame": 42,
  "tracks": [
    {
      "track_id": 3,
      "class_name": "car",
      "confidence": 0.87,
      "bbox": [120.0, 200.0, 350.0, 310.0],
      "center": [235, 255]
    }
  ]
}
```

**Live stats overlay:**
```
Frame  : 42
FPS    : 24.3
Active : 5
Total  : 12
```

---

## Project Structure

```
multi-object-tracker/
├── tracker/
│   ├── kalman_filter.py   # Kalman filter — state prediction and update
│   ├── track.py           # Track class — lifecycle, state, trajectory
│   ├── tracker.py         # MultiObjectTracker — association pipeline
│   └── visualizer.py      # Drawing — boxes, trails, stats overlay
├── detect_and_track.py    # Entry point
├── sample_video/          # Sample test video
├── output/                # Tracked video and JSON log (git-ignored)
└── requirements.txt
```

---

## Author

**Usha Rani Jujjuru**
M.Sc. Automotive Software Engineering — TU Chemnitz
Perception Engineer | Computer Vision | ADAS | Autonomous Driving
[LinkedIn](https://linkedin.com/in/usha-rani-jujjuru) · [GitHub](https://github.com/usha-jujjuru)
