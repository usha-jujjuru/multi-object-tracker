import cv2
import json
import time
import argparse
from pathlib import Path
from ultralytics import YOLO

from tracker.tracker import MultiObjectTracker
from tracker.track import Track
from tracker.visualizer import draw_tracks, draw_stats


def run(video_path, output_path, model_name, conf, iou_thresh, max_lost, min_hits, save_json):
    Track._id_counter = 0

    model   = YOLO(f"{model_name}.pt")
    tracker = MultiObjectTracker(iou_threshold=iou_thresh, max_lost=max_lost, min_hits=min_hits)

    cap          = cv2.VideoCapture(video_path)
    src_fps      = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (width, height))

    json_log    = []
    frame_num   = 0
    frame_times = []

    print(f"Video   : {video_path}")
    print(f"Model   : {model_name}  |  conf={conf}  |  iou={iou_thresh}")
    print(f"Frames  : {total_frames} @ {src_fps:.1f} fps  ({width}x{height})")
    print("-" * 50)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        # ── Detection ────────────────────────────────────────────────────────
        results    = model(frame, conf=conf, verbose=False)[0]
        detections = [
            {
                "bbox":       box.xyxy[0].tolist(),
                "class_id":   int(box.cls),
                "class_name": results.names[int(box.cls)],
                "conf":       round(float(box.conf), 3),
            }
            for box in results.boxes
        ]

        # ── Tracking ─────────────────────────────────────────────────────────
        active_tracks = tracker.update(detections)

        # ── FPS ──────────────────────────────────────────────────────────────
        frame_times.append(time.perf_counter() - t0)
        if len(frame_times) > 30:
            frame_times.pop(0)
        current_fps = 1.0 / (sum(frame_times) / len(frame_times))

        # ── Visualise ────────────────────────────────────────────────────────
        frame = draw_tracks(frame, active_tracks)
        frame = draw_stats(frame, frame_num, current_fps, len(active_tracks), Track._id_counter)
        out.write(frame)

        # ── JSON log ─────────────────────────────────────────────────────────
        if save_json:
            json_log.append({
                "frame": frame_num,
                "tracks": [
                    {
                        "track_id":   t.track_id,
                        "class_name": t.class_name,
                        "confidence": t.conf,
                        "bbox":       [round(v, 1) for v in t.get_bbox()],
                        "center":     list(t.get_center()),
                    }
                    for t in active_tracks
                ],
            })

        frame_num += 1
        if frame_num % 30 == 0:
            print(f"  Frame {frame_num:4d}/{total_frames} | FPS {current_fps:5.1f} | Active tracks: {len(active_tracks)}")

    cap.release()
    out.release()

    if save_json:
        json_path = str(Path(output_path).with_suffix(".json"))
        with open(json_path, "w") as f:
            json.dump(json_log, f, indent=2)
        print(f"\nJSON log  : {json_path}")

    print(f"Output    : {output_path}")
    print(f"Total unique objects tracked: {Track._id_counter}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Object Tracker — YOLOv8 + Kalman Filter + Hungarian Algorithm")
    parser.add_argument("--video",     required=True,          help="Input video path")
    parser.add_argument("--output",    default="output/tracked.mp4", help="Output video path")
    parser.add_argument("--model",     default="yolov8n",      help="YOLOv8 model (yolov8n/s/m/l/x)")
    parser.add_argument("--conf",      type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--iou",       type=float, default=0.3, help="IoU threshold for association")
    parser.add_argument("--max-lost",  type=int,   default=10,  help="Frames before lost track is removed")
    parser.add_argument("--min-hits",  type=int,   default=3,   help="Detections before track is confirmed")
    parser.add_argument("--save-json", action="store_true",    help="Save per-frame JSON tracking log")
    args = parser.parse_args()

    Path("output").mkdir(exist_ok=True)
    run(args.video, args.output, args.model, args.conf, args.iou, args.max_lost, args.min_hits, args.save_json)
