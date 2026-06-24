import cv2
import colorsys


def get_color(track_id):
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR


def draw_tracks(frame, tracks):
    for track in tracks:
        x1, y1, x2, y2 = [int(v) for v in track.get_bbox()]
        color = get_color(track.track_id)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background + text
        label = f"ID:{track.track_id} {track.class_name} {track.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Trajectory trail — fades from thin/dim to thick/bright
        pts = track.trajectory
        for i in range(1, len(pts)):
            alpha     = i / len(pts)
            thickness = max(1, int(alpha * 3))
            cv2.line(frame, pts[i - 1], pts[i], color, thickness)

    return frame


def draw_stats(frame, frame_num, fps, active_tracks, total_tracked):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (230, 95), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"Frame : {frame_num}",         (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"FPS   : {fps:.1f}",           (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"Active: {active_tracks}",     (8, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0),     1)
    cv2.putText(frame, f"Total : {total_tracked}",     (8, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255),   1)

    return frame
