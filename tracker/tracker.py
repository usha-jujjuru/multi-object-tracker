import numpy as np
from scipy.optimize import linear_sum_assignment
from tracker.track import Track


def compute_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


class MultiObjectTracker:
    """
    Tracking-by-detection pipeline:
      1. Predict all existing tracks forward one step (Kalman predict)
      2. Associate detections to tracks via IoU cost + Hungarian algorithm
      3. Update matched tracks (Kalman update)
      4. Create new tracks for unmatched detections
      5. Mark unmatched tracks as LOST
      6. Remove tracks that have been lost too long
    """

    def __init__(self, iou_threshold=0.3, max_lost=10, min_hits=3):
        self.iou_threshold = iou_threshold
        self.max_lost      = max_lost
        self.min_hits      = min_hits
        self.tracks        = []

    def update(self, detections):
        """
        detections: list of dicts — {bbox, class_id, class_name, conf}
        returns:    list of confirmed Track objects
        """
        for track in self.tracks:
            track.predict()

        matched, unmatched_dets, unmatched_trks = self._associate(detections)

        for det_idx, trk_idx in matched:
            d = detections[det_idx]
            self.tracks[trk_idx].update(d["bbox"], d["class_id"], d["class_name"], d["conf"])

        for det_idx in unmatched_dets:
            d = detections[det_idx]
            self.tracks.append(Track(d["bbox"], d["class_id"], d["class_name"], d["conf"], self.min_hits))

        for trk_idx in unmatched_trks:
            self.tracks[trk_idx].mark_lost()

        self.tracks = [
            t for t in self.tracks
            if not (t.is_lost() and t.time_since_update > self.max_lost)
        ]

        return [t for t in self.tracks if t.is_confirmed()]

    def _associate(self, detections):
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self.tracks)))

        n_dets = len(detections)
        n_trks = len(self.tracks)
        cost   = np.zeros((n_dets, n_trks), dtype=float)

        for d, det in enumerate(detections):
            for t, trk in enumerate(self.tracks):
                cost[d, t] = compute_iou(det["bbox"], trk.get_bbox())

        det_ids, trk_ids = linear_sum_assignment(-cost)

        matched, unmatched_dets, unmatched_trks = [], [], []

        matched_det_set = set(det_ids)
        matched_trk_set = set(trk_ids)

        for d in range(n_dets):
            if d not in matched_det_set:
                unmatched_dets.append(d)

        for t in range(n_trks):
            if t not in matched_trk_set:
                unmatched_trks.append(t)

        for d, t in zip(det_ids, trk_ids):
            if cost[d, t] < self.iou_threshold:
                unmatched_dets.append(d)
                unmatched_trks.append(t)
            else:
                matched.append((d, t))

        return matched, unmatched_dets, unmatched_trks
