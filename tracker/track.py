from enum import Enum
from tracker.kalman_filter import KalmanFilter


class TrackState(Enum):
    TENTATIVE = 1   # newly created, not yet confirmed
    CONFIRMED = 2   # seen consistently, shown in output
    LOST      = 3   # not matched recently, pending removal


class Track:
    _id_counter = 0

    def __init__(self, bbox_xyxy, class_id, class_name, conf, min_hits=3):
        Track._id_counter += 1
        self.track_id   = Track._id_counter
        self.class_id   = class_id
        self.class_name = class_name
        self.conf       = conf
        self.min_hits   = min_hits

        self.kf = KalmanFilter()
        self.kf.initialize(self._xyxy_to_cxcywh(bbox_xyxy))

        self.state             = TrackState.TENTATIVE
        self.hits              = 1
        self.age               = 1
        self.time_since_update = 0
        self.trajectory        = []
        self._record_position()

    # ── coordinate helpers ──────────────────────────────────────────────────

    @staticmethod
    def _xyxy_to_cxcywh(bbox):
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]

    @staticmethod
    def _cxcywh_to_xyxy(state):
        cx, cy, w, h = state
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    # ── lifecycle ────────────────────────────────────────────────────────────

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox_xyxy, class_id, class_name, conf):
        self.kf.update(self._xyxy_to_cxcywh(bbox_xyxy))
        self.class_id          = class_id
        self.class_name        = class_name
        self.conf              = conf
        self.hits             += 1
        self.time_since_update = 0
        self._record_position()
        if self.hits >= self.min_hits:
            self.state = TrackState.CONFIRMED

    def mark_lost(self):
        self.state = TrackState.LOST

    # ── accessors ────────────────────────────────────────────────────────────

    def get_bbox(self):
        return self._cxcywh_to_xyxy(self.kf.get_state())

    def get_center(self):
        state = self.kf.get_state()
        return int(state[0]), int(state[1])

    def is_confirmed(self):
        return self.state == TrackState.CONFIRMED

    def is_lost(self):
        return self.state == TrackState.LOST

    # ── trajectory ───────────────────────────────────────────────────────────

    def _record_position(self):
        self.trajectory.append(self.get_center())
        if len(self.trajectory) > 40:
            self.trajectory.pop(0)
