import numpy as np


class KalmanFilter:
    """
    Tracks a bounding box using a constant-velocity motion model.
    State vector: [cx, cy, w, h, vcx, vcy, vw, vh]
    Measurement:  [cx, cy, w, h]
    """

    def __init__(self):
        dt = 1.0

        # State transition matrix — constant velocity model
        self.F = np.array([
            [1, 0, 0, 0, dt, 0,  0,  0 ],
            [0, 1, 0, 0, 0,  dt, 0,  0 ],
            [0, 0, 1, 0, 0,  0,  dt, 0 ],
            [0, 0, 0, 1, 0,  0,  0,  dt],
            [0, 0, 0, 0, 1,  0,  0,  0 ],
            [0, 0, 0, 0, 0,  1,  0,  0 ],
            [0, 0, 0, 0, 0,  0,  1,  0 ],
            [0, 0, 0, 0, 0,  0,  0,  1 ],
        ], dtype=float)

        # Measurement matrix — we observe position and size only
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=float)

        # Process noise — velocity components have lower noise
        self.Q = np.eye(8, dtype=float)
        self.Q[4:, 4:] *= 0.01

        # Measurement noise
        self.R = np.eye(4, dtype=float) * 1.0

        # Initial error covariance — high uncertainty for velocity
        self.P = np.eye(8, dtype=float)
        self.P[4:, 4:] *= 100.0

        self.x = np.zeros((8, 1), dtype=float)

    def initialize(self, bbox_cxcywh):
        self.x[:4] = np.array(bbox_cxcywh, dtype=float).reshape(4, 1)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4].flatten()

    def update(self, bbox_cxcywh):
        z = np.array(bbox_cxcywh, dtype=float).reshape(4, 1)
        y = z - self.H @ self.x                          # innovation
        S = self.H @ self.P @ self.H.T + self.R          # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)         # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        return self.x[:4].flatten()

    def get_state(self):
        return self.x[:4].flatten()
