"""
kalman_filter.py
================
Kalman Filter for tracking objects in 2D image space.

State vector: [x, y, s, r, dx, dy, ds]
  x, y  = center position
  s     = scale (area of bounding box)
  r     = aspect ratio (width/height)
  dx,dy = velocity of x and y
  ds    = velocity of scale

This is the exact Kalman filter used in SORT.
Used in production at Waymo and Mobileye
as the prediction component of their trackers.

Author  : Vamshikrishna Gadde
Program : MS Robotics, Arizona State University
Series  : Day 5 of 90 — Perception Series
"""

import numpy as np
from filterpy.kalman import KalmanFilter


def convert_bbox_to_z(bbox):
    """
    Convert bounding box [x1,y1,x2,y2] to state vector [x,y,s,r].

    x = center x
    y = center y
    s = scale (area)
    r = aspect ratio (width/height)
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Convert state vector [x,y,s,r] back to [x1,y1,x2,y2].
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([
            x[0] - w/2.0,
            x[1] - h/2.0,
            x[0] + w/2.0,
            x[1] + h/2.0
        ]).reshape((1, 4))
    else:
        return np.array([
            x[0] - w/2.0,
            x[1] - h/2.0,
            x[0] + w/2.0,
            x[1] + h/2.0,
            score
        ]).reshape((1, 5))


class BoundingBoxKalmanFilter:
    """
    Kalman Filter for a single tracked object.

    Models object motion as constant velocity.
    Predicts next position before seeing the new frame.
    Updates estimate using new detection if available.

    This prediction-update cycle is what makes
    tracking robust to brief occlusions.
    Even if the detector misses an object for 3 frames,
    the Kalman filter keeps predicting where it should be.
    """

    count = 0   # global ID counter

    def __init__(self, bbox):
        """Initialize tracker with first detection."""

        # 7 state variables, 4 measurements
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix
        # Assumes constant velocity model
        # x_new = x_old + dx*dt (dt=1 frame)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],   # x
            [0, 1, 0, 0, 0, 1, 0],   # y
            [0, 0, 1, 0, 0, 0, 1],   # s
            [0, 0, 0, 1, 0, 0, 0],   # r
            [0, 0, 0, 0, 1, 0, 0],   # dx
            [0, 0, 0, 0, 0, 1, 0],   # dy
            [0, 0, 0, 0, 0, 0, 1],   # ds
        ])

        # Measurement matrix
        # We only observe x, y, s, r (not velocities)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])

        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0

        # Covariance matrix
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P        *= 10.0

        # Process noise
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialize state from first detection
        self.kf.x[:4] = convert_bbox_to_z(bbox)

        # Track metadata
        self.time_since_update = 0
        self.id                = BoundingBoxKalmanFilter.count
        BoundingBoxKalmanFilter.count += 1
        self.history           = []
        self.hits              = 0
        self.hit_streak        = 0
        self.age               = 0

    def predict(self):
        """
        Predict next state using motion model.

        Called at the START of each frame,
        BEFORE seeing new detections.

        This is what allows tracking through occlusion:
        even if detector misses the object,
        we still have a predicted position.
        """
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0.0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(
            convert_x_to_bbox(self.kf.x)
        )
        return self.history[-1]

    def update(self, bbox):
        """
        Update state using new detection.

        Called when Hungarian algorithm matches
        this track to a detection in the current frame.

        Combines prediction with measurement.
        Reduces uncertainty.
        """
        self.time_since_update = 0
        self.history           = []
        self.hits             += 1
        self.hit_streak       += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def get_state(self):
        """Return current bounding box estimate."""
        return convert_x_to_bbox(self.kf.x)