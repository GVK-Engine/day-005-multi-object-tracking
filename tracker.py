"""
tracker.py
==========
SORT: Simple Online and Realtime Tracking.

Combines:
  Kalman Filter       — predict object motion
  Hungarian Algorithm — match predictions to detections
  IoU distance        — measure similarity between boxes

This is the SORT algorithm from:
  Bewley et al. (2016) "Simple Online and Realtime Tracking"

Used as the baseline tracker in most AV systems.
Everything more advanced (DeepSORT, ByteTrack, OC-SORT)
is built on top of these exact concepts.

Author  : Vamshikrishna Gadde
Program : MS Robotics, Arizona State University
Series  : Day 5 of 90 — Perception Series
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from kalman_filter import BoundingBoxKalmanFilter, convert_x_to_bbox


# ── IoU COMPUTATION ───────────────────────────────────────────────────

def iou_batch(bb_test, bb_gt):
    """
    Compute IoU between two sets of bounding boxes.

    IoU = Intersection / Union

    IoU = 1.0  → perfect overlap (same box)
    IoU = 0.0  → no overlap at all

    Used as the similarity metric for matching
    predictions to detections.

    bb_test : (N, 4) predicted boxes
    bb_gt   : (M, 4) detected boxes
    Returns : (N, M) IoU matrix
    """
    bb_gt   = np.expand_dims(bb_gt,   0)  # (1, M, 4)
    bb_test = np.expand_dims(bb_test, 1)  # (N, 1, 4)

    # Intersection corners
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w     = np.maximum(0.0, xx2 - xx1)
    h     = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    # Union
    area_test = ((bb_test[..., 2] - bb_test[..., 0]) *
                 (bb_test[..., 3] - bb_test[..., 1]))
    area_gt   = ((bb_gt[..., 2]   - bb_gt[..., 0]) *
                 (bb_gt[..., 3]   - bb_gt[..., 1]))

    iou = inter / (area_test + area_gt - inter + 1e-6)
    return iou


# ── HUNGARIAN ASSIGNMENT ──────────────────────────────────────────────

def associate_detections_to_trackers(detections, trackers,
                                     iou_threshold=0.3):
    """
    Match detections to existing tracks using Hungarian algorithm.

    Steps:
      1. Compute IoU matrix between all predictions and detections
      2. Run Hungarian algorithm to find optimal assignment
      3. Filter matches below IoU threshold
      4. Return matched pairs + unmatched detections + unmatched tracks

    Returns
    -------
    matched        : pairs of (tracker_idx, detection_idx)
    unmatched_dets : detections with no matching track → new tracks
    unmatched_trks : tracks with no matching detection → lost tracks
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int)
        )

    # Build IoU cost matrix
    iou_matrix = iou_batch(detections, trackers)

    # Hungarian algorithm — minimize cost = maximize IoU
    if min(iou_matrix.shape) > 0:
        matched_indices = np.stack(
            linear_sum_assignment(-iou_matrix), axis=1
        )
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    # Find unmatched detections
    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 1]:
            unmatched_detections.append(d)

    # Find unmatched trackers
    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 0]:
            unmatched_trackers.append(t)

    # Filter out weak matches below IoU threshold
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[1])
            unmatched_trackers.append(m[0])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return (
        matches,
        np.array(unmatched_detections),
        np.array(unmatched_trackers)
    )


# ── SORT TRACKER ──────────────────────────────────────────────────────

class Sort:
    """
    SORT: Simple Online and Realtime Tracking.

    Core tracking pipeline per frame:
      1. Predict  : move all existing tracks forward one step
      2. Associate: match predictions to new detections
      3. Update   : update matched tracks with detections
      4. Create   : start new tracks for unmatched detections
      5. Delete   : remove tracks lost for too many frames

    Parameters
    ----------
    max_age       : frames to keep a track alive with no detection
    min_hits      : frames needed before reporting a track
    iou_threshold : minimum IoU to match detection to track
    """

    def __init__(self, max_age=3, min_hits=3,
                 iou_threshold=0.3):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.trackers      = []
        self.frame_count   = 0

    def update(self, detections=np.empty((0, 5))):
        """
        Update tracker with detections from current frame.

        detections : (N, 5) array [x1, y1, x2, y2, score]
        Returns    : (M, 5) array [x1, y1, x2, y2, track_id]
        """
        self.frame_count += 1

        # ── Step 1: Predict new positions for all tracks ──────────────
        trks      = np.zeros((len(self.trackers), 5))
        to_delete = []

        for t in range(len(trks)):
            pos = self.trackers[t].predict()[0]
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_delete.append(t)

        for t in reversed(to_delete):
            self.trackers.pop(t)

        trks = np.ma.compress_rows(
            np.ma.masked_invalid(trks)
        )

        # ── Step 2: Associate detections to tracks ────────────────────
        matched, unmatched_dets, unmatched_trks = \
            associate_detections_to_trackers(
                detections[:, :4],
                trks[:, :4],
                self.iou_threshold
            )

        # ── Step 3: Update matched tracks with detections ─────────────
        for m in matched:
            t = int(m[0])
            d = int(m[1])
            if t < len(self.trackers) and d < len(detections):
                self.trackers[t].update(detections[d, :])

        # ── Step 4: Create new tracks for unmatched detections ────────
        for i in unmatched_dets:
            if i < len(detections):
                trk = BoundingBoxKalmanFilter(detections[i, :])
                self.trackers.append(trk)

        # ── Step 5: Remove dead tracks and collect output ─────────────
        ret       = []
        to_delete = []

        for t in range(len(self.trackers)):
            trk = self.trackers[t]
            d   = trk.get_state()[0]

            # Only report confirmed tracks
            if (trk.time_since_update < 1 and
                    (trk.hit_streak >= self.min_hits or
                     self.frame_count <= self.min_hits)):
                ret.append(
                    np.concatenate((d, [trk.id + 1]))
                )

            # Mark old tracks for deletion
            if trk.time_since_update > self.max_age:
                to_delete.append(t)

        # Delete in reverse order to preserve indices
        for t in reversed(to_delete):
            self.trackers.pop(t)

        if len(ret) > 0:
            return np.stack(ret)
        return np.empty((0, 5))

    def reset(self):
        """Reset tracker state between sequences."""
        self.trackers    = []
        self.frame_count = 0
        BoundingBoxKalmanFilter.count = 0