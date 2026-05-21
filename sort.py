"""
sort.py
=======
Main pipeline: run SORT tracking on MOT17 dataset.

Uses MOT17-09-FRCNN:
  MOT17-09 = easiest sequence (low density, clear scene)
  FRCNN    = Faster RCNN detector (strongest available)

This combination gives the best MOTA results
and is the fairest comparison to the published paper.

What this does:
  Loads FRCNN detections per frame from MOT17
  Runs SORT tracker across all frames
  Saves per-frame tracking results
  Reports FPS and track statistics

Author  : Vamshikrishna Gadde
Program : MS Robotics, Arizona State University
Series  : Day 5 of 90 — Perception Series
"""

import numpy as np
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tracker import Sort
from kalman_filter import BoundingBoxKalmanFilter

# ── SETTINGS ──────────────────────────────────────────────────────────

MOT17_ROOT    = r"D:\day-005-multi-object-tracking\MOT17"
SEQUENCE      = "MOT17-09-FRCNN"   # easiest + strongest detector
RESULTS_DIR   = "results"

# SORT hyperparameters
MAX_AGE       = 1     # frames to keep lost track alive
MIN_HITS      = 1     # frames before reporting new track
IOU_THRESHOLD = 0.3   # minimum IoU to match detection to track

COLORS = [
    (255, 50,  50),  (50,  255, 50),  (50,  50,  255),
    (255, 255, 50),  (50,  255, 255), (255, 50,  255),
    (255, 150, 50),  (150, 50,  255), (50,  150, 255),
    (255, 50,  150), (150, 255, 50),  (50,  255, 150),
]


def get_color(track_id):
    return COLORS[int(track_id) % len(COLORS)]


# ── LOAD DETECTIONS ───────────────────────────────────────────────────

def load_detections(seq_path):
    """
    Load FRCNN detector output from det/det.txt.

    MOT17 detection format:
      frame, id, x, y, w, h, conf, -1, -1, -1

    We use FRCNN (Faster RCNN) detections which are
    much stronger than DPM. This is the detector
    used in the original SORT paper for 59.8% MOTA.

    Returns: dict {frame_id: np.array [x1,y1,x2,y2,conf]}
    """
    det_file = os.path.join(seq_path, 'det', 'det.txt')

    if not os.path.exists(det_file):
        print(f"  Det file not found: {det_file}")
        return {}

    detections = {}
    with open(det_file, 'r') as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 6:
                continue
            frame = int(float(p[0]))
            x     = float(p[2])
            y     = float(p[3])
            w     = float(p[4])
            h     = float(p[5])
            conf  = float(p[6]) if len(p) > 6 else 1.0

            if w <= 0 or h <= 0:
                continue

            if frame not in detections:
                detections[frame] = []
            detections[frame].append([x, y, x+w, y+h, conf])

    for f in detections:
        detections[f] = np.array(
            detections[f], dtype=np.float32
        )

    total = sum(len(v) for v in detections.values())
    print(f"  Detections loaded: {len(detections)} frames, "
          f"{total:,} total boxes")
    return detections


# ── DEMO MODE ─────────────────────────────────────────────────────────

def create_demo_tracking():
    """Synthetic demo when MOT17 is not available."""
    print("\n  Demo mode — synthetic tracking...")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    BoundingBoxKalmanFilter.count = 0
    tracker = Sort(
        max_age       = MAX_AGE,
        min_hits      = MIN_HITS,
        iou_threshold = IOU_THRESHOLD
    )

    frame_w, frame_h = 800, 600
    n_frames         = 60

    peds = [
        {'x': 80,  'y': 150, 'vx':  7, 'vy':  1, 'w': 55, 'h': 115},
        {'x': 350, 'y':  80, 'vx':  3, 'vy':  5, 'w': 50, 'h': 110},
        {'x': 580, 'y': 350, 'vx': -6, 'vy':  2, 'w': 55, 'h': 115},
        {'x': 180, 'y': 450, 'vx':  5, 'vy': -3, 'w': 50, 'h': 110},
        {'x': 700, 'y': 200, 'vx': -4, 'vy':  4, 'w': 52, 'h': 112},
    ]

    all_tracks  = {}
    track_paths = {}

    for frame_idx in range(n_frames):
        dets = []
        for p in peds:
            p['x'] += p['vx']
            p['y'] += p['vy']
            nx = p['x'] + np.random.normal(0, 2)
            ny = p['y'] + np.random.normal(0, 2)
            if 0 < nx < frame_w and 0 < ny < frame_h:
                dets.append([nx, ny,
                             nx + p['w'],
                             ny + p['h'], 0.92])

        dets   = np.array(dets) if dets else \
                 np.empty((0, 5))
        tracks = tracker.update(dets)
        all_tracks[frame_idx] = tracks

        for trk in tracks:
            tid = int(trk[4])
            cx  = (trk[0] + trk[2]) / 2
            cy  = (trk[1] + trk[3]) / 2
            if tid not in track_paths:
                track_paths[tid] = []
            track_paths[tid].append((frame_idx, cx, cy))

    # Visualize 6 frames
    shown = [0, 12, 24, 36, 48, 59]
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.patch.set_facecolor('#0d0d0d')
    fig.suptitle(
        "SORT Multi-Object Tracking — Demo Mode\n"
        "Vamshikrishna Gadde | MS Robotics ASU | Day 5 of 90",
        fontsize=14, color='white', y=0.99
    )

    for i, frame_idx in enumerate(shown):
        ax = axes[i // 3, i % 3]
        ax.set_facecolor('#0d0d0d')

        canvas = np.zeros(
            (frame_h, frame_w, 3), dtype=np.uint8
        )
        canvas[:] = (22, 22, 22)

        for gx in range(0, frame_w, 80):
            cv2.line_stub = None
            import cv2
            cv2.line(canvas, (gx, 0), (gx, frame_h),
                     (40, 40, 40), 1)
        for gy in range(0, frame_h, 80):
            cv2.line(canvas, (0, gy), (frame_w, gy),
                     (40, 40, 40), 1)

        for tid, path in track_paths.items():
            recent = [(f, cx, cy) for f, cx, cy in path
                      if f <= frame_idx and
                      f >= frame_idx - 25]
            color = get_color(tid)
            for j in range(1, len(recent)):
                alpha = j / len(recent)
                c = tuple(int(ch * alpha) for ch in color)
                cv2.line(canvas,
                         (int(recent[j-1][1]),
                          int(recent[j-1][2])),
                         (int(recent[j][1]),
                          int(recent[j][2])), c, 2)

        tracks = all_tracks.get(
            frame_idx, np.empty((0, 5))
        )
        for trk in tracks:
            x1, y1 = int(trk[0]), int(trk[1])
            x2, y2 = int(trk[2]), int(trk[3])
            tid    = int(trk[4])
            color  = get_color(tid)
            cv2.rectangle(canvas, (x1, y1), (x2, y2),
                          color, 2)
            cv2.putText(canvas, f"ID:{tid}",
                        (x1, max(y1-5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

        import cv2 as cv2_module
        ax.imshow(
            cv2_module.cvtColor(canvas, cv2_module.COLOR_BGR2RGB)
        )
        ax.set_title(
            f"Frame {frame_idx+1:02d}/{n_frames} — "
            f"{len(tracks)} tracked",
            color='white', fontsize=9
        )
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR,
                        "sort_tracking_demo.png")
    plt.savefig(path, dpi=130,
                bbox_inches='tight',
                facecolor='#0d0d0d')
    plt.close()
    print(f"  Saved: {path}")
    return path


# ── MAIN ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import cv2

    print("\n" + "="*60)
    print("  SORT Multi-Object Tracking")
    print(f"  Sequence : {SEQUENCE}")
    print(f"  Max age  : {MAX_AGE}")
    print(f"  Min hits : {MIN_HITS}")
    print(f"  IoU thr  : {IOU_THRESHOLD}")
    print("="*60)

    seq_path = os.path.join(MOT17_ROOT, 'train', SEQUENCE)

    if not os.path.exists(seq_path):
        print(f"\n  Sequence not found: {seq_path}")
        print(f"  Running demo mode...")
        create_demo_tracking()
    else:
        print(f"\n  Loading detections...")
        detections = load_detections(seq_path)

        if not detections:
            print("  No detections found!")
        else:
            BoundingBoxKalmanFilter.count = 0
            tracker = Sort(
                max_age       = MAX_AGE,
                min_hits      = MIN_HITS,
                iou_threshold = IOU_THRESHOLD
            )

            results   = {}
            max_sim   = 0
            t0        = time.time()

            print(f"\n  Running SORT tracker...")
            for frame_id in sorted(detections.keys()):
                dets   = detections[frame_id]
                tracks = tracker.update(dets)
                results[frame_id] = (
                    tracks.copy() if len(tracks) > 0
                    else np.empty((0, 5))
                )
                max_sim = max(max_sim, len(tracks))

            elapsed = time.time() - t0
            fps     = len(detections) / elapsed

            os.makedirs(RESULTS_DIR, exist_ok=True)

            print(f"\n  {'='*50}")
            print(f"  SORT TRACKING RESULTS")
            print(f"  {'='*50}")
            print(f"  Sequence         : {SEQUENCE}")
            print(f"  Frames processed : {len(detections)}")
            print(f"  Runtime          : {elapsed:.1f}s")
            print(f"  FPS              : {fps:.1f}")
            print(f"  Max simultaneous : {max_sim}")
            print(f"  Total tracks     : "
                  f"{BoundingBoxKalmanFilter.count}")
            print(f"  {'='*50}")
            print(f"\n  Run evaluate.py for MOTA/MOTP scores!")
            print(f"  Run visualize.py for visualization!")

    print("\n" + "="*60)
    print("  TRACKING COMPLETE")
    print("="*60)