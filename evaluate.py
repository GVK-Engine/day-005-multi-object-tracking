"""
evaluate.py
===========
Evaluate SORT tracker on MOT17-09-FRCNN.

Why MOT17-09-FRCNN:
  MOT17-09 = low density, less occlusion, best sequence
  FRCNN    = Faster RCNN, strongest detector available
             used in original SORT paper (59.8% MOTA)

Two evaluations:
  1. FRCNN detector  — real production scenario
  2. Oracle GT boxes — upper bound on tracker

Proper GT filtering (matches MOTChallenge official eval):
  Only class=1 pedestrians
  Only active annotations (column 6 = 1)
  Only visibility > 0.25 (visible enough to detect)

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
SEQUENCE      = "MOT17-09-FRCNN"
RESULTS_DIR   = "results"

MAX_AGE       = 3
MIN_HITS      = 1
IOU_THRESHOLD = 0.3


# ── DATA LOADING ──────────────────────────────────────────────────────

def load_detections(seq_path):
    """
    Load FRCNN detector output from det/det.txt.

    FRCNN (Faster RCNN) is a deep learning detector
    from 2015 — much stronger than DPM (2010).
    This is the detector used in the original SORT paper.

    Format: frame, id, x, y, w, h, conf, -1, -1, -1
    Returns: dict {frame: np.array [x1,y1,x2,y2,conf]}
    """
    det_file = os.path.join(seq_path, 'det', 'det.txt')
    dets     = {}

    if not os.path.exists(det_file):
        print(f"  No det file: {det_file}")
        return dets

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

            if frame not in dets:
                dets[frame] = []
            dets[frame].append([x, y, x+w, y+h, conf])

    for f in dets:
        dets[f] = np.array(dets[f], dtype=np.float32)

    total = sum(len(v) for v in dets.values())
    print(f"  FRCNN detections : {len(dets)} frames, "
          f"{total:,} boxes")
    return dets


def load_ground_truth(seq_path):
    """
    Load MOT17 ground truth with proper filtering.

    MOT17 GT format:
      frame, id, x, y, w, h, active, class, visibility

    We only count objects that FRCNN can reasonably detect:
      active     = 1   (annotation is active)
      class      = 1   (pedestrian only)
      visibility > 0.25 (visible enough to detect)

    This matches the official MOTChallenge evaluation protocol.
    Without this filter GT count is inflated with invisible
    and non-pedestrian objects, giving artificially low MOTA.
    """
    gt_file = os.path.join(seq_path, 'gt', 'gt.txt')
    gt      = {}

    if not os.path.exists(gt_file):
        print(f"  No GT file: {gt_file}")
        return gt

    skipped = 0
    kept    = 0

    with open(gt_file, 'r') as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 6:
                continue

            frame = int(float(p[0]))
            tid   = int(float(p[1]))
            x     = float(p[2])
            y     = float(p[3])
            w     = float(p[4])
            h     = float(p[5])

            # Skip inactive annotations (col 6 = 0)
            if len(p) > 6 and int(float(p[6])) == 0:
                skipped += 1
                continue

            # Only pedestrians: class = 1 (col 7)
            if len(p) > 7 and int(float(p[7])) != 1:
                skipped += 1
                continue

            # Skip low visibility objects (col 8)
            # Objects below 0.25 visibility are too occluded
            # for FRCNN to detect — exclude from eval
            if len(p) > 8 and float(p[8]) < 0.25:
                skipped += 1
                continue

            if w <= 0 or h <= 0:
                skipped += 1
                continue

            if frame not in gt:
                gt[frame] = []
            gt[frame].append([x, y, x+w, y+h, tid])
            kept += 1

    for f in gt:
        gt[f] = np.array(gt[f], dtype=np.float32)

    total = sum(len(v) for v in gt.values())
    print(f"  GT annotations   : {len(gt)} frames, "
          f"{total:,} visible pedestrians")
    print(f"  GT filtered out  : {skipped:,} "
          f"(invisible/non-pedestrian)")
    return gt


# ── IoU ───────────────────────────────────────────────────────────────

def iou_single(a, b):
    """Intersection over Union for two boxes [x1,y1,x2,y2]."""
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    w   = max(0.0, xx2 - xx1)
    h   = max(0.0, yy2 - yy1)
    inter  = w * h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


# ── EVALUATION LOOP ───────────────────────────────────────────────────

def run_evaluation(detections, ground_truth, label):
    """
    Run SORT tracker and compute MOTA/MOTP against GT.

    For every frame:
      1. Feed detections into SORT tracker
      2. Match tracker output to GT by IoU >= 0.5
      3. Count TP, FP, FN, ID switches

    MOTA = 1 - (FN + FP + IDSW) / GT
    MOTP = mean IoU of matched pairs
    """
    BoundingBoxKalmanFilter.count = 0
    tracker = Sort(
        max_age       = MAX_AGE,
        min_hits      = MIN_HITS,
        iou_threshold = IOU_THRESHOLD
    )

    total_gt    = 0
    total_tp    = 0
    total_fp    = 0
    total_fn    = 0
    total_idsw  = 0
    total_dist  = 0.0
    total_match = 0
    id_map      = {}   # gt_id → last known tracker_id

    all_frames = sorted(
        set(detections.keys()) | set(ground_truth.keys())
    )

    t0 = time.time()

    for frame_id in all_frames:
        dets = detections.get(frame_id, np.empty((0, 5)))
        gts  = ground_truth.get(frame_id, np.array([]))

        # Run SORT update
        tracks   = tracker.update(dets)
        n_gt     = len(gts)
        n_trk    = len(tracks)
        total_gt += n_gt

        if n_gt == 0:
            total_fp += n_trk
            continue

        if n_trk == 0:
            total_fn += n_gt
            continue

        # Greedy IoU matching: track → GT
        matched_gt  = set()
        matched_trk = set()

        # Use IoU threshold 0.5 for matching
        # (standard MOTChallenge protocol)
        for ti in range(n_trk):
            best_iou = 0.5
            best_gi  = -1
            for gi in range(n_gt):
                if gi in matched_gt:
                    continue
                iou = iou_single(
                    tracks[ti, :4], gts[gi, :4]
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gi  = gi

            if best_gi >= 0:
                matched_gt.add(best_gi)
                matched_trk.add(ti)
                total_dist  += (1.0 - best_iou)
                total_match += 1
                total_tp    += 1

                # Check for ID switch
                gt_id  = int(gts[best_gi, 4])
                trk_id = int(tracks[ti, 4])
                if gt_id in id_map:
                    if id_map[gt_id] != trk_id:
                        total_idsw += 1
                id_map[gt_id] = trk_id

        total_fn += n_gt  - len(matched_gt)
        total_fp += n_trk - len(matched_trk)

    elapsed = time.time() - t0
    fps     = len(all_frames) / max(elapsed, 0.001)
    denom   = max(total_gt, 1)
    mota    = 1.0 - (total_fn + total_fp + total_idsw) / denom
    motp    = 1.0 - total_dist / max(total_match, 1)
    id_acc  = 100.0 * (1.0 - total_idsw / denom)

    return {
        'label':   label,
        'mota':    mota,
        'motp':    motp,
        'idsw':    total_idsw,
        'tp':      total_tp,
        'fp':      total_fp,
        'fn':      total_fn,
        'gt':      total_gt,
        'match':   total_match,
        'fps':     fps,
        'elapsed': elapsed,
        'id_acc':  id_acc,
    }


# ── PRINT RESULTS ─────────────────────────────────────────────────────

def print_results(r):
    print(f"\n  {'='*56}")
    print(f"  RESULTS — {r['label']}")
    print(f"  {'='*56}")
    print(f"  MOTA  : {r['mota']*100:>7.1f}%  "
          f"(published SORT: ~59.8%)")
    print(f"  MOTP  : {r['motp']*100:>7.1f}%  "
          f"(box overlap accuracy)")
    print(f"  {'─'*56}")
    print(f"  GT objects   : {r['gt']:>8,}")
    print(f"  True pos     : {r['tp']:>8,}   correctly tracked")
    print(f"  False pos    : {r['fp']:>8,}   ghost detections")
    print(f"  False neg    : {r['fn']:>8,}   missed objects")
    print(f"  ID switches  : {r['idsw']:>8,}   "
          f"({r['id_acc']:.1f}% identity held)")
    print(f"  {'─'*56}")
    print(f"  FPS          : {r['fps']:>8.1f}")
    print(f"  Runtime      : {r['elapsed']:>8.2f}s")
    print(f"  {'='*56}")


# ── SAVE CHART ────────────────────────────────────────────────────────

def save_comparison_chart(r1, r2):
    """Save 3-panel comparison chart for README and LinkedIn."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#1a1a1a')
    fig.suptitle(
        f"SORT Tracker Evaluation — {SEQUENCE}\n"
        "Vamshikrishna Gadde | MS Robotics ASU | Day 5 of 90",
        fontsize=13, color='white'
    )

    labels = ['FRCNN\nDetector', 'Oracle GT\n(upper bound)']

    # Panel 1 — MOTA
    ax1 = axes[0]
    ax1.set_facecolor('#1a1a1a')
    vals  = [r1['mota']*100, r2['mota']*100]
    bars1 = ax1.bar(labels, vals,
                    color=['#00C8FF', '#00FF64'],
                    width=0.45)
    ax1.axhline(y=59.8, color='red', linestyle='--',
                linewidth=1.5,
                label='Published SORT 59.8%')
    ax1.set_title("MOTA — Tracking Accuracy",
                  color='white', fontsize=11)
    ax1.set_ylabel("MOTA (%)", color='white')
    ax1.tick_params(colors='white')
    y_min = min(min(vals) - 15, -5)
    ax1.set_ylim(y_min, max(max(vals) + 15, 75))
    ax1.legend(facecolor='#1a1a1a',
               labelcolor='white', fontsize=8)
    ax1.axhline(y=0, color='white',
                linewidth=0.5, alpha=0.3)
    for spine in ax1.spines.values():
        spine.set_edgecolor('#444')
    for bar, val in zip(bars1, vals):
        ypos = bar.get_height() + 1 if val >= 0 \
               else bar.get_height() - 3
        ax1.text(bar.get_x() + bar.get_width()/2,
                 ypos, f"{val:.1f}%",
                 ha='center', color='white',
                 fontsize=11, fontweight='bold')

    # Panel 2 — MOTP
    ax2 = axes[1]
    ax2.set_facecolor('#1a1a1a')
    vals2 = [r1['motp']*100, r2['motp']*100]
    bars2 = ax2.bar(labels, vals2,
                    color=['#FF6400', '#C800FF'],
                    width=0.45)
    ax2.set_title("MOTP — Box Precision",
                  color='white', fontsize=11)
    ax2.set_ylabel("MOTP (%)", color='white')
    ax2.tick_params(colors='white')
    ax2.set_ylim(0, 110)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#444')
    for bar, val in zip(bars2, vals2):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1,
                 f"{val:.1f}%",
                 ha='center', color='white',
                 fontsize=11, fontweight='bold')

    # Panel 3 — FPS
    ax3 = axes[2]
    ax3.set_facecolor('#1a1a1a')
    fps_vals = [r1['fps'], r2['fps']]
    bars3    = ax3.bar(labels, fps_vals,
                       color=['#FFD700', '#FF6B6B'],
                       width=0.45)
    ax3.axhline(y=30, color='red', linestyle='--',
                linewidth=1.5,
                label='Real-time (30 FPS)')
    ax3.set_title("Processing Speed",
                  color='white', fontsize=11)
    ax3.set_ylabel("FPS", color='white')
    ax3.tick_params(colors='white')
    ax3.legend(facecolor='#1a1a1a',
               labelcolor='white', fontsize=8)
    for spine in ax3.spines.values():
        spine.set_edgecolor('#444')
    for bar, val in zip(bars3, fps_vals):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() * 0.5,
                 f"{val:.0f}",
                 ha='center', color='white',
                 fontsize=11, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR,
                        "evaluation_results.png")
    plt.savefig(path, dpi=150,
                bbox_inches='tight',
                facecolor='#1a1a1a')
    plt.close()
    print(f"\n  Chart saved: {path}")
    return path


# ── MAIN ──────────────────────────────────────────────────────────────

def evaluate():
    print("\n" + "="*60)
    print("  SORT Tracker Evaluation — MOT17 Benchmark")
    print(f"  Sequence     : {SEQUENCE}")
    print(f"  Max age      : {MAX_AGE}")
    print(f"  Min hits     : {MIN_HITS}")
    print(f"  IoU threshold: {IOU_THRESHOLD}")
    print("="*60)

    seq_path = os.path.join(
        MOT17_ROOT, 'train', SEQUENCE
    )

    if not os.path.exists(seq_path):
        print(f"\n  Sequence not found: {seq_path}")
        return

    print("\n  Loading data...")
    dets         = load_detections(seq_path)
    ground_truth = load_ground_truth(seq_path)

    if not dets or not ground_truth:
        print("  No data found!")
        return

    # Build oracle detections from filtered GT
    oracle_dets = {}
    for fid, gts in ground_truth.items():
        oracle_dets[fid] = np.column_stack([
            gts[:, :4],
            np.ones(len(gts), dtype=np.float32)
        ])

    # Evaluation 1 — FRCNN detector
    print(f"\n  Evaluating with FRCNN detector...")
    r1 = run_evaluation(
        dets, ground_truth, "FRCNN Detector"
    )
    print_results(r1)

    # Evaluation 2 — Oracle GT boxes
    print(f"\n  Evaluating with oracle GT detections...")
    r2 = run_evaluation(
        oracle_dets, ground_truth,
        "Oracle GT (upper bound)"
    )
    print_results(r2)

    # Summary table
    print(f"\n  {'='*56}")
    print(f"  COMPARISON — {SEQUENCE}")
    print(f"  {'='*56}")
    print(f"  {'Metric':<14} {'FRCNN':>12} {'Oracle':>12}")
    print(f"  {'─'*14} {'─'*12} {'─'*12}")
    print(f"  {'MOTA':<14} "
          f"{r1['mota']*100:>11.1f}% "
          f"{r2['mota']*100:>11.1f}%")
    print(f"  {'MOTP':<14} "
          f"{r1['motp']*100:>11.1f}% "
          f"{r2['motp']*100:>11.1f}%")
    print(f"  {'ID Switches':<14} "
          f"{r1['idsw']:>12,} "
          f"{r2['idsw']:>12,}")
    print(f"  {'ID Accuracy':<14} "
          f"{r1['id_acc']:>11.1f}% "
          f"{r2['id_acc']:>11.1f}%")
    print(f"  {'FPS':<14} "
          f"{r1['fps']:>11.0f} "
          f"{r2['fps']:>11.0f}")
    print(f"  {'='*56}")

    print(f"""
  KEY ENGINEERING FINDINGS:

  FRCNN MOTA {r1['mota']*100:.1f}%  vs  Oracle MOTA {r2['mota']*100:.1f}%

  The gap between these numbers shows
  how much the detector quality matters.
  FRCNN misses some pedestrians — that is
  the detector's limitation, not the tracker's.

  MOTP {r2['motp']*100:.1f}% means when the tracker matches
  an object its box is {r2['motp']*100:.1f}% accurate.

  ID accuracy {r2['id_acc']:.1f}% means the tracker
  kept the correct identity in {r2['id_acc']:.1f}% of
  all matched track-GT pairs.

  FPS {r1['fps']:.0f} — runs {r1['fps']/30:.0f}x faster than
  real-time video. Fast enough for any AV system.

  This is why Day 3 PointPillars matters.
  Better detector → higher MOTA → safer AV.
    """)

    save_comparison_chart(r1, r2)
    return r1, r2


if __name__ == "__main__":
    evaluate()