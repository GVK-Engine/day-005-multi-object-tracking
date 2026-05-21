"""
visualize.py
============
Visualize SORT tracking results on real MOT17 data.

What this produces:
  Panel A: 6 real MOT17 frames with tracked boxes
           and motion trails — shows tracking in action
  Panel B: Track lifetime chart — shows how long
           each unique track persisted
  Panel C: Objects per frame over time

Author  : Vamshikrishna Gadde
Program : MS Robotics, Arizona State University
Series  : Day 5 of 90 — Perception Series
"""

import numpy as np
import cv2
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tracker import Sort
from kalman_filter import BoundingBoxKalmanFilter

# ── SETTINGS ──────────────────────────────────────────────────────────

MOT17_ROOT    = r"D:\day-005-multi-object-tracking\MOT17"
SEQUENCE      = "MOT17-09-FRCNN"
RESULTS_DIR   = "results"
MAX_AGE       = 3
MIN_HITS      = 1
IOU_THRESHOLD = 0.3

COLORS = [
    (255, 80,  80),  (80,  255, 80),  (80,  80,  255),
    (255, 255, 80),  (80,  255, 255), (255, 80,  255),
    (255, 160, 80),  (160, 80,  255), (80,  160, 255),
    (255, 80,  160), (160, 255, 80),  (80,  255, 160),
    (255, 200, 80),  (200, 80,  255), (80,  200, 255),
]


def get_color(tid):
    return COLORS[int(tid) % len(COLORS)]


# ── DATA LOADING ──────────────────────────────────────────────────────

def load_detections(seq_path):
    det_file = os.path.join(seq_path, 'det', 'det.txt')
    dets     = {}
    if not os.path.exists(det_file):
        return dets
    with open(det_file, 'r') as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 6:
                continue
            frame = int(float(p[0]))
            x, y  = float(p[2]), float(p[3])
            w, h  = float(p[4]), float(p[5])
            conf  = float(p[6]) if len(p) > 6 else 1.0
            if w <= 0 or h <= 0:
                continue
            if frame not in dets:
                dets[frame] = []
            dets[frame].append([x, y, x+w, y+h, conf])
    for f in dets:
        dets[f] = np.array(dets[f], dtype=np.float32)
    return dets


def load_images(seq_path):
    img_dir = os.path.join(seq_path, 'img1')
    images  = {}
    if not os.path.exists(img_dir):
        return images
    for fname in sorted(os.listdir(img_dir)):
        if fname.lower().endswith(('.jpg', '.png')):
            try:
                fid = int(os.path.splitext(fname)[0])
                images[fid] = os.path.join(img_dir, fname)
            except ValueError:
                continue
    return images


# ── RUN TRACKER AND COLLECT DATA ──────────────────────────────────────

def run_tracker(seq_path):
    """
    Run SORT on all frames and collect tracking data.
    Returns frame results, track paths, objects/frame counts.
    """
    dets   = load_detections(seq_path)
    images = load_images(seq_path)

    BoundingBoxKalmanFilter.count = 0
    tracker = Sort(
        max_age       = MAX_AGE,
        min_hits      = MIN_HITS,
        iou_threshold = IOU_THRESHOLD
    )

    frame_results  = {}    # frame_id → tracks array
    track_paths    = {}    # tid → list of (frame, cx, cy)
    track_lifespan = {}    # tid → (start_frame, end_frame)
    objects_per_frame = [] # (frame_id, n_tracks)

    for frame_id in sorted(dets.keys()):
        d      = dets.get(frame_id, np.empty((0, 5)))
        tracks = tracker.update(d)

        frame_results[frame_id] = (
            tracks.copy() if len(tracks) > 0
            else np.empty((0, 5))
        )
        objects_per_frame.append((frame_id, len(tracks)))

        for trk in tracks:
            tid = int(trk[4])
            cx  = (trk[0] + trk[2]) / 2
            cy  = (trk[1] + trk[3]) / 2

            if tid not in track_paths:
                track_paths[tid] = []
                track_lifespan[tid] = [frame_id, frame_id]
            track_paths[tid].append((frame_id, cx, cy))
            track_lifespan[tid][1] = frame_id

    return (frame_results, track_paths,
            track_lifespan, objects_per_frame, images)


# ── DRAW ONE FRAME ────────────────────────────────────────────────────

def draw_frame(frame_id, frame_results, track_paths,
               images, trail_length=40):
    """
    Draw tracked boxes and motion trails on one frame.
    Uses real MOT17 image if available.
    """
    # Load real image or create dark canvas
    if frame_id in images:
        img = cv2.imread(images[frame_id])
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((600, 1000, 3), dtype=np.uint8)
            img[:] = (25, 25, 25)
    else:
        img = np.zeros((600, 1000, 3), dtype=np.uint8)
        img[:] = (25, 25, 25)

    img_draw = img.copy()

    # Draw motion trails
    for tid, path in track_paths.items():
        recent = [
            (f, cx, cy) for f, cx, cy in path
            if frame_id - trail_length <= f <= frame_id
        ]
        if len(recent) < 2:
            continue
        color = get_color(tid)
        for i in range(1, len(recent)):
            alpha = i / len(recent)
            c = tuple(int(ch * alpha) for ch in color)
            pt1 = (int(recent[i-1][1]), int(recent[i-1][2]))
            pt2 = (int(recent[i][1]),   int(recent[i][2]))
            cv2.line(img_draw, pt1, pt2, c, 2)

    # Draw tracked bounding boxes
    tracks = frame_results.get(frame_id, np.empty((0, 5)))
    for trk in tracks:
        x1, y1 = int(trk[0]), int(trk[1])
        x2, y2 = int(trk[2]), int(trk[3])
        tid    = int(trk[4])
        color  = get_color(tid)

        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

        label_bg = (max(0, x1), max(0, y1-22))
        cv2.rectangle(img_draw,
                      label_bg,
                      (min(img_draw.shape[1], x1+70),
                       max(0, y1-2)),
                      color, -1)
        cv2.putText(img_draw, f"ID:{tid}",
                    (x1+2, max(y1-5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 2)

    return img_draw, len(tracks)


# ── MAIN VISUALIZATION ────────────────────────────────────────────────

def create_visualization():
    """
    Create complete 3-section visualization:

    Section 1 (top 2 rows): 6 tracking frames
      Shows real MOT17 pedestrian footage with
      colored ID boxes and motion trails.

    Section 2 (bottom left): Track lifetime chart
      Shows how long each unique track persisted.
      Longer bars = tracker maintained identity longer.

    Section 3 (bottom right): Objects per frame
      Shows how many objects were tracked each frame.
      Reveals tracking density over time.
    """
    seq_path = os.path.join(MOT17_ROOT, 'train', SEQUENCE)

    if not os.path.exists(seq_path):
        print(f"  Sequence not found: {seq_path}")
        print(f"  Run demo visualization instead.")
        create_demo_visualization()
        return

    print(f"\n  Loading and running tracker...")
    t0 = time.time()

    (frame_results, track_paths, track_lifespan,
     objects_per_frame, images) = run_tracker(seq_path)

    elapsed = time.time() - t0
    n_frames   = len(frame_results)
    n_tracks   = len(track_paths)
    all_frames = sorted(frame_results.keys())

    print(f"  Frames tracked  : {n_frames}")
    print(f"  Unique tracks   : {n_tracks}")
    print(f"  Runtime         : {elapsed:.1f}s")

    # Pick 6 evenly spaced frames to visualize
    indices     = [0, 0.2, 0.4, 0.6, 0.8, 0.99]
    show_frames = [
        all_frames[int(i * (len(all_frames) - 1))]
        for i in indices
    ]

    # Build figure
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor('#0d0d0d')
    fig.suptitle(
        f"SORT Multi-Object Tracking — {SEQUENCE}\n"
        "Vamshikrishna Gadde | MS Robotics ASU | Day 5 of 90",
        fontsize=15, color='white', y=0.99
    )

    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.12, wspace=0.06,
        height_ratios=[1, 1, 0.8]
    )

    # ── Section 1: 6 tracking frames ─────────────────────────────────
    for idx, frame_id in enumerate(show_frames):
        row = idx // 3
        col = idx % 3
        ax  = fig.add_subplot(gs[row, col])
        ax.set_facecolor('#0d0d0d')

        img_draw, n_trk = draw_frame(
            frame_id, frame_results,
            track_paths, images
        )
        ax.imshow(img_draw)
        ax.set_title(
            f"Frame {frame_id:04d}  |  "
            f"{n_trk} tracked objects",
            color='white', fontsize=9, pad=4
        )
        ax.axis('off')

    # ── Section 2: Track lifetime chart ──────────────────────────────
    ax_life = fig.add_subplot(gs[2, :2])
    ax_life.set_facecolor('#0d0d0d')

    # Show top 20 longest tracks
    sorted_tracks = sorted(
        track_lifespan.items(),
        key=lambda x: x[1][1] - x[1][0],
        reverse=True
    )[:20]

    if sorted_tracks:
        tids    = [str(t[0]) for t in sorted_tracks]
        starts  = [t[1][0] for t in sorted_tracks]
        lengths = [t[1][1] - t[1][0] + 1 for t in sorted_tracks]
        colors  = [
            tuple(c/255 for c in get_color(int(tid)))
            for tid in tids
        ]

        y_pos = np.arange(len(tids))
        ax_life.barh(y_pos, lengths, left=starts,
                     color=colors, height=0.7)
        ax_life.set_yticks(y_pos)
        ax_life.set_yticklabels(
            [f"ID:{t}" for t in tids],
            color='white', fontsize=8
        )
        ax_life.set_xlabel(
            "Frame Number", color='white'
        )
        ax_life.set_title(
            "Track Lifetimes — Top 20 Longest Tracks",
            color='white', fontsize=10
        )
        ax_life.tick_params(colors='white')
        for spine in ax_life.spines.values():
            spine.set_edgecolor('#444')

    # ── Section 3: Objects per frame ─────────────────────────────────
    ax_count = fig.add_subplot(gs[2, 2])
    ax_count.set_facecolor('#0d0d0d')

    if objects_per_frame:
        frames_arr = [x[0] for x in objects_per_frame]
        counts_arr = [x[1] for x in objects_per_frame]
        avg_count  = np.mean(counts_arr)

        ax_count.fill_between(
            frames_arr, counts_arr,
            alpha=0.4, color='#00C8FF'
        )
        ax_count.plot(
            frames_arr, counts_arr,
            color='#00C8FF', linewidth=1.2
        )
        ax_count.axhline(
            y=avg_count, color='yellow',
            linestyle='--', linewidth=1.2,
            label=f"Avg: {avg_count:.1f}"
        )
        ax_count.set_xlabel(
            "Frame", color='white'
        )
        ax_count.set_ylabel(
            "Tracked Objects", color='white'
        )
        ax_count.set_title(
            "Objects Tracked Per Frame",
            color='white', fontsize=10
        )
        ax_count.tick_params(colors='white')
        ax_count.legend(
            facecolor='#0d0d0d',
            labelcolor='white', fontsize=9
        )
        for spine in ax_count.spines.values():
            spine.set_edgecolor('#444')

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(
        RESULTS_DIR, "tracking_visualization.png"
    )
    plt.savefig(path, dpi=130,
                bbox_inches='tight',
                facecolor='#0d0d0d')
    plt.close()

    print(f"\n  Saved: {path}")
    print(f"\n  Summary:")
    print(f"  Unique tracks  : {n_tracks}")
    print(f"  Frames         : {n_frames}")
    if objects_per_frame:
        counts = [x[1] for x in objects_per_frame]
        print(f"  Avg tracked/frame: {np.mean(counts):.1f}")
        print(f"  Max tracked/frame: {max(counts)}")

    return path


# ── DEMO FALLBACK ─────────────────────────────────────────────────────

def create_demo_visualization():
    """Demo visualization without real MOT17 data."""
    print("\n  Creating demo visualization...")

    BoundingBoxKalmanFilter.count = 0
    tracker = Sort(
        max_age=MAX_AGE, min_hits=MIN_HITS,
        iou_threshold=IOU_THRESHOLD
    )

    frame_w, frame_h = 900, 600
    n_frames         = 60

    peds = [
        {'x': 80,  'y': 150, 'vx':  8, 'vy':  1,
         'w': 55, 'h': 120},
        {'x': 350, 'y':  80, 'vx':  3, 'vy':  5,
         'w': 50, 'h': 115},
        {'x': 600, 'y': 350, 'vx': -7, 'vy':  2,
         'w': 55, 'h': 120},
        {'x': 180, 'y': 450, 'vx':  5, 'vy': -4,
         'w': 50, 'h': 115},
        {'x': 750, 'y': 200, 'vx': -5, 'vy':  4,
         'w': 52, 'h': 118},
    ]

    all_tracks  = {}
    track_paths = {}
    opf         = []

    for fi in range(n_frames):
        dets = []
        for p in peds:
            p['x'] += p['vx']
            p['y'] += p['vy']
            nx = p['x'] + np.random.normal(0, 2)
            ny = p['y'] + np.random.normal(0, 2)
            if 0 < nx < frame_w and 0 < ny < frame_h:
                dets.append([nx, ny,
                             nx+p['w'], ny+p['h'],
                             0.92])
        dets   = np.array(dets) if dets \
                 else np.empty((0, 5))
        tracks = tracker.update(dets)
        all_tracks[fi] = tracks
        opf.append((fi, len(tracks)))

        for trk in tracks:
            tid = int(trk[4])
            cx  = (trk[0]+trk[2])/2
            cy  = (trk[1]+trk[3])/2
            if tid not in track_paths:
                track_paths[tid] = []
            track_paths[tid].append((fi, cx, cy))

    indices     = [0, 0.2, 0.4, 0.6, 0.8, 0.99]
    show_frames = [int(i*(n_frames-1)) for i in indices]

    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor('#0d0d0d')
    fig.suptitle(
        "SORT Multi-Object Tracking — Demo\n"
        "Vamshikrishna Gadde | MS Robotics ASU | Day 5 of 90",
        fontsize=14, color='white', y=0.99
    )

    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.12, wspace=0.06,
        height_ratios=[1, 1, 0.8]
    )

    for idx, fi in enumerate(show_frames):
        ax = fig.add_subplot(gs[idx//3, idx%3])
        ax.set_facecolor('#0d0d0d')

        canvas = np.zeros(
            (frame_h, frame_w, 3), dtype=np.uint8
        )
        canvas[:] = (22, 22, 22)
        for gx in range(0, frame_w, 90):
            cv2.line(canvas, (gx,0), (gx,frame_h),
                     (38,38,38), 1)
        for gy in range(0, frame_h, 90):
            cv2.line(canvas, (0,gy), (frame_w,gy),
                     (38,38,38), 1)

        for tid, path in track_paths.items():
            recent = [
                (f,cx,cy) for f,cx,cy in path
                if fi-35 <= f <= fi
            ]
            color = get_color(tid)
            for i in range(1, len(recent)):
                alpha = i / len(recent)
                c = tuple(int(ch*alpha) for ch in color)
                cv2.line(canvas,
                         (int(recent[i-1][1]),
                          int(recent[i-1][2])),
                         (int(recent[i][1]),
                          int(recent[i][2])), c, 2)

        tracks = all_tracks.get(fi, np.empty((0,5)))
        for trk in tracks:
            x1,y1 = int(trk[0]),int(trk[1])
            x2,y2 = int(trk[2]),int(trk[3])
            tid   = int(trk[4])
            color = get_color(tid)
            cv2.rectangle(canvas,(x1,y1),(x2,y2),color,2)
            cv2.rectangle(canvas,
                          (x1,max(0,y1-20)),
                          (min(frame_w,x1+65),y1),
                          color,-1)
            cv2.putText(canvas,f"ID:{tid}",
                        (x1+2,max(y1-4,12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,(0,0,0),2)

        ax.imshow(
            cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        )
        ax.set_title(
            f"Frame {fi+1:02d}/{n_frames}  |  "
            f"{len(tracks)} tracked",
            color='white', fontsize=9
        )
        ax.axis('off')

    # Track lifetime
    ax_life = fig.add_subplot(gs[2,:2])
    ax_life.set_facecolor('#0d0d0d')
    for i, (tid, path) in enumerate(track_paths.items()):
        frames = [p[0] for p in path]
        color  = tuple(c/255 for c in get_color(tid))
        ax_life.barh(i, len(frames),
                     left=min(frames),
                     color=color, height=0.7)
    ax_life.set_yticks(range(len(track_paths)))
    ax_life.set_yticklabels(
        [f"ID:{t}" for t in track_paths.keys()],
        color='white', fontsize=9
    )
    ax_life.set_xlabel("Frame", color='white')
    ax_life.set_title(
        "Track Lifetimes", color='white', fontsize=10
    )
    ax_life.tick_params(colors='white')
    for spine in ax_life.spines.values():
        spine.set_edgecolor('#444')

    # Objects per frame
    ax_c = fig.add_subplot(gs[2,2])
    ax_c.set_facecolor('#0d0d0d')
    frs = [x[0] for x in opf]
    cts = [x[1] for x in opf]
    ax_c.fill_between(frs, cts, alpha=0.4,
                      color='#00C8FF')
    ax_c.plot(frs, cts, color='#00C8FF', linewidth=1.5)
    ax_c.axhline(y=np.mean(cts), color='yellow',
                 linestyle='--', linewidth=1.2,
                 label=f"Avg: {np.mean(cts):.1f}")
    ax_c.set_xlabel("Frame", color='white')
    ax_c.set_ylabel("Tracked", color='white')
    ax_c.set_title(
        "Objects Per Frame", color='white', fontsize=10
    )
    ax_c.tick_params(colors='white')
    ax_c.legend(facecolor='#0d0d0d',
                labelcolor='white', fontsize=9)
    for spine in ax_c.spines.values():
        spine.set_edgecolor('#444')

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(
        RESULTS_DIR, "tracking_visualization.png"
    )
    plt.savefig(path, dpi=130,
                bbox_inches='tight',
                facecolor='#0d0d0d')
    plt.close()
    print(f"  Saved: {path}")
    return path


# ── MAIN ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  SORT Tracking Visualization")
    print(f"  Sequence : {SEQUENCE}")
    print("="*60)

    seq_path = os.path.join(
        MOT17_ROOT, 'train', SEQUENCE
    )

    if os.path.exists(seq_path):
        create_visualization()
    else:
        print(f"\n  Sequence not found — running demo.")
        create_demo_visualization()

    print(f"\n  Open results/tracking_visualization.png")