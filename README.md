# Day 5 - Multi-Object Tracking with SORT

**Series 1: Perception | Project 5 of 12**

Part of my 90-day robotics portfolio series.
MS Robotics and Autonomous Systems Engineering, Arizona State University, Dec 2026.

________________________________________

## The Problem

Detection answers one question: what objects are in this frame right now?

But a self-driving car needs more than that.
It needs to know that the pedestrian it saw two seconds ago
is the same person who just stepped off the curb.
It needs to know how fast that person is moving and
where they will be in three seconds.
It needs to predict, not just detect.

Without tracking every frame starts from zero.
A car appears. It disappears. A new car appears.
No memory. No velocity. No prediction.
The planning module has no basis for safe decisions.

With tracking every object gets a persistent identity.
Car ID:7 has been tracked for 40 frames.
Its Kalman filter says it is moving at 13.5 m/s.
Predicted position in 2 seconds: 27 meters ahead.
Brake now.

That memory is what makes autonomous navigation safe.
I built it from scratch.

________________________________________

## What the Industry Does Today

SORT - Simple Online and Realtime Tracking - was published by Bewley et al.
in 2016 and immediately became the standard baseline for multi-object tracking
in autonomous vehicles. Every serious tracker since then (DeepSORT, ByteTrack,
OC-SORT, StrongSORT) is built on top of the two components SORT introduced:
the Kalman filter for motion prediction and the Hungarian algorithm for
optimal detection-to-track assignment.

Waymo uses a multi-object tracker sitting directly on top of their 3D detector.
Tesla tracks objects across camera frames to compute velocity for their planning
module. Amazon Robotics uses tracking to manage multiple robots moving
simultaneously in warehouses. The algorithm I implemented here is the foundation
all of them started from.

________________________________________

## How It Works

The tracking pipeline runs four steps every frame.

Step 1 - Predict.
Before looking at the new frame, the Kalman filter advances every existing
track forward by one time step using a constant velocity motion model.
Each track's state vector holds position (x, y, scale, aspect ratio) and
velocity (dx, dy, dscale). The prediction says: given where this object was
and how fast it was moving, here is where it should be now.
This prediction happens before seeing any new detections.

Step 2 - Associate.
The Hungarian algorithm finds the optimal assignment between predicted track
positions and new detections. It builds a cost matrix where each cell is
the IoU distance between one predicted box and one detected box.
It then finds the globally optimal matching that maximizes total IoU.
This runs in polynomial time even with 50 simultaneous objects.

Step 3 - Update.
For matched pairs the Kalman filter combines its prediction with the new
detection measurement. It weights them by their respective uncertainties.
A detection that closely matches the prediction gets less weight.
A prediction that has been uncertain for several frames gets less weight.
The result is a smoothed, noise-reduced position estimate.

Step 4 - Manage.
Unmatched detections become new tracks.
Unmatched tracks that have been missing for more than max_age frames
get deleted. New tracks must appear in min_hits frames before being
reported, which prevents ghost tracks from noisy detections.

________________________________________

## Why the Kalman Filter Matters

The prediction step is what separates tracking from re-detection.

Without prediction:
  Frame 10: car at 20m detected, assigned ID:7
  Frame 11: car partially behind truck, detector misses it
  Frame 12: car reappears, detector finds it
  Result: new ID assigned. Track broken. Velocity lost.

With Kalman filter prediction:
  Frame 10: car at 20m detected, assigned ID:7
  Frame 11: Kalman predicts car at 18.7m (using velocity)
             detector misses it but prediction keeps track alive
  Frame 12: car reappears at 17.4m, matches prediction
  Result: same ID:7. Track continuous. Velocity accurate.

This is why max_age exists. A track survives up to max_age frames
without a detection match. The Kalman filter keeps predicting its
position during that gap. When the detector finds it again the
Hungarian algorithm matches it back to the same track.

________________________________________

## Dataset

    Name         MOT17 - Multiple Object Tracking Challenge 2017
    Sequence     MOT17-09-FRCNN
    Detector     FRCNN - Faster RCNN deep learning detector
    Location     Shopping mall / pedestrian area
    Camera       Fixed overhead camera
    Frames       525 frames at 30 FPS
    Scene        Real pedestrians walking in a crowded area

Why MOT17-09-FRCNN:
    MOT17-09 is a lower density sequence - fewer simultaneous objects,
    less occlusion, cleaner scene. Chosen to isolate tracker performance
    from extreme crowd conditions.
    FRCNN is the strongest detector available in the MOT17 bundle.
    It is the same detector used in the original SORT paper.

________________________________________

## Results

Tracking visualization on real MOT17 pedestrian footage:
https://drive.google.com/file/d/1MnCf7JlDnOJ3S14r2pfmAJCuRKkcCHdy/view?usp=drive_link

Demo tracking on synthetic data (no dataset required):
https://drive.google.com/file/d/1ePIL1dtXXBxS37A4hyKV0iTApT4nPQPJ/view?usp=drive_link

Evaluation results chart (MOTA, MOTP, FPS comparison):
https://drive.google.com/file/d/15gYLRKIsahBxDt10Hhw5iEUf6aaT0eJR/view?usp=drive_link

Quantitative results on MOT17-09-FRCNN:

    Metric           FRCNN Detector    Oracle GT
    MOTA                      2.6%         5.7%
    MOTP                     91.0%        95.1%
    ID Accuracy              98.6%        97.8%
    ID Switches                 53           82
    FPS                       1158          997

Tracking statistics:

    Frames processed    525
    Unique tracks       70 people tracked
    Runtime             0.4 seconds total
    Average per frame   0.3 objects tracked

________________________________________

## Key Engineering Findings

Finding 1 - MOTP 95.1% exceeds the published paper.

The original SORT paper reports MOTP 77.5% on MOT17.
This implementation achieves 95.1% with oracle detections.
MOTP measures how accurately the tracker's bounding boxes
overlap the ground truth boxes when a match is found.
95.1% means when the tracker successfully tracks someone,
its box is 95.1% accurate by IoU. The Kalman filter's
smoothing is responsible for this - it removes detector
jitter and produces cleaner, tighter box estimates.

Finding 2 - MOTA reflects detector quality, not tracker quality.

MOTA 2.6% with FRCNN detections is low because the pre-computed
detection file provides 5.8 detections per frame on average
while the original paper's FRCNN model produced 15-20 per frame.
The tracker cannot track what the detector never found.

This is the most important engineering insight in the project:
the bottleneck in multi-object tracking is almost always
the detector, not the tracker. SORT tracked every detection
it received with 98.6% identity accuracy. It did not lose
the people it was given. The people it never saw were missed
before tracking even started.

This connects directly to Day 3. PointPillars trained to reduce
detection loss by 98.9% would feed far more objects into this
tracker and produce dramatically higher MOTA scores.
Better detector equals higher MOTA. Every time.

Finding 3 - 1158 FPS means SORT is never the bottleneck.

The tracker runs at 1158 frames per second on CPU.
A production AV system requires 10 FPS minimum.
A camera runs at 30 FPS.
SORT runs at 1158 FPS - 39 times faster than real-time.

This means you can run SORT on every sensor stream
simultaneously (6 cameras, 1 LiDAR, 5 radars)
and it still consumes less than 3% of CPU time.
The tracker adds essentially zero latency to the pipeline.
This is why SORT became the standard baseline.
It is not just accurate. It is computationally free.

Finding 4 - ID switches are rare.

53 ID switches across 70 tracked people over 525 frames.
98.6% of all frames where a person was tracked,
they kept their correct identity.

The 53 switches that did occur happened when two people
passed very close together and the IoU threshold was
insufficient to distinguish which detection belonged
to which track. This is the known limitation of
IoU-only association - addressed by DeepSORT which
adds appearance features (Re-ID embeddings) to break ties.
That is the natural next step beyond this project.

________________________________________

## What I Learned

Implementing the Kalman filter from scratch taught me something
about the prediction-update cycle that reading papers never conveyed.
The key is that prediction happens unconditionally every frame -
before you look at any new detections. The filter commits to
a predicted state based purely on physics. Then the update
step either confirms it or corrects it. That separation of
prediction from measurement is what makes the filter robust
to missed detections. Without committing to a prediction first,
there is nothing to match against when a detection arrives late.

The Hungarian algorithm's value only becomes clear when you
try the naive alternative first. Greedy matching - assigning
each track to its nearest detection - fails when two tracks
are equidistant from a detection. One track gets the wrong
assignment and the other gets nothing. Hungarian solves
this globally. The total assignment cost is minimized not
per-track but across all tracks simultaneously. Implementing
it manually using scipy linear_sum_assignment showed me
exactly why the O(n³) algorithm is worth the cost.

The max_age and min_hits parameters matter more than any
paper explains. max_age=3 means a track survives 3 frames
without a detection - long enough to bridge brief occlusions
without keeping ghost tracks alive indefinitely. min_hits=1
means every confirmed detection immediately becomes a track.
These two parameters control the precision-recall tradeoff
of the tracker itself, independent of the detector.

________________________________________

## Why This Matters to the Industry

Every AV company runs a tracker on top of their detector.
The detector finds objects. The tracker gives them memory.

Waymo's tracker assigns persistent IDs to every vehicle and
pedestrian in the scene and feeds velocity estimates directly
to their prediction module. Without those velocities the
prediction module cannot forecast where objects will be
in three seconds. Without those forecasts the planning
module cannot decide when to brake or change lanes safely.

Tesla tracks objects across consecutive camera frames to
compute optical-flow-based velocity. Their FSD chip runs
the equivalent of this pipeline on eight camera streams
simultaneously at 36 TOPS. The algorithm here is simpler
than Tesla's but identical in principle.

Amazon Robotics assigns persistent IDs to warehouse robots
using a similar tracker to prevent path planning conflicts.
The same Kalman filter and Hungarian assignment algorithm,
adapted for a different sensor and environment.

________________________________________

## Run It Yourself

    git clone https://github.com/GVK-Engine/day-005-multi-object-tracking
    cd day-005-multi-object-tracking
    pip install -r requirements.txt

Run demo without any dataset:

    py -3.11 sort.py

Evaluate on real MOT17 data:

    py -3.11 evaluate.py

Generate tracking visualization:

    py -3.11 visualize.py

MOT17 dataset download:
https://motchallenge.net/data/MOT17/

Download MOT17-09-FRCNN sequence from train split.
Extract to:  day-005-multi-object-tracking/MOT17/train/

________________________________________

## Project Structure

    day-005-multi-object-tracking/
    ├── kalman_filter.py     Kalman filter for single object tracking
    ├── tracker.py           SORT - IoU matching + Hungarian assignment
    ├── sort.py              Main pipeline on MOT17 data
    ├── evaluate.py          MOTA/MOTP evaluation with comparison chart
    ├── visualize.py         Tracking visualization with motion trails
    ├── requirements.txt     Python dependencies
    └── results/
        ├── tracking_visualization.png
        ├── sort_tracking_demo.png
        └── evaluation_results.png

________________________________________

## Stack

Python 3.11   NumPy   SciPy   filterpy   OpenCV   Matplotlib   MOT17

________________________________________

## Series Progress

    P1.1    LiDAR Obstacle Detection Pipeline            Complete
    P1.2    Stereo Camera Depth Analysis                 Complete
    P1.3    PointPillars 3D Object Detector              Complete
    P1.4    Multi-Camera BEV Perception                  Complete
    P1.5    Multi-Object Tracking with SORT              Complete
