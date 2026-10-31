# Camera + LiDAR late fusion for 3D car detection (KITTI)

A pretrained PointPillars LiDAR detector (OpenPCDet) and a COCO-pretrained YOLO
camera detector, combined by late fusion: each LiDAR 3D box is projected into
the image with the KITTI calibration, matched to camera boxes by 2D IoU, and
re-scored. Projection, fusion, evaluation analysis and latency profiling are
my own code; the two detectors use public pretrained weights.

## Results (KITTI val, 3,769 frames)

| | LiDAR-only | Fused |
|---|---|---|
| 3D AP, Car moderate (R40) | | |
| 3D AP, Car hard (R40) | | |
| False positives (score >= 0.5), 0-30 m | | |
| False positives (score >= 0.5), beyond 30 m | | |

Latency per frame on one A100 (batch size 1):

| Stage | FP32 (ms) | FP16 (ms) |
|---|---|---|
| Load + pillarize (CPU) | | |
| PointPillars network (GPU) | | |
| NMS | | |
| YOLO (camera) | | |
| Fusion (CPU) | | |

## What I found

(Where fusion helped, where it hurt, and why.)

## Method

- Projection: LiDAR -> camera -> image via `P2 @ R0_rect @ Tr_velo_to_cam`; 3D box corners projected and bounded to get a 2D box (checked against OpenPCDet's own projection).
- Fusion: matched boxes (2D IoU >= threshold) get `1 - (1 - s_lidar)(1 - s_camera)`; unmatched boxes taller than 25 px are down-weighted; smaller (far) boxes are left alone because the camera detector often misses them.
- Two fusion thresholds were tuned on the val split itself, so the fused numbers are slightly optimistic.

## Files

- `kitti_utils.py`: calibration, point-cloud and label reading, projection, IoU
- `visualize.py`: LiDAR points and 3D boxes drawn on the camera image
- `run_yolo.py`: camera detections for every val image
- `fuse_eval.py`: late fusion, official KITTI AP, false positives by distance
- `latency.py`: per-stage PointPillars latency, FP32 vs FP16
