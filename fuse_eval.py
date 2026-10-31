"""Late fusion of LiDAR 3D detections with camera 2D detections, then
evaluation of LiDAR-only vs. fused. Runs in the `lidar` environment
(it uses OpenPCDet's official KITTI evaluation code).

    python fuse_eval.py --result PATH/TO/result.pkl --yolo results/yolo_val.json \
        --infos PATH/TO/OpenPCDet/data/kitti/kitti_infos_val.pkl

How fusion works, per LiDAR box:
  1. Project the 3D box into the image (our own projection code) to get a 2D rectangle.
  2. Find the YOLO car box that overlaps it most (2D IoU).
  3. If IoU >= --iou: both sensors agree, so combine confidences:
         new = 1 - (1 - lidar_score) * (1 - yolo_conf)
     (the box is a false alarm only if BOTH detectors are wrong).
  4. If nothing matches and the box is big enough in the image that YOLO
     should have seen it: multiply its score by --penalty.
     Small (far) boxes are left alone, because YOLO misses small cars often
     and a missing camera box says little about them.
"""

import argparse
import copy
import json
import os
import pickle
import time

import numpy as np

from kitti_utils import box_to_image_rect, iou_2d

MIN_BOX_HEIGHT = 25  # pixels; KITTI's own cutoff for counting an object at all


def fuse_frame(det, yolo_boxes, P2, image_wh, iou_thr, penalty):
    """Return new scores for one frame's detections, plus our 2D projections."""
    yolo = np.array(yolo_boxes).reshape(-1, 5)
    scores = det["score"].copy()
    rects = []
    for i in range(len(scores)):
        rect = box_to_image_rect(det["location"][i], det["dimensions"][i], det["rotation_y"][i], P2, image_wh)
        rects.append(rect)
        if rect is None:  # outside the camera's view: camera can't confirm or deny
            continue
        ious = iou_2d(rect, yolo[:, :4])
        if len(ious) and ious.max() >= iou_thr:
            conf = yolo[ious.argmax(), 4]
            scores[i] = 1 - (1 - scores[i]) * (1 - conf)
        elif rect[3] - rect[1] >= MIN_BOX_HEIGHT:
            scores[i] = scores[i] * penalty
    return scores, rects


def false_positives_by_distance(dets, gts, score_thr=0.5, match_dist=2.0, split=30.0):
    """Count confident car detections that match no real object.
    A detection matches if its centre is within `match_dist` metres (on the
    ground plane) of any annotated object; Van, Truck and DontCare count as
    objects too, so detecting a van isn't called a false alarm."""
    near = far = 0
    for d, g in zip(dets, gts):
        keep = (d["name"] == "Car") & (d["score"] >= score_thr)
        det_xz = d["location"][keep][:, [0, 2]]
        gt_xz = g["location"][:, [0, 2]]
        for xz in det_xz:
            if len(gt_xz) == 0 or np.min(np.linalg.norm(gt_xz - xz, axis=1)) > match_dist:
                if np.linalg.norm(xz) <= split:
                    near += 1
                else:
                    far += 1
    return near, far


def rotate_iou_torch(boxes, query_boxes, criterion=-1, device_id=0):
    """Drop-in replacement for OpenPCDet's numba-CUDA `rotate_iou_gpu_eval`.

    The KITTI evaluation needs the overlap of rotated rectangles on the ground
    plane. OpenPCDet computes it with numba's CUDA support, which crashes on
    newer NVIDIA drivers. This computes the same thing with OpenPCDet's own
    compiled CUDA op (the one PointPillars' NMS uses).

    Boxes are [cx, cy, size_x, size_y, angle]. The numba version rotates the
    box corners clockwise by `angle`; OpenPCDet's op rotates counter-clockwise
    by `heading`, so heading = -angle gives exactly the same rectangle.
    criterion: -1 IoU, 0 overlap / area of `boxes`, 1 overlap / area of
    `query_boxes`, 2 raw overlap area (the same meanings as the original).
    """
    import torch
    from pcdet.ops.iou3d_nms import iou3d_nms_cuda

    boxes = np.asarray(boxes)
    query_boxes = np.asarray(query_boxes)
    n, k = len(boxes), len(query_boxes)
    if n == 0 or k == 0:
        return np.zeros((n, k), dtype=boxes.dtype)

    def to_7dof(b):
        t = torch.zeros((len(b), 7), dtype=torch.float32)
        b = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32))
        t[:, 0], t[:, 1] = b[:, 0], b[:, 1]  # centre
        t[:, 3], t[:, 4] = b[:, 2], b[:, 3]  # sizes along the box's own axes
        t[:, 5] = 1.0                        # height (unused for ground-plane overlap)
        t[:, 6] = -b[:, 4]                   # heading, opposite rotation convention
        return t.cuda().contiguous()

    overlap = torch.zeros((n, k), dtype=torch.float32, device="cuda")
    iou3d_nms_cuda.boxes_overlap_bev_gpu(to_7dof(boxes), to_7dof(query_boxes), overlap)
    inter = overlap.cpu().numpy().astype(np.float64)

    area_a = (boxes[:, 2] * boxes[:, 3])[:, None]
    area_b = (query_boxes[:, 2] * query_boxes[:, 3])[None, :]
    if criterion == -1:
        out = inter / (area_a + area_b - inter)
    elif criterion == 0:
        out = inter / area_a
    elif criterion == 1:
        out = inter / area_b
    else:
        out = inter
    return out.astype(boxes.dtype)


def check_rotate_iou():
    """Quick check that the replacement matches the original's geometry,
    including the rotation direction. Fails loudly instead of giving wrong AP."""
    a = np.array([[0.0, 0.0, 4.0, 2.0, 0.3]])
    assert abs(rotate_iou_torch(a, a)[0, 0] - 1.0) < 1e-4, "identical boxes should give IoU 1"
    turned = np.array([[0.0, 0.0, 4.0, 2.0, 0.3 + np.pi / 2]])
    assert abs(rotate_iou_torch(a, turned)[0, 0] - 1 / 3) < 1e-3, "90-degree turn should give IoU 1/3"
    # A small square placed along the long box's axis, using the original's
    # clockwise convention. With the wrong rotation direction it misses entirely.
    long_box = np.array([[0.0, 0.0, 4.0, 1.0, 0.5]])
    p = (1.5 * np.cos(0.5), -1.5 * np.sin(0.5))
    square = np.array([[p[0], p[1], 0.4, 0.4, 0.5]])
    assert abs(rotate_iou_torch(long_box, square, criterion=2)[0, 0] - 0.16) < 1e-3, \
        "rotation direction mismatch"


def install_rotate_iou_replacement():
    """Make OpenPCDet's evaluation import our function instead of the numba
    module (whose CUDA code crashes when it is merely imported)."""
    import sys
    import types
    name = "pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou"
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.rotate_iou_gpu_eval = rotate_iou_torch
        sys.modules[name] = module
        check_rotate_iou()


def evaluate(gt_annos, dt_annos):
    install_rotate_iou_replacement()
    from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
    text, ap = kitti_eval.get_official_eval_result(copy.deepcopy(gt_annos), copy.deepcopy(dt_annos), ["Car"])
    return text, ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="OpenPCDet result.pkl from test.py")
    ap.add_argument("--yolo", required=True, help="results/yolo_val.json from run_yolo.py")
    ap.add_argument("--infos", required=True, help="OpenPCDet data/kitti/kitti_infos_val.pkl")
    ap.add_argument("--iou", type=float, default=0.5, help="2D IoU needed to call it a match")
    ap.add_argument("--penalty", type=float, default=0.5, help="score multiplier for unconfirmed boxes")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dets = pickle.load(open(args.result, "rb"))
    infos = {info["image"]["image_idx"]: info for info in pickle.load(open(args.infos, "rb"))}
    yolo = json.load(open(args.yolo))["boxes"]
    gt_annos = [infos[d["frame_id"]]["annos"] for d in dets]

    fused, proj_errors, fuse_ms = [], [], []
    for d in dets:
        info = infos[d["frame_id"]]
        h, w = info["image"]["image_shape"]
        P2 = info["calib"]["P2"][:3]  # stored as 4x4; projection needs the top 3 rows
        t0 = time.perf_counter()
        new_scores, rects = fuse_frame(d, yolo.get(d["frame_id"], []), P2, (w, h), args.iou, args.penalty)
        fuse_ms.append((time.perf_counter() - t0) * 1000)
        nd = copy.deepcopy(d)
        nd["score"] = new_scores
        fused.append(nd)
        # Sanity check: our projection vs. the 2D box OpenPCDet computed itself
        for r, b in zip(rects, d["bbox"]):
            if r is not None:
                proj_errors.append(np.abs(r - b).mean())

    print(f"Projection check: mean |our 2D box - OpenPCDet's 2D box| = {np.mean(proj_errors):.2f} px "
          f"over {len(proj_errors)} boxes (should be well under 1-2 px)")

    base_text, base_ap = evaluate(gt_annos, dets)
    fused_text, fused_ap = evaluate(gt_annos, fused)
    base_fp = false_positives_by_distance(dets, gt_annos)
    fused_fp = false_positives_by_distance(fused, gt_annos)

    rows = [
        ("3D AP, Car easy (R40)", "Car_3d/easy_R40"),
        ("3D AP, Car moderate (R40)", "Car_3d/moderate_R40"),
        ("3D AP, Car hard (R40)", "Car_3d/hard_R40"),
        ("BEV AP, Car moderate (R40)", "Car_bev/moderate_R40"),
    ]
    print(f"\n{'':32s}{'LiDAR-only':>12s}{'Fused':>10s}")
    for label, key in rows:
        print(f"{label:32s}{base_ap[key]:12.2f}{fused_ap[key]:10.2f}")
    print(f"{'False positives, 0-30 m':32s}{base_fp[0]:12d}{fused_fp[0]:10d}")
    print(f"{'False positives, >30 m':32s}{base_fp[1]:12d}{fused_fp[1]:10d}")
    print(f"\nFusion step: {np.mean(fuse_ms):.2f} ms per frame (CPU, numpy)")

    summary = {
        "iou": args.iou, "penalty": args.penalty,
        "lidar_only": {k: base_ap[k] for _, k in rows},
        "fused": {k: fused_ap[k] for _, k in rows},
        "false_positives_lidar_only": {"0-30m": base_fp[0], ">30m": base_fp[1]},
        "false_positives_fused": {"0-30m": fused_fp[0], ">30m": fused_fp[1]},
        "fusion_ms_per_frame": round(float(np.mean(fuse_ms)), 3),
        "projection_check_px": round(float(np.mean(proj_errors)), 3),
    }
    tag = f"iou{args.iou}_pen{args.penalty}"
    with open(os.path.join(args.out, f"fusion_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, f"fusion_{tag}_full_report.txt"), "w") as f:
        f.write("LIDAR-ONLY\n" + base_text + "\n\nFUSED\n" + fused_text)
    print("Saved", os.path.join(args.out, f"fusion_{tag}.json"))


if __name__ == "__main__":
    main()