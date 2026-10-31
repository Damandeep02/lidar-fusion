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


def evaluate(gt_annos, dt_annos):
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
