"""Draw one KITTI frame: LiDAR points projected into the camera image
(coloured by distance) and the ground-truth 3D car boxes drawn as wireframes.
Optionally also the detector's boxes.

If the points line up with the cars and the boxes sit on them, your
calibration and projection code is right. This is the check to run before
building anything on top of it.

    python visualize.py --kitti_root $SCRATCH/data/kitti --frames 000010 000025 000100 000200 000300
    python visualize.py --kitti_root ... --frames 000010 --dets PATH/TO/result.pkl
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kitti_utils import (box_corners_cam, cam_to_image, read_calib, read_labels,
                         read_velodyne, velo_to_cam)

# corner index pairs that form the 12 edges of a box (see box_corners_cam)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def draw_box(ax, location, dims_lhw, rotation_y, P2, color):
    corners = box_corners_cam(location, dims_lhw, rotation_y)
    if np.any(corners[:, 2] < 0.1):
        return
    uv, _ = cam_to_image(corners, P2)
    for a, b in EDGES:
        ax.plot([uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]], color=color, linewidth=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", required=True, help="folder that contains training/")
    ap.add_argument("--frames", nargs="+", default=["000010"])
    ap.add_argument("--dets", default=None, help="OpenPCDet result.pkl (val frames only)")
    ap.add_argument("--score", type=float, default=0.3, help="only draw detections above this score")
    ap.add_argument("--out", default="vis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dets = {}
    if args.dets:
        dets = {d["frame_id"]: d for d in pickle.load(open(args.dets, "rb"))}

    for frame in args.frames:
        base = os.path.join(args.kitti_root, "training")
        calib = read_calib(f"{base}/calib/{frame}.txt")
        image = plt.imread(f"{base}/image_2/{frame}.png")
        h, w = image.shape[:2]

        # LiDAR points -> camera -> image; keep points in front of the camera and inside the image
        points = read_velodyne(f"{base}/velodyne/{frame}.bin")
        cam = velo_to_cam(points[:, :3], calib)
        cam = cam[cam[:, 2] > 0.5]
        uv, depth = cam_to_image(cam, calib["P2"])
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        uv, depth = uv[inside], depth[inside]

        fig, ax = plt.subplots(figsize=(16, 5))
        ax.imshow(image)
        ax.scatter(uv[:, 0], uv[:, 1], c=depth, cmap="jet", s=0.5, vmin=0, vmax=60)

        for obj in read_labels(f"{base}/label_2/{frame}.txt"):
            if obj["name"] == "Car":
                draw_box(ax, obj["location"], obj["dimensions"], obj["rotation_y"], calib["P2"], "lime")

        if frame in dets:
            d = dets[frame]
            for i in range(len(d["score"])):
                if d["name"][i] == "Car" and d["score"][i] >= args.score:
                    draw_box(ax, d["location"][i], d["dimensions"][i], d["rotation_y"][i], calib["P2"], "red")

        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")
        title = f"frame {frame}: LiDAR points coloured by distance, ground-truth cars in green"
        ax.set_title(title + (", detections in red" if frame in dets else ""))
        path = os.path.join(args.out, f"{frame}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print("Saved", path, f"({inside.sum()} points in view)")


if __name__ == "__main__":
    main()
