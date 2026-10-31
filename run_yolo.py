"""Run a COCO-pretrained YOLO on every KITTI validation image and save the
2D car boxes. Runs in the `ml` environment (the GRPO one), on a GPU.

    python run_yolo.py --kitti_root $SCRATCH/data/kitti --split_file PATH/TO/ImageSets/val.txt

Output: results/yolo_val.json  {frame_id: [[x1, y1, x2, y2, confidence], ...]}
plus the average time per image, used in the latency table.
"""

import argparse
import json
import os
import time

from ultralytics import YOLO

COCO_CAR = 2  # class index of "car" in the COCO dataset YOLO was trained on


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", required=True)
    ap.add_argument("--split_file", required=True, help="OpenPCDet's data/kitti/ImageSets/val.txt")
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--conf", type=float, default=0.1,
                    help="keep low-confidence boxes too; fusion uses the confidence")
    ap.add_argument("--imgsz", type=int, default=1280, help="KITTI images are 1242 px wide")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="only the first N images (for latency runs)")
    ap.add_argument("--out", default="results/yolo_val.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    frames = [line.strip() for line in open(args.split_file) if line.strip()][: args.limit]
    paths = [os.path.join(args.kitti_root, "training", "image_2", f"{f}.png") for f in frames]
    model = YOLO(args.weights)

    boxes, speeds = {}, []
    t0 = time.time()
    for i in range(0, len(paths), args.batch):
        results = model.predict(paths[i : i + args.batch], classes=[COCO_CAR], conf=args.conf,
                                imgsz=args.imgsz, half=True, verbose=False)
        for frame, r in zip(frames[i : i + args.batch], results):
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            boxes[frame] = [[*map(float, b), float(c)] for b, c in zip(xyxy, conf)]
            speeds.append(r.speed)  # ms per image: preprocess, inference, postprocess
        print(f"{min(i + args.batch, len(paths))}/{len(paths)} images", flush=True)

    n = len(speeds)
    avg = {k: round(sum(s[k] for s in speeds) / n, 2) for k in speeds[0]}
    print("Average ms per image (batched):", avg)
    print(f"Total {time.time() - t0:.0f} s for {n} images")
    with open(args.out, "w") as f:
        json.dump({"boxes": boxes, "ms_per_image": avg, "weights": args.weights}, f)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
