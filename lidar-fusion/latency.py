"""Per-stage latency of PointPillars, in FP32 and FP16, one frame at a time
(batch size 1, like a robot processing frames as they arrive).
Runs in the `lidar` environment, on a GPU node.

    python latency.py --openpcdet $SCRATCH/projects/OpenPCDet \
        --ckpt $SCRATCH/projects/OpenPCDet/checkpoints/pointpillar_7728.pth

Why torch.cuda.synchronize(): GPU work is asynchronous. Python returns from
a CUDA call before the GPU has finished, so without synchronizing you'd time
how long it took to *queue* the work, not to do it.
"""

import argparse
import json
import os
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpcdet", required=True, help="path to the OpenPCDet repo")
    ap.add_argument("--cfg", default="cfgs/kitti_models/pointpillar.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20, help="untimed frames first (GPU warm-up)")
    ap.add_argument("--out", default=os.path.abspath("results/latency_pointpillars.json"))
    args = ap.parse_args()
    args.ckpt = os.path.abspath(args.ckpt)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # OpenPCDet's config files use paths relative to its tools/ folder
    os.chdir(os.path.join(args.openpcdet, "tools"))
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils

    cfg_from_yaml_file(args.cfg, cfg)
    logger = common_utils.create_logger()
    dataset, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                                     batch_size=1, dist=False, workers=0, logger=logger, training=False)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda().eval()
    stage_names = [type(m).__name__ for m in model.module_list]

    def run_frame(i, fp16):
        ms = {}
        t = time.perf_counter()
        batch = dataset.collate_batch([dataset[i]])  # read the .bin file, crop, voxelize into pillars (CPU)
        ms["load + pillarize (CPU)"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        load_data_to_gpu(batch)
        torch.cuda.synchronize()
        ms["copy to GPU"] = (time.perf_counter() - t) * 1000

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            for name, module in zip(stage_names, model.module_list):
                t = time.perf_counter()
                batch = module(batch)
                torch.cuda.synchronize()
                ms[name] = (time.perf_counter() - t) * 1000

        # NMS runs in FP32 (its CUDA kernel expects float32)
        for key in ("batch_cls_preds", "batch_box_preds"):
            batch[key] = batch[key].float()
        t = time.perf_counter()
        with torch.no_grad():
            model.post_processing(batch)
        torch.cuda.synchronize()
        ms["post-processing (NMS)"] = (time.perf_counter() - t) * 1000
        return ms

    results = {}
    for fp16 in (False, True):
        label = "FP16" if fp16 else "FP32"
        for i in range(args.warmup):
            run_frame(i, fp16)
        timings = [run_frame(args.warmup + i, fp16) for i in range(args.frames)]
        mean = {k: float(np.mean([t[k] for t in timings])) for k in timings[0]}
        gpu_keys = [k for k in mean if k not in ("load + pillarize (CPU)",)]
        mean["total"] = float(sum(mean.values()))
        mean["total GPU part"] = float(sum(mean[k] for k in gpu_keys))
        results[label] = {k: round(v, 2) for k, v in mean.items()}

    print(f"\nPointPillars latency, ms per frame (mean of {args.frames} frames, {torch.cuda.get_device_name(0)})")
    print(f"{'stage':30s}{'FP32':>10s}{'FP16':>10s}")
    for k in results["FP32"]:
        print(f"{k:30s}{results['FP32'][k]:10.2f}{results['FP16'][k]:10.2f}")
    results["gpu"] = torch.cuda.get_device_name(0)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
