#!/usr/bin/env python
"""
ai_depth.py
===========

Generate per-image depth (and optional object mask) using MiDaS / DPT
and, if requested, Segment-Anything.
Depth is written as **16-bit PNG** so every other tool (Open3D,
CloudCompare, Meshlab) can read it without extra code.

Directory layout created
------------------------
workspace/
└── ai/
    ├── depth/   *.png   # 16-bit depth maps (relative, in metres)
    └── masks/   *.png   # optional binary masks from SAM
"""

# ────────────────────────────────────────────────────────── imports
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm

# try Segment-Anything (only loaded if --sam_ckpt is given)
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    SamPredictor = None

# ────────────────────────────────────────────────────────── CLI
P = argparse.ArgumentParser()
P.add_argument("--images_dir", required=True,
               help="Folder containing input JPG / PNG images")
P.add_argument("--prefix", default="",
               help="Only process files whose name starts with this string")
P.add_argument("--work_dir", required=True,
               help="Workspace root (ai/depth will be created inside)")
P.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
               help="Inference device")
P.add_argument("--model_name", default="DPT_Large",
               help="Model ID in torch.hub isl-org/MiDaS")
P.add_argument("--resize", type=int, default=384,
               help="Resize long side to this many pixels before MiDaS")
P.add_argument("--sam_ckpt",
               help="Path to Segment-Anything *.pth (optional masking)")
args = P.parse_args()

# ───────────────────────────────────────── housekeeping
images_dir = Path(args.images_dir).expanduser().resolve()
work_dir   = Path(args.work_dir).expanduser().resolve()
depth_dir  = work_dir / "ai" / "depth"
mask_dir   = work_dir / "ai" / "masks"
depth_dir.mkdir(parents=True, exist_ok=True)
if args.sam_ckpt:
    mask_dir.mkdir(parents=True, exist_ok=True)

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ────────────────────────── load MiDaS / DPT
print(f"[INFO] Loading MiDaS model “{args.model_name}”…")
midas = torch.hub.load("isl-org/MiDaS", args.model_name,
                       trust_repo=True).to(device).eval()
midas_trans = torch.hub.load("isl-org/MiDaS", "transforms",
                             trust_repo=True).dpt_transform

# ────────────────────────── optional SAM predictor
sam_predictor: SamPredictor | None = None
if args.sam_ckpt:
    if SamPredictor is None:
        sys.exit("[ERROR] segment-anything not installed; "
                 "pip install git+https://github.com/facebookresearch/segment-anything.git")
    print("[INFO] Loading Segment-Anything …")
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt)
    sam.to(device).eval()
    sam_predictor = SamPredictor(sam)

# ────────────────────────── helper functions
def save_depth_png(depth: np.ndarray, path: Path) -> None:
    """Normalise depth to 0–65535 and save as 16-bit PNG."""
    d_min, d_max = np.percentile(depth, (1, 99))
    depth = np.clip((depth - d_min) / (d_max - d_min), 0, 1)
    depth16 = (depth * 65535).astype(np.uint16)
    cv.imwrite(str(path), depth16)

def image_list() -> list[Path]:
    return sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        and p.name.startswith(args.prefix))

# ────────────────────────── main loop
if not image_list():
    sys.exit("[ERROR] No matching images found.")

t0 = time.time()
for img_path in tqdm(image_list(), desc="Depth"):
    out_png = depth_dir / f"{img_path.stem}.png"
    if out_png.exists():
        continue

    bgr = cv.imread(str(img_path), cv.IMREAD_COLOR)
    if bgr is None:
        print(f"[WARN] Cannot read {img_path}")
        continue
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)

    # optional SAM mask
    mask: np.ndarray | None = None
    if sam_predictor:
        sam_predictor.set_image(rgb)
        masks, _, _ = sam_predictor.predict(point_coords=None,
                                            point_labels=None,
                                            multimask_output=True)
        if masks.size:
            mask = masks[np.argmax(masks.sum(axis=(1, 2)))]
            cv.imwrite(str(mask_dir / f"{img_path.stem}.png"),
                       (mask * 255).astype(np.uint8))

    # MiDaS inference (resize for speed)
    h, w = rgb.shape[:2]
    scale = args.resize / max(h, w)
    if scale < 1.0:
        rgb_small = cv.resize(rgb, (int(w*scale), int(h*scale)),
                              interpolation=cv.INTER_AREA)
    else:
        rgb_small = rgb

    tensor = midas_trans(rgb_small).to(device)

    # MiDaS hub (2025) already returns a tensor with batch-dim
    if tensor.ndim == 3:          # [3,H,W]  → add batch
        tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        pred = midas(tensor).squeeze().cpu().numpy()

    # upsample back to original size
    pred_full = cv.resize(pred, (w, h), interpolation=cv.INTER_CUBIC)

    if mask is not None:
        pred_full *= mask.astype(np.float32)

    save_depth_png(pred_full, out_png)

print(f"[INFO] Processed {len(image_list())} images "
      f"in {time.time()-t0:.1f}s → {depth_dir}")

