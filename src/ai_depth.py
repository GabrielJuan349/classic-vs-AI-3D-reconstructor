"""
ai_depth.py  – Generate depth-map + mask per image using MiDaS + Segment Anything.
Outputs two .npy files per photo:
    <stem>_depth.npy  float32  (H×W)
    <stem>_mask.npy   bool     (H×W)
"""

from pathlib import Path
import cv2
import numpy as np, torch
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def load_models(device="cuda"):
    # MiDaS Hybrid via torch.hub (downloads weights on first call)
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid").to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    # Segment Anything ViT-T
    sam = sam_model_registry["vit_t"]("/models/sam_vit_t.pth").to(device)
    mask_gen = SamAutomaticMaskGenerator(sam)
    return midas, transforms, mask_gen


@torch.no_grad()
def run_ai_depth(image_dir: Path, out_dir: Path, device="cuda"):
    out_dir.mkdir(parents=True, exist_ok=True)
    midas, transforms, mask_gen = load_models(device)
    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            continue
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        depth_tensor = midas(transforms(rgb).to(device))
        depth = torch.nn.functional.interpolate(
            depth_tensor.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        mask = mask_gen(rgb)[0]["segmentation"].astype(bool)

        stem = img_path.stem
        np.save(out_dir / f"{stem}_depth.npy", depth)
        np.save(out_dir / f"{stem}_mask.npy",  mask)
        print("AI depth done →", stem)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir")
    ap.add_argument("out_dir")
    run_ai_depth(Path(ap.parse_args().image_dir), Path(ap.parse_args().out_dir))

