"""
compare.py – Run classic COLMAP dense and AI-depth dense, then visualise
both sparse and dense results in colour overlay.

Usage
-----
python compare.py \
       --images photos/ --prefix obj_ \
       --work workspace \
       --view_sparse --view_dense
"""

import argparse, logging, shutil
from pathlib import Path
import subprocess, sys

from viewer import sparse_to_pcd, ply_to_pcd, show
from reconstruct_classic import run_colmap_pipeline, pointcloud_to_mesh
from ai_depth import run_ai_depth
from fuse_ai import fuse_ai

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


def main(args):
    work    = Path(args.work).resolve()
    img_tmp = work / "images"
    sparse  = work / "sparse"
    classic = work / "classic"
    ai_out  = work / "ai"

    # reset
    if args.reset_workspace and work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    # copy selected photos
    photos = sorted(Path(args.images).glob(f"{args.prefix}*"))
    img_tmp.mkdir(parents=True, exist_ok=True)
    for p in photos:
        shutil.copy2(p, img_tmp/p.name)

    # --- sparse SfM (shared) ---
    logging.info("Running sparse SfM …")
    run_colmap_pipeline(img_tmp, work, max_image_size=args.max_size, only_sfm=True)

    if args.view_sparse:
        show(
            sparse_to_pcd(sparse, (1,0,0)),      # classic colour red
        )

    # --- classic dense ---
    logging.info("Running classic Patch-Match …")
    fused_cl = run_colmap_pipeline(img_tmp, work, max_image_size=args.max_size)  # gets dense too
    (classic/"dense").mkdir(parents=True, exist_ok=True)
    shutil.copy2(fused_cl, classic/"dense/fused.ply")

    # --- AI branch ---
    logging.info("Running AI depth …")
    ai_maps = ai_out / "maps"
    run_ai_depth(img_tmp, ai_maps)
    fused_ai = ai_out / "fused_ai.ply"
    fuse_ai(sparse, ai_maps, fused_ai)

    # --- Visualise dense ---
    if args.view_dense:
        show(
            ply_to_pcd(fused_cl, (1,0,0)),    # red
            ply_to_pcd(fused_ai, (0,1,0))     # green
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--work",   default="workspace")
    ap.add_argument("--max_size", type=int, default=1600)
    ap.add_argument("--reset_workspace", action="store_true")
    ap.add_argument("--view_sparse", action="store_true")
    ap.add_argument("--view_dense",  action="store_true")
    main(ap.parse_args())

