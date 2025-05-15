#!/usr/bin/env python3
"""
reconstruct_phase1.py
---------------------

Minimal, working COLMAP pipeline for pycolmap 3.11.x + COLMAP CLI.

Steps
1.  SfM          – pycolmap (import → SIFT + exhaustive matching → incremental mapping)
2.  Depth maps   – COLMAP CLI `patch_match_stereo`
3.  StereoFusion – pycolmap  (dense/fused.ply)
4.  ICP + Poisson mesh – Open3D
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import open3d as o3d
import pycolmap


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_images_with_prefix(images_dir: Path, prefix: str) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    return sorted(
        [f for f in images_dir.iterdir() if f.suffix.lower() in exts and f.name.startswith(prefix)]
    )


def copy_subset(src_paths: List[Path], dst_dir: Path) -> None:
    ensure_dir(dst_dir)
    for src in src_paths:
        shutil.copy2(src, dst_dir / src.name)


# -------------------------------------------------------------------
# COLMAP pipeline (pycolmap + CLI)
# -------------------------------------------------------------------
def run_colmap_pipeline(
    images_dir: Path,
    work_dir: Path,
    max_image_size: int = 2000,
) -> Path:
    """
    Sparse via pycolmap  → patch‑match via CLI → fusion via pycolmap.
    Returns dense/fused.ply Path.
    """
    work_dir = work_dir.resolve()
    db_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    dense_dir = work_dir / "dense"
    ensure_dir(work_dir)

    # -- 1  IMPORT IMAGES ------------------------------------------------------
    ir_opts = pycolmap.ImageReaderOptions()
    ir_opts.camera_model = "PINHOLE"
    pycolmap.import_images(
        database_path=str(db_path),
        image_path=str(images_dir),
        camera_mode=pycolmap.CameraMode.SINGLE,
        options=ir_opts,
    )

    # -- 2  FEATURES + MATCHING -----------------------------------------------
    pycolmap.extract_features(str(db_path), str(images_dir))
    pycolmap.match_exhaustive(str(db_path))

    # -- 3  INCREMENTAL MAPPING -----------------------------------------------
    recon = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(images_dir),
        output_path=str(sparse_dir),
    )
    if not recon:
        raise RuntimeError(
            "SfM failed: no valid reconstruction. Ensure ≥60 % overlap and good texture."
        )

    # -- 4  PATCH‑MATCH STEREO  (CLI) -----------------------------------------
    colmap_bin = shutil.which("colmap")
    if colmap_bin is None:
        raise RuntimeError("COLMAP executable not found in PATH.")

    subprocess.run(
        [
            colmap_bin,
            "patch_match_stereo",
            "--workspace_path",
            str(work_dir),
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.max_image_size",
            str(max_image_size),
        ],
        check=True,
    )

    # -- 5  STEREO FUSION ------------------------------------------------------
    fusion_opts = pycolmap.StereoFusionOptions()
    fusion_opts.max_image_size = max_image_size
    ensure_dir(dense_dir)
    fused_ply = dense_dir / "fused.ply"

    pycolmap.stereo_fusion(
        output_path=str(fused_ply),
        workspace_path=str(work_dir),
        workspace_format="COLMAP",
        options=fusion_opts,
    )
    if not fused_ply.exists():
        raise RuntimeError("StereoFusion failed: fused.ply not created.")

    return fused_ply


# -------------------------------------------------------------------
# Mesh utilities
# -------------------------------------------------------------------
def refine_icp(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pcd_down = pcd.voxel_down_sample(0.003)
    pcd_down.estimate_normals()
    reg = o3d.pipelines.registration.registration_icp(
        pcd_down,
        pcd_down,
        max_correspondence_distance=0.02,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    pcd.transform(reg.transformation)
    return pcd


def pointcloud_to_mesh(pcd: o3d.geometry.PointCloud, depth: int = 9) -> o3d.geometry.TriangleMesh:
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=150_000)
    return mesh


# -------------------------------------------------------------------
# Main CLI
# -------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser("Simple COLMAP reconstruction (CPU/GPU)")
    ap.add_argument("--images_dir", required=True, help="Folder with photos")
    ap.add_argument("--prefix", required=True, help="Common filename prefix to select photos")
    ap.add_argument("--work_dir", default="workspace", help="Working directory")
    ap.add_argument("--out", default="model.obj", help="Output mesh file (OBJ/STL/GLB)")
    ap.add_argument("--max_image_size", type=int, default=2000, help="Max resolution for MVS")
    args = ap.parse_args()

    orig_dir = Path(args.images_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    images = list_images_with_prefix(orig_dir, args.prefix)
    if len(images) < 6:
        sys.exit("Need at least 6 photos with the given prefix.")
    imgs_tmp = work_dir / "images"
    copy_subset(images, imgs_tmp)
    print(f"Copied {len(images)} images.")

    print("▶️  Executing SfM + MVS (COLMAP)…")
    fused_ply = run_colmap_pipeline(imgs_tmp, work_dir, args.max_image_size)
    print("➡  dense/fused.ply created")

    pcd = o3d.io.read_point_cloud(str(fused_ply))
    pcd = refine_icp(pcd)
    mesh = pointcloud_to_mesh(pcd)

    out_path = Path(args.out).resolve()
    o3d.io.write_triangle_mesh(str(out_path), mesh, write_ascii=False)
    print("✅  Mesh saved to", out_path)


if __name__ == "__main__":
    main()

