#!/usr/bin/env python3
"""
fuse_ai.py
==========

Back‐project AI‐estimated depth maps into one point‐cloud and (optionally)
run Poisson surface reconstruction to produce an OBJ mesh — in a single script,
without pycolmap.

Usage:
  python src/fuse_ai.py \
      <sparse_dir> <ai_dir> <out_ply> \
      [--voxel 0.005] [--mesh_out workspace/ai/fused_ai.obj]

Examples:
  python src/fuse_ai.py workspace/sparse workspace/ai/depth \
        workspace/ai/fused_ai.ply --voxel 0.005 \
        --mesh_out workspace/ai/fused_ai.obj
"""

from __future__ import annotations
from pathlib import Path
import argparse, subprocess, tempfile, shutil, sys, re
import numpy as np
import open3d as o3d
import cv2 as cv


def qvec2rot(qw, qx, qy, qz) -> np.ndarray:
    """Quaternion → 3×3 rotation matrix (COLMAP order)."""
    return np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]
    ], dtype=np.float64)


def parse_cameras(txt: Path) -> dict[int, np.ndarray]:
    cams = {}
    for ln in txt.read_text().splitlines():
        if ln.startswith("#") or not ln.strip(): continue
        toks = ln.split()
        cid, model = int(toks[0]), toks[1]
        vals = list(map(float, toks[4:]))
        if model in ("SIMPLE_PINHOLE","SIMPLE_RADIAL"):
            fx=fy=vals[0]; cx,cy=vals[1:3]
        elif model in ("PINHOLE","OPENCV","OPENCV_FISHEYE"):
            fx,fy,cx,cy=vals[:4]
        else:
            raise RuntimeError(f"Unsupported camera model {model}")
        cams[cid] = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
    return cams


def parse_images(txt: Path) -> list[dict]:
    imgs, lines = [], txt.read_text().splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#") or not ln.strip():
            i += 1; continue
        t = ln.split()
        imgs.append(dict(
            qw=float(t[1]), qx=float(t[2]), qy=float(t[3]), qz=float(t[4]),
            tx=float(t[5]), ty=float(t[6]), tz=float(t[7]),
            cam_id=int(t[8]), name=t[9]))
        i += 2
    return imgs


def backproject(depth16: np.ndarray, mask: np.ndarray,
                K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask & (depth16>0))
    if xs.size == 0:
        return np.empty((0,3), np.float32)
    z = depth16[ys, xs].astype(np.float32)/65535.0
    x = (xs - K[0,2])*z / K[0,0]
    y = (ys - K[1,2])*z / K[1,1]
    pts_cam = np.vstack((x,y,z))
    pts_w = (R @ pts_cam) + t[:,None]
    return pts_w.T.astype(np.float32)


def fuse_ai(sparse_dir: Path, ai_dir: Path, out_ply: Path,
            voxel: float|None, mesh_out: Path|None) -> Path:

    # 1) find COLMAP sparse model folder
    if (sparse_dir/"cameras.txt").exists():
        mdl = sparse_dir
    else:
        subs = sorted(d for d in sparse_dir.iterdir()
                      if d.is_dir() and re.fullmatch(r"\d+", d.name))
        if not subs:
            sys.exit(f"[fuse_ai] No sparse model in {sparse_dir}")
        mdl = subs[0]

    # 2) ensure TXT (else convert BIN→TXT)
    if not (mdl/"cameras.txt").exists():
        if not (mdl/"cameras.bin").exists():
            sys.exit("[fuse_ai] Missing cameras.txt|bin")
        print("[fuse_ai] Converting BIN→TXT …")
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([
                "colmap", "model_converter",
                "--input_path", str(mdl),
                "--output_path", tmp,
                "--output_type", "TXT"
            ], check=True)
            for fn in ("cameras.txt","images.txt"):
                shutil.move(Path(tmp)/fn, mdl/fn)

    cams   = parse_cameras(mdl/"cameras.txt")
    images = parse_images (mdl/"images.txt")

    # 3) locate depth & mask dirs
    depth_dir = (ai_dir/"depth") if (ai_dir/"depth").is_dir() else ai_dir
    mask_dir  = (ai_dir/"masks") if (ai_dir/"masks").is_dir() else ai_dir

    all_pts, skipped = [], 0
    for img in images:
        stem = Path(img["name"]).stem

        # depth candidates
        dc = [depth_dir/f"{stem}_depth.png",
              depth_dir/f"{stem}.png",
              depth_dir/f"{stem}_depth.npy",
              depth_dir/f"{stem}.npy"]
        d_file = next((p for p in dc if p.exists()), None)
        if d_file is None:
            skipped += 1; continue

        # load depth into uint16
        if d_file.suffix == ".png":
            depth16 = cv.imread(str(d_file), cv.IMREAD_UNCHANGED)
            if depth16 is None:
                skipped +=1; continue
            if depth16.dtype != np.uint16:
                depth16 = depth16.astype(np.uint16)
        else:
            arr = np.load(d_file).astype(np.float32)
            depth16 = (np.clip(arr,0,1)*65535).astype(np.uint16)

        # mask candidate
        mc = [mask_dir/f"{stem}_mask.png", mask_dir/f"{stem}.png"]
        m_file = next((p for p in mc if p.exists()), None)
        mask = (cv.imread(str(m_file),cv.IMREAD_GRAYSCALE)>127) if m_file else np.ones_like(depth16, bool)

        K = cams[img["cam_id"]]
        R = qvec2rot(img["qw"], img["qx"], img["qy"], img["qz"])
        t = np.array([img["tx"], img["ty"], img["tz"]], dtype=np.float64)

        all_pts.append(backproject(depth16, mask, K, R, t))

    if not all_pts:
        sys.exit("[fuse_ai] No valid depth maps — check your ai_dir")

    cloud = np.vstack(all_pts)
    pcd   = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud))
    if voxel:
        pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals()

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_ply), pcd, write_ascii=False)
    print(f"[fuse_ai] Saved {len(pcd.points):,} pts → {out_ply}")
    if skipped:
        print(f"[fuse_ai] Skipped {skipped} images.")

    # 4) optional Poisson mesh → OBJ
    if mesh_out:
        print("[fuse_ai] Running Poisson surface reconstruction …")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8)
        dens = np.asarray(densities)
        thresh = np.quantile(dens, 0.01)
        vidx = np.where(dens > thresh)[0]
        mesh = mesh.select_by_index(vidx)
        mesh_out.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(mesh_out), mesh, write_ascii=False)
        print(f"[fuse_ai] Saved mesh → {mesh_out}")

    return out_ply


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Fuse AI depth-maps into PLY + optional OBJ mesh")
    p.add_argument("sparse_dir",
                   help="workspace/sparse or workspace/sparse/0")
    p.add_argument("ai_dir",
                   help="workspace/ai or workspace/ai/depth")
    p.add_argument("out_ply",
                   help="output fused point-cloud (.ply)")
    p.add_argument("--voxel", type=float,
                   help="voxel size for downsampling")
    p.add_argument("--mesh_out", type=Path,
                   help="write OBJ mesh here (e.g. out.obj)")
    args = p.parse_args()
    fuse_ai(Path(args.sparse_dir),
            Path(args.ai_dir),
            Path(args.out_ply),
            args.voxel,
            args.mesh_out)

