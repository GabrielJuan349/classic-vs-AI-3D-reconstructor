"""
fuse_ai.py – Back-project AI depth + mask into world coordinates, fuse to one PLY.
Requires the COLMAP sparse model (cameras.txt, images.txt) produced by SfM.
"""

from pathlib import Path
import numpy as np
import open3d as o3d
import pycolmap


def backproject(depth, mask, K, T):
    ys, xs = np.where(mask)
    z = depth[ys, xs]
    # 3-D in camera coords
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    pts_cam = np.stack([x, y, z, np.ones_like(z)])
    pts_w = (T @ pts_cam)[:3].T
    return pts_w


def fuse_ai(sparse_dir: Path, ai_maps: Path, out_ply: Path):
    cams   = pycolmap.read_cameras_text(sparse_dir/"cameras.txt")
    images = pycolmap.read_images_text(sparse_dir/"images.txt")
    all_pts = []
    for img_id, img in images.items():
        stem = Path(img.name).stem
        depth_file = ai_maps/f"{stem}_depth.npy"
        mask_file  = ai_maps/f"{stem}_mask.npy"
        if not depth_file.exists():             # AI map missing → skip
            continue
        depth = np.load(depth_file)
        mask  = np.load(mask_file)
        K = cams[img.camera_id].calibration_matrix()
        T = img.inverse_projection_matrix()     # 4×4 world←cam
        all_pts.append(backproject(depth, mask, K, T))
    if not all_pts:
        raise RuntimeError("No AI depth maps found.")
    pts = np.vstack(all_pts)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    o3d.io.write_point_cloud(str(out_ply), pcd)
    print("AI point cloud written to", out_ply)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("sparse_dir")
    ap.add_argument("ai_maps")
    ap.add_argument("out_ply")
    fuse_ai(Path(ap.parse_args().sparse_dir),
            Path(ap.parse_args().ai_maps),
            Path(ap.parse_args().out_ply))

