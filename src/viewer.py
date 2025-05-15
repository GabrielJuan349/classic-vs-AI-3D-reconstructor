import open3d as o3d, numpy as np, pycolmap
from pathlib import Path

def sparse_to_pcd(sparse_root: Path, color):
    pts = pycolmap.read_points3D_text(sparse_root/"0"/"points3D.txt")
    xyz = np.array([p.xyz for p in pts.values()])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd.paint_uniform_color(color)
    return pcd

def ply_to_pcd(ply_path: Path, color):
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pcd.paint_uniform_color(color)
    return pcd

def show(*geoms):
    o3d.visualization.draw_geometries(list(geoms))

