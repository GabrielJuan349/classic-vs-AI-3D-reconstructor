#!/usr/bin/env python3
"""
viewer.py ­– lightweight visualisation helpers for the COLMAP + AI pipeline
----------------------------------------------------------------------------

Capabilities
------------
*  `sparse_to_pcd()`   → Open3D point cloud from a COLMAP *sparse/0* folder.
*  `ply_to_pcd()`      → Open3D point cloud from any *.ply*.
*  `show()`            → Interactive window *or* head-less PNG snapshot.

The module works both on laptops with a display and on head-less servers
(e.g. AWS EC2).  When no X-server is available we fall back to an OpenGL
off-screen context and save a **snapshot```viewer.png``` next to the script
so you can download / view it locally.

Dependencies: `open3d>=0.17`, `numpy`, `pycolmap` (only for sparse loader).
"""

from __future__ import annotations
from pathlib import Path
import os, sys, numpy as np, open3d as o3d

# ---------------------------------------------------------------------- #
#  POINT-CLOUD LOADERS
# ---------------------------------------------------------------------- #
def sparse_to_pcd(sparse_root: Path, colour=(1.0, 0.0, 0.0)
                  ) -> o3d.geometry.PointCloud:
    """
    Read COLMAP sparse model (points3D.txt) and return an Open3D cloud.

    Parameters
    ----------
    sparse_root : Path
        Folder that contains 0/points3D.txt  (or 1/, 2/ …).
    colour : tuple(float, float, float)
        RGB in [0,1] applied to the whole cloud.

    Returns
    -------
    o3d.geometry.PointCloud
    """
    import pycolmap                                           # local import
    pts_path = sparse_root / "0" / "points3D.txt"
    if not pts_path.exists():
        raise FileNotFoundError(f"{pts_path} not found")

    pts3d = pycolmap.read_points3D_text(pts_path)
    xyz   = np.asarray([p.xyz for p in pts3d.values()], dtype=np.float32)
    pcd   = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd.paint_uniform_color(colour)
    return pcd


def ply_to_pcd(ply_path: Path, colour=(0.0, 1.0, 0.0)
               ) -> o3d.geometry.PointCloud:
    """
    Load any *.ply file (dense or AI) and tint it.

    Raises
    ------
    RuntimeError if the PLY cannot be opened or is empty.
    """
    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(str(ply_path))

    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        raise RuntimeError(f"{ply_path} contains 0 points")

    pcd.paint_uniform_color(colour)
    return pcd


# ---------------------------------------------------------------------- #
#  VISUALISATION
# ---------------------------------------------------------------------- #
def _draw_gui(geoms: list[o3d.geometry.Geometry]) -> None:
    """Standard on-screen Open3D viewer."""
    o3d.visualization.draw_geometries(geoms)


def _draw_offscreen(geoms: list[o3d.geometry.Geometry],
                    out_png="viewer.png", w=1600, h=1200) -> None:
    """
    Off-screen renderer → PNG (head-less servers, CI, Jupyter).

    Relies on Open3D > 0.17’s EGL / OSMesa context.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=w, height=h)
    for g in geoms:
        vis.add_geometry(g)
    vis.get_render_option().background_color = np.asarray([1, 1, 1])
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    vis.poll_events(); vis.update_renderer()
    vis.capture_screen_image(out_png)
    vis.destroy_window()
    print(f"[Viewer] Saved off-screen snapshot to {out_png}")


def show(*geoms: o3d.geometry.Geometry, out_png="viewer.png") -> None:
    """
    Try interactive GUI first; if that fails (head-less) save a PNG.

    Parameters
    ----------
    geoms : Open3D geometries
    out_png : str
        File name for the snapshot when GUI is unavailable.
    """
    geoms = list(geoms)
    try:
        _draw_gui(geoms)
    except Exception as exc:
        # Most common head-less failure → fallback
        print(f"[Viewer] GUI unavailable ({exc.__class__.__name__}); "
              "rendering off-screen …", file=sys.stderr)
        _draw_offscreen(geoms, out_png)


# ---------------------------------------------------------------------- #
#  CONVENIENT CLI
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse, textwrap
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
        Quick visual check of reconstruction outputs.

        Examples
        --------
        # show sparse model (red) + dense classic (green)
        python viewer.py sparse/0 points classic/dense/fused.ply

        # save PNG only (no X server)
        QT_QPA_PLATFORM=offscreen python viewer.py sparse/0 points dense/fused.ply --png result.png
        """))
    ap.add_argument("paths", nargs="+",
                    help="Any number of sparse/<id> folders or *.ply files")
    ap.add_argument("--png", help="Head-less: save to this PNG, no window")
    args = ap.parse_args()

    clouds = []
    colours = [(1,0,0), (0,1,0), (0,0,1), (1,0,1), (0,1,1)]
    for idx, p in enumerate(args.paths):
        pth = Path(p)
        col = colours[idx % len(colours)]
        if pth.is_dir():
            clouds.append(sparse_to_pcd(pth, col))
        else:
            clouds.append(ply_to_pcd(pth, col))

    if args.png:
        show(*clouds, out_png=args.png)
    else:
        show(*clouds)

