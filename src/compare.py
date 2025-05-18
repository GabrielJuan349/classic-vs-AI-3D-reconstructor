#!/usr/bin/env python
"""
compare.py
==========

Visual comparison between
  • the COLMAP-dense cloud        (classic)
  • the AI-fused depth cloud      (ai)

Two display modes
-----------------
overlay   – colour-code the two clouds in the *same* coordinate frame
side      – translate the AI cloud +X so the two appear side-by-side

The script runs completely head-less: it uses Open3D off-screen
rendering and saves a PNG snapshot that you can download or open locally.
"""

# ──────────────────────────────────────────────────────────── imports
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
import open3d as o3d


# ───────────────────────────────────────── utility: load cloud
def load_cloud(path: Path, color: tuple[float, float, float]) -> o3d.geometry.PointCloud:
    """Read PLY/OBJ/… and paint every point with the given colour (0-1)."""
    if not path.exists():
        sys.exit(f"[ERROR] File not found: {path}")
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        sys.exit(f"[ERROR] Empty cloud: {path}")
    pcd.paint_uniform_color(color)
    return pcd


# ───────────────────────────────────────── snapshot generator
def save_snapshot(geoms: list[o3d.geometry.Geometry],
                  out_png: Path,
                  width: int = 1600,
                  height: int = 900) -> None:
    """Head-less render of the given geometries → PNG."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    for g in geoms:
        vis.add_geometry(g)
    vis.poll_events(); vis.update_renderer()
    vis.capture_screen_image(str(out_png), do_render=True)
    vis.destroy_window()
    print(f"[Compare] Snapshot saved → {out_png}")


# ──────────────────────────────────────────────── main
def main():
    ap = argparse.ArgumentParser(description="Visualise classic vs AI clouds.")
    ap.add_argument("--classic", required=True, help="workspace/dense/fused.ply")
    ap.add_argument("--ai",      required=True, help="workspace/ai/fused_ai.ply")
    ap.add_argument("--out",     default="compare.png", help="Output PNG file")
    ap.add_argument("--mode",    choices=["overlay", "side"], default="side",
                    help="Visualisation layout")
    args = ap.parse_args()

    pcd_classic = load_cloud(Path(args.classic), (0.0, 0.6, 1.0))  # blue
    pcd_ai      = load_cloud(Path(args.ai),      (1.0, 0.3, 0.0))  # orange

    if args.mode == "side":
        # translate AI cloud along +X so both are visible separately
        bbox = pcd_classic.get_axis_aligned_bounding_box()
        dx = bbox.get_extent()[0] * 1.2          # 20 % gap
        pcd_ai.translate((dx, 0, 0))

    save_snapshot([pcd_classic, pcd_ai], Path(args.out))


# ───────────────────────────────────────── entry-point
if __name__ == "__main__":
    main()

