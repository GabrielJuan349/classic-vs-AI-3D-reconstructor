#!/usr/bin/env python3
"""
reconstruct_phase1.py
---------------------
Prototip ràpid de reconstrucció 3D (Fase 1) basat exclusivament en fotogrametria clàssica:

1. Structure‑from‑Motion (SfM) + Multi‑View Stereo (MVS) amb colmap
2. Refinament opcional amb ICP
3. Reconstrucció de malla amb Poisson Surface Reconstruction
4. Exportació OBJ / STL / GLB

El programa rep:
    --images_dir  Carpeta d'entrada amb totes les fotos
    --prefix      Cadena comuna que han de tenir al nom les imatges vàlides
    --work_dir    Carpeta temporal (es crea si no existeix)
    --out         Fitxer de sortida (extensió .obj, .stl, .glb, etc.)

Aquesta versió és autònoma i permet iniciar la Fase 2 simplement afegint
un mòdul que insereixi profunditats i màscares abans del pas d'ICP o del meshing.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List
import shutil
import subprocess
import open3d as o3d
import os


# ---------------------------------------------------------------------- #
# Utils
# ---------------------------------------------------------------------- #
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_images_with_prefix(images_dir: Path, prefix: str) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    return sorted(
        [
            f
            for f in images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in exts and f.name.startswith(prefix)
        ]
    )


def copy_subset(src_paths: List[Path], dst_dir: Path) -> None:
    ensure_dir(dst_dir)
    for p in src_paths:
        shutil.copy2(p, dst_dir / p.name)


# ---------------------------------------------------------------------- #
# Fase 1 – SfM + MVS
# ---------------------------------------------------------------------- #



def run_colmap_pipeline(
    images_dir: Path,
    work_dir: Path,
    *,
    max_image_size: int = 2000,
    use_gpu: bool = True,
    only_sfm: bool = False,
) -> Path | None:
    """
    Windows‑friendly COLMAP CLI pipeline (no pycolmap, no CUDA required).

    Steps
    -----
    1. feature_extractor      -> database.db
    2. exhaustive_matcher     -> matches in DB
    3. mapper                 -> sparse/0,1,…  (at least one model)
    4. patch_match_stereo     -> depth maps (GPU if available)
    5. stereo_fusion          -> dense/fused.ply  (point cloud)

    Parameters
    ----------
    images_dir : Path
        Folder with input photos.
    work_dir   : Path
        Workspace created by the function.
    max_image_size : int
        Long edge down‑scale before Patch‑Match (quality vs. speed).
    use_gpu : bool
        False forces CPU via  --PatchMatchStereo.gpu_index=-1.

    Returns
    -------
    Path to fused.ply (dense cloud) inside  <work_dir>/dense/

    Raises
    ------
    RuntimeError if any COLMAP stage fails.
    """
    images_dir = images_dir.resolve()
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    db_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    dense_dir = work_dir / "dense"
    dense_dir.mkdir(exist_ok=True)

    # Locate colmap executable (installed via ZIP/Chocolatey)
    colmap = shutil.which("colmap")
    if colmap is None:
        raise RuntimeError(
            "COLMAP executable not found in PATH. "
            "Install COLMAP for Windows and add its folder to PATH."
        )

    def run(cmd: list[str], tag: str):
        print(f"▶ {tag}")
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["LIBGL_ALWAYS_SOFTWARE"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = ""        # CPU box
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,              # ← capture bytes, not str
        )
        for line in proc.stdout:
            sys.stdout.buffer.write(line)            # live print
        proc.wait()
        if proc.returncode:
            # decode loosely so we can still print the log
            print(proc.stdout.decode("utf-8", errors="replace"))
            raise RuntimeError(f"COLMAP step '{tag}' failed (exit {proc.returncode}).")



    # ---------- 1. FEATURE EXTRACTION ----------
    run(
        [
            colmap, "automatic_reconstructor",
            f"--workspace_path={work_dir}",
            f"--image_path={images_dir}",

            # --- general settings ---
            #"--quality",           "low",     # fewer levels = faster on CPU
            "--use_gpu",            "0",
            "--gpu_index",         "-1",      # stays on CPU even if CUDA libs appear
            "--camera_model",      "PINHOLE", # replaces old ImageReader.camera_model
            "--single_camera",     "1",       # same intrinsics for all images

            # --- dense settings ---
            "--dense",             "1",       # run Patch-Match
        ],
        "automatic_reconstructor",
    )

    fused_ply = work_dir / "dense" / "fused.ply"
    alt_clean = work_dir / "dense" / "0" / "clean.ply"
    dense0 = work_dir / "dense" / "0" 
    depth_dir = dense0 / "stereo" / "depth_maps"

    if not depth_dir.exists() or not any(depth_dir.iterdir()):
        print("ℹ️  No depth maps found after automatic – running PatchMatch + Fusion …")
        # ➊  Patch-Match stereo  (writes depth_maps + normal_maps + conf)
        subprocess.run([
            colmap, "patch_match_stereo",
            "--workspace_path",   str(dense0),
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.geom_consistency", "1",
            "--PatchMatchStereo.max_image_size", str(max_image_size),
            "--gpu_index", "-1"                       # CPU-only
        ], check=True)

        # ➋  Stereo Fusion  → dense/fused.ply
        fused_ply = work_dir / "dense" / "fused.ply"
        subprocess.run([
            colmap, "stereo_fusion",
            "--workspace_path",   str(dense0),
            "--workspace_format", "COLMAP",
            "--output_path",      str(fused_ply),
            "--StereoFusion.min_num_pixels", "5"
        ], check=True)
    
    # pick whichever file now exists
    if fused_ply.exists():
        final_cloud = fused_ply
    elif alt_clean.exists():
        final_cloud = alt_clean
    else:
        raise RuntimeError("No dense cloud produced by automatic_reconstructor.")

    print(f"✓ Dense cloud saved at {final_cloud}")
    return final_cloud
# ---------------------------------------------------------------------- #
# Fase 1 – ICP + Poisson Mesh
# ---------------------------------------------------------------------- #
def refine_icp(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Aplica ICP lleu (auto‑alineació) – placeholder per futurs subnúvols."""
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
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_duplicated_triangles()
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=150_000)
    return mesh


# ---------------------------------------------------------------------- #
# Main CLI
# ---------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser("Reconstrucció 3D – Fase 1 (MVS + ICP)")
    parser.add_argument("--images_dir", required=True, help="Directori amb totes les fotos")
    parser.add_argument("--prefix", required=True, help="Prefix comú (filtre de fitxers)")
    parser.add_argument("--work_dir", default="workspace", help="Carpeta temporal COLMAP")
    parser.add_argument("--out", default="model.obj", help="Fitxer de sortida 3D")
    parser.add_argument("--only_sfm", action="store_true", help="Run COLMAP only up to sparse model")
    parser.add_argument("--max_image_size", type=int, default=2000)
    parser.add_argument("--reset_workspace", type=bool, default=False)

    args = parser.parse_args()


    orig_dir = Path(args.images_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    ensure_dir(work_dir)

    if args.reset_workspace:
        if (work_dir / "dense" / "0").exists():
            shutil.rmtree(work_dir / "dense/0")
            shutil.rmtree(work_dir / "sparse/0")
            os.remove(work_dir/"database.db")

    # 0. Filtrar i copiar només les imatges que compleixen el prefix
    selected = list_images_with_prefix(orig_dir, args.prefix)
    print(f"🔍 S'han trobat {len(selected)} imatges amb prefix '{args.prefix}'")
    if len(selected) < 5:
        sys.exit("Calen almenys 5 imatges que compleixin el prefix.")
    imgs_tmp = work_dir / "images"
    copy_subset(selected, imgs_tmp)
    print(f"⚙️  S'han copiat {len(selected)} imatges al workspace.")

    # Fase 1: COLMAP
    print("▶️  Executant SfM + MVS (colmap)…")
    fused_ply = run_colmap_pipeline(
        imgs_tmp, work_dir,
        max_image_size=args.max_image_size,
        only_sfm=args.only_sfm)

    if(args.only_sfm):
        return

    print(f"✅ Núvol dens generat: {fused_ply}")
    # Fase 2: Open3D refinament + meshing
    print("▶️  Carregant núvol dens amb Open3D…")
    pcd = o3d.io.read_point_cloud(str(fused_ply))
    pcd = refine_icp(pcd)

    print("▶️  Reconstruint malla Poisson…")
    mesh = pointcloud_to_mesh(pcd)
    out_path = Path(args.out).resolve()
    o3d.io.write_triangle_mesh(str(out_path), mesh, write_ascii=False)
    print(f"🎉 Malla guardada a: {out_path}")


if __name__ == "__main__":
    main()
