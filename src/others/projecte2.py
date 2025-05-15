#!/usr/bin/env python3
"""
---------------------
Prototip ràpid de reconstrucció 3D (Fase 1) basat exclusivament en fotogrametria clàssica:

1. Structure‑from‑Motion (SfM) + Multi‑View Stereo (MVS) amb pycolmap
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

import open3d as o3d
import pycolmap


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
from pathlib import Path
import shutil
import subprocess
import pycolmap


def run_colmap_pipeline(
    images_dir: Path,
    work_dir: Path,
    max_image_size: int = 2000,
) -> Path:
    """
    Reconstrueix un núvol de punts dens amb COLMAP utilitzant:
      1. API Python (pycolmap)   → SfM esparsa
      2. CLI COLMAP (subprocess) → Patch‑Match Stereo (depth‑maps)
      3. API Python (pycolmap)   → Stereo Fusion (fused.ply)

    Retorna el path a dense/fused.ply; llança RuntimeError si alguna etapa falla.
    """
    work_dir = work_dir.resolve()
    db_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    dense_dir = work_dir / "dense"
    work_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. Importar imatges (crea database.db si no existeix) ----------
    ir_opts = pycolmap.ImageReaderOptions()
    ir_opts.camera_model = "PINHOLE"                     # un model senzill
    pycolmap.import_images(
        database_path=str(db_path),
        image_path=str(images_dir),
        camera_mode=pycolmap.CameraMode.SINGLE,
        options=ir_opts,
    )

    # ---------- 2. Extracció SIFT i matching exhaustiu ------------------------
    pycolmap.extract_features(str(db_path), str(images_dir))
    pycolmap.match_exhaustive(str(db_path))

    # ---------- 3. SfM incremental (reconstrucció esparsa) --------------------
    recon = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(images_dir),
        output_path=str(sparse_dir),
    )
    if not recon:
        raise RuntimeError(
            "SfM no ha trobat cap parell inicial. Revisa solapament i qualitat de les fotos."
        )

    # ---------- 4. Patch‑Match Stereo (profunditats) via CLI ------------------
    colmap_bin = shutil.which("colmap")
    if colmap_bin is None:
        raise RuntimeError("No s'ha trobat l'executable COLMAP al PATH.")

    subprocess.run(
        [
            colmap_bin,
            "patch_match_stereo",
            "--workspace_path", str(work_dir),
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.max_image_size", str(max_image_size),
        ],
        check=True,
    )

    # ---------- 5. Stereo Fusion (núvol dens) amb pycolmap --------------------
    fusion_opts = pycolmap.StereoFusionOptions()
    fusion_opts.max_image_size = max_image_size

    dense_dir.mkdir(exist_ok=True)
    fused_ply = dense_dir / "fused.ply"

    pycolmap.stereo_fusion(
        output_path=str(fused_ply),
        workspace_path=str(work_dir),
        workspace_format="COLMAP",
        options=fusion_opts,
    )
    if not fused_ply.exists():
        raise RuntimeError("StereoFusion no ha generat fused.ply")

    return fused_ply

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
    args = parser.parse_args()

    orig_dir = Path(args.images_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    ensure_dir(work_dir)

    # 0. Filtrar i copiar només les imatges que compleixen el prefix
    selected = list_images_with_prefix(orig_dir, args.prefix)
    print(f"🔍 S'han trobat {len(selected)} imatges amb prefix '{args.prefix}'")
    if len(selected) < 5:
        sys.exit("Calen almenys 5 imatges que compleixin el prefix.")
    imgs_tmp = work_dir / "images"
    copy_subset(selected, imgs_tmp)
    print(f"⚙️  S'han copiat {len(selected)} imatges al workspace.")

    # Fase 1: COLMAP
    print("▶️  Executant SfM + MVS (pycolmap)…")
    fused_ply = run_colmap_pipeline(imgs_tmp, work_dir)
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
