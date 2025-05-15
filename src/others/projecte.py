#!/usr/bin/env python3
"""
reconstruct_phase1.py
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
def run_colmap_pipeline(images_dir: Path, work_dir: Path) -> Path:
    """
    Executa SfM incremental + densificació amb pycolmap.
    Retorna el camí al núvol dens `fused.ply`.
    """
    db_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    dense_dir = work_dir / "dense"

    # 1. Crear base de dades i importar les imatges
    pycolmap.database_create(db_path)
    pycolmap.import_images(db_path, images_dir, camera_model="PINHOLE")

    # 2. Extracció i matching de features
    pycolmap.feature_extractor(db_path)
    pycolmap.match_extractor(db_path)

    # 3. SfM incremental -> núvol espars + càmeres
    pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=images_dir,
        output_path=sparse_dir,
        camera_model="PINHOLE",
    )

    # 4. MVS: estèreo i fusió
    pycolmap.stereo_fusion(
        model_path=sparse_dir,
        output_path=dense_dir,
        image_path=images_dir,
        max_image_size=2000,  # redueix si cal memòria
    )

    fused_ply = dense_dir / "fused.ply"
    if not fused_ply.exists():
        sys.exit("Error: No s'ha generat fused.ply; comprova la sortida de COLMAP.")
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
