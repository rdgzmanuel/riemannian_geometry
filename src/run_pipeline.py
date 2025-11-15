#!/usr/bin/env python3
"""
Pipeline completa HDM05 → esqueletos → ventanas → Grassmann
Usa directamente las funciones internas del paquete hdm05_grassmann.
"""

# Importamos directamente lo que necesitamos
# DATA_DIR = Path("data")
# RAW_DIR = DATA_DIR / "HDM05"
# INTERIM_DIR = DATA_DIR / "interim" / "hdm05_cleaned"
# PROCESSED_DIR = DATA_DIR / "processed"
# HDM05_WINDOWS_DIR = PROCESSED_DIR / "hdm05_windows"
# HDM05_GRASSMANN_DIR = PROCESSED_DIR / "hdm05_grassmann"
# # Donde están los .c3d crudos
# HDM05_CUTS_C3D_DIR = RAW_DIR / "cuts"
from src.config.paths import (
    HDM05_CUTS_C3D_DIR,
    HDM05_GRASSMANN_DIR,
    HDM05_WINDOWS_DIR,
    INTERIM_DIR,
)
from src.data.grassman_repr import build_all_grassmann
from src.data.hdm05_loader import list_c3d_files
from src.data.preprocessing import preprocess_all
from src.data.windowing import build_all_windows


def main():
    print("===============================================")
    print("       PIPELINE COMPLETA HDM05 → GRASSMANN")
    print("===============================================")

    # ----------------------------------------------------------
    # 0. Verificar que hay .c3d disponibles
    # ----------------------------------------------------------
    print("\n🔍 Verificando datos RAW .c3d...")
    c3d_files = list_c3d_files(use_cuts=True, pattern="*.C3D")
    if len(c3d_files) == 0:
        print(f"❌ ERROR: No se encontraron .c3d en {HDM05_CUTS_C3D_DIR}")
        print("Ejecuta antes: scripts/download_hdm05.py")
        return
    print(f"✓ Se encontraron {len(c3d_files)} archivos .c3d")

    # ----------------------------------------------------------
    # 1. Preprocesado: raw → interim
    # ----------------------------------------------------------
    print("\n🏗  [1/3] Preprocesando esqueletos (raw → interim)...")
    preprocess_all(src_dir=HDM05_CUTS_C3D_DIR, dst_dir=INTERIM_DIR)
    print("✓ Preprocesado completado.")

    # ----------------------------------------------------------
    # 2. Ventanas: interim → windows
    # ----------------------------------------------------------
    print("\n🏗  [2/3] Generando ventanas (interim → windows)...")
    build_all_windows(src_dir=INTERIM_DIR, dst_dir=HDM05_WINDOWS_DIR)
    print("✓ Ventanas generadas.")

    # ----------------------------------------------------------
    # 3. Grassmann: windows → grassmann
    # ----------------------------------------------------------
    print("\n🏗  [3/3] Generando representación Grassmann...")
    build_all_grassmann(src_dir=HDM05_WINDOWS_DIR, dst_dir=HDM05_GRASSMANN_DIR)
    print("✓ Representaciones Grassmann generadas.")

    # ----------------------------------------------------------
    # FIN
    # ----------------------------------------------------------
    print("\n🎉 PIPELINE COMPLETADA CON ÉXITO")
    print("📁 Resultados:")
    print(f"  - Secuencias limpias: {INTERIM_DIR}")
    print(f"  - Ventanas:           {HDM05_WINDOWS_DIR}")
    print(f"  - Grassmann:          {HDM05_GRASSMANN_DIR}")
    print("🚀 Listo para entrenar GRNet o baselines.")


if __name__ == "__main__":
    main()
