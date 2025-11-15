# src/hdm05_grassmann/config/paths.py
from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "HDM05"
INTERIM_DIR = DATA_DIR / "interim" / "hdm05_cleaned"
PROCESSED_DIR = DATA_DIR / "processed"

HDM05_WINDOWS_DIR = PROCESSED_DIR / "hdm05_windows"
HDM05_GRASSMANN_DIR = PROCESSED_DIR / "hdm05_grassmann"

# Donde están los .c3d crudos
HDM05_FULL_C3D_DIR = RAW_DIR / "full_takes"
HDM05_CUTS_C3D_DIR = RAW_DIR / "cuts"


# Helper
def ensure_dirs():
    for d in [
        DATA_DIR,
        RAW_DIR,
        INTERIM_DIR,
        HDM05_WINDOWS_DIR,
        HDM05_GRASSMANN_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
