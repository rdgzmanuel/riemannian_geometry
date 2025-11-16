import os
import shutil
import zipfile

import requests
from tqdm import tqdm

# =========================
# 1. DOWNLOAD FILE
# =========================


def download_file(url: str, out_path: str, overwrite: bool = False):
    if os.path.exists(out_path) and not overwrite:
        print(f" File already exists: {out_path}")
        return

    print(f" Downloading {url}")

    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
    except Exception:
        print(f"ERROR: {url} does NOT exist on the server")
        print(f"Creating empty zip to allow pipeline to continue: {out_path}")
        with zipfile.ZipFile(out_path, "w") as z:
            pass
        return

    total_size = int(r.headers.get("content-length", 0))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with (
        open(out_path, "wb") as f,
        tqdm(
            total=total_size, unit="B", unit_scale=True, desc=os.path.basename(out_path)
        ) as pbar,
    ):
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"Downloaded: {out_path}")


# =========================
# 2. UNZIP FILE
# =========================


def unzip_file(zip_path: str, extract_to: str):
    print(f" Extracting {zip_path} → {extract_to}")
    os.makedirs(extract_to, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_to)
    except zipfile.BadZipFile:
        print(f"WARNING: {zip_path} is not a valid zip (likely empty).")

    print("Extraction complete.")


# =========================
# 3. MOVE ALL .c3d TO FINAL STRUCTURE
# =========================


def move_all_c3d(from_dir: str, to_dir: str):
    os.makedirs(to_dir, exist_ok=True)
    count = 0

    for root, dirs, files in os.walk(from_dir):
        for f in files:
            if f.lower().endswith(".c3d"):
                src = os.path.join(root, f)
                dst = os.path.join(to_dir, f)
                shutil.move(src, dst)
                count += 1

    print(f"Moved {count} .c3d files → {to_dir}")


# =========================
# 4. MAIN
# =========================

if __name__ == "__main__":
    BASE = "data/HDM05"

    cuts_zip_url = "https://resources.mpi-inf.mpg.de/HDM05/cuts/HDM05_cut_c3d.zip"
    full_zip_url = "https://resources.mpi-inf.mpg.de/HDM05/05-03/HDM_05-03_c3d.zip"

    cuts_zip_path = os.path.join(BASE, "HDM05_cut_c3d.zip")
    full_zip_path = os.path.join(BASE, "HDM_05-03_c3d.zip")

    cuts_extract = os.path.join(BASE, "HDM05_cut_c3d")  # temp
    full_extract = os.path.join(BASE, "HDM_05-03_c3d")  # temp

    final_cuts = os.path.join(BASE, "cuts")
    final_full = os.path.join(BASE, "full_takes")

    # ---- DOWNLOAD ----
    download_file(cuts_zip_url, cuts_zip_path)
    download_file(full_zip_url, full_zip_path)

    # ---- UNZIP ----
    unzip_file(cuts_zip_path, cuts_extract)
    unzip_file(full_zip_path, full_extract)

    # ---- MOVE FILES ----
    move_all_c3d(cuts_extract, final_cuts)
    move_all_c3d(full_extract, final_full)

    # ---- CLEANUP ----
    print("Cleaning up…")
    shutil.rmtree(cuts_extract, ignore_errors=True)
    shutil.rmtree(full_extract, ignore_errors=True)

    os.remove(cuts_zip_path)
    os.remove(full_zip_path)

    print("DONE!")
