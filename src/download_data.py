# src/download_data.py
# ================================================================
# STEP 1 — Download all 4 datasets using Kaggle API
# Run: python src/download_data.py
# ================================================================

import os
import subprocess
import sys
import zipfile

BASE_DATA_PATH = "data"

DATASETS = [
    {
        "name":    "Home Credit Default Risk",
        "cmd":     "kaggle competitions download -c home-credit-default-risk",
        "dest":    f"{BASE_DATA_PATH}/home_credit",
        "size":    "~2.7 GB",
        "check":   f"{BASE_DATA_PATH}/home_credit/application_train.csv",
        "note":    "Accept rules at: kaggle.com/c/home-credit-default-risk",
    },
    {
        "name":    "Give Me Some Credit",
        "cmd":     "kaggle competitions download -c GiveMeSomeCredit",
        "dest":    f"{BASE_DATA_PATH}/give_me_some_credit",
        "size":    "~25 MB",
        "check":   f"{BASE_DATA_PATH}/give_me_some_credit/cs-training.csv",
        "note":    "Accept rules at: kaggle.com/c/GiveMeSomeCredit",
    },
    {
        "name":    "Lending Club",
        "cmd":     "kaggle datasets download -d wordsforthewise/lending-club",
        "dest":    f"{BASE_DATA_PATH}/lending_club",
        "size":    "~2 GB",
        "check":   f"{BASE_DATA_PATH}/lending_club/loan.csv",
        "note":    "Download once to accept at: kaggle.com/datasets/wordsforthewise/lending-club",
    },
    {
        "name":    "PKDD Czech Financial",
        "cmd":     "kaggle datasets download -d ntnu-testimon/pkldd-99",
        "dest":    f"{BASE_DATA_PATH}/pkdd_czech",
        "size":    "~30 MB",
        "check":   f"{BASE_DATA_PATH}/pkdd_czech/loan.csv",
        "note":    "Download once at: kaggle.com/datasets/ntnu-testimon/pkldd-99",
    },
]


def check_kaggle_setup():
    """Check if kaggle.json API key is configured."""
    kaggle_path = os.path.expanduser("~/.kaggle/kaggle.json")
    win_path    = os.path.expandvars(r"%USERPROFILE%\.kaggle\kaggle.json")

    if os.path.exists(kaggle_path) or os.path.exists(win_path):
        print(" Kaggle API key found!")
        return True
    else:
        print(" Kaggle API key NOT found!")
        print("\nHow to fix:")
        print("  1. Go to kaggle.com → Profile → Settings → API → Create New Token")
        print("  2. A kaggle.json file will download")
        print("  3. On Windows: move it to C:\\Users\\YourName\\.kaggle\\kaggle.json")
        print("  4. On Mac/Linux: move it to ~/.kaggle/kaggle.json")
        print("  5. Run this script again")
        return False


def unzip_file(zip_path, dest):
    """Unzip a downloaded file."""
    print(f"     Unzipping {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest)
    os.remove(zip_path)
    print(f"    Unzipped and zip deleted")


def download_dataset(dataset):
    """Download and unzip one dataset."""
    name  = dataset["name"]
    cmd   = dataset["cmd"]
    dest  = dataset["dest"]
    check = dataset["check"]

    # Skip if already downloaded
    if os.path.exists(check):
        size = os.path.getsize(check) / 1e6
        print(f"    {name} already exists ({size:.0f} MB) — skipping")
        return True

    print(f"\n    Downloading {name} ({dataset['size']})...")
    print(f"      Note: {dataset['note']}")

    os.makedirs(dest, exist_ok=True)
    result = subprocess.run(
        f"{cmd} -p \"{dest}\"",
        shell=True, capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"   Download failed: {result.stderr[:200]}")
        return False

    print(f"   Downloaded! Unzipping...")

    # Unzip all .zip files in dest
    for fname in os.listdir(dest):
        if fname.endswith(".zip"):
            unzip_file(os.path.join(dest, fname), dest)

    # Verify
    if os.path.exists(check):
        size = os.path.getsize(check) / 1e6
        print(f"   {name} ready ({size:.0f} MB)")
        return True
    else:
        print(f"    Downloaded but check file not found: {check}")
        print(f"      Files in {dest}: {os.listdir(dest)}")
        return False


def verify_all():
    """Verify all required files exist."""
    required = {
        "Home Credit": [
            "data/home_credit/application_train.csv",
            "data/home_credit/bureau.csv",
            "data/home_credit/bureau_balance.csv",
            "data/home_credit/installments_payments.csv",
            "data/home_credit/POS_CASH_balance.csv",
            "data/home_credit/credit_card_balance.csv",
            "data/home_credit/previous_application.csv",
        ],
        "Give Me Some Credit": ["data/give_me_some_credit/cs-training.csv"],
        "Lending Club":        ["data/lending_club/loan.csv"],
        "PKDD Czech":          [
            "data/pkdd_czech/loan.csv",
            "data/pkdd_czech/trans.csv",
            "data/pkdd_czech/account.csv",
        ],
    }

    print("\n" + "="*55)
    print(" FILE VERIFICATION")
    print("="*55)
    all_ok     = True
    total_size = 0

    for dataset, files in required.items():
        print(f"\n {dataset}:")
        for fpath in files:
            if os.path.exists(fpath):
                size = os.path.getsize(fpath) / 1e6
                total_size += size
                print(f"     {os.path.basename(fpath):<45} {size:>8.1f} MB")
            else:
                print(f"     {os.path.basename(fpath):<45} MISSING")
                all_ok = False

    print(f"\n{'='*55}")
    print(f"Total data size: {total_size/1000:.2f} GB")
    print(" ALL FILES READY!" if all_ok else " Some files missing — check errors above")
    return all_ok


if __name__ == "__main__":
    print("="*55)
    print("ALTERNATIVE CREDIT SCORING — DATA DOWNLOAD")
    print("="*55)

    # Create folder structure
    for ds in DATASETS:
        os.makedirs(ds["dest"], exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Check Kaggle API
    if not check_kaggle_setup():
        sys.exit(1)

    # Download each dataset
    print("\n Starting downloads...")
    results = []
    for dataset in DATASETS:
        ok = download_dataset(dataset)
        results.append((dataset["name"], ok))

    # Final verification
    verify_all()
