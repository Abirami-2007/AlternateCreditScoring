# run_pipeline.py
# ================================================================
# MASTER SCRIPT — runs the full pipeline in order
# Run: python run_pipeline.py
#
# Steps:
#   1. Download datasets
#   2. Feature engineering (all 4 datasets)
#   3. Merge datasets
#   4. EDA
#   5. Preprocessing + feature selection
# ================================================================

import os
import sys
import time

def run_step(name, script):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    start = time.time()
    ret = os.system(f"python {script}")
    elapsed = time.time() - start
    if ret == 0:
        print(f" {name} completed in {elapsed:.0f}s")
        return True
    else:
        print(f" {name} FAILED (exit code {ret})")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("ALTERNATIVE CREDIT SCORING — FULL PIPELINE")
    print("=" * 60)

    steps = [
        ("Data Download",          "src/download_data.py"),
        ("Feature Engineering",    "src/feature_engineering.py"),
        ("EDA",                    "src/eda.py"),
        ("Preprocessing",          "src/preprocessing.py"),
    ]

    results = []
    for name, script in steps:
        ok = run_step(name, script)
        results.append((name, ok))
        if not ok:
            print(f"\n Pipeline stopped at: {name}")
            print("Fix the error above and re-run this script.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETE")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'C' if ok else 'W'}  {name}")
    print(f"\n Outputs saved to: outputs/")
    print(f"   Merged dataset:    outputs/unified_credit_dataset.csv")
    print(f"   Final features:    outputs/final_80_features.csv")
    print(f"   EDA plots:         outputs/eda_plots/")
    print(f"   EDA summary:       outputs/eda_summary.txt")
    print(f"\n   Next: python src/model_training.py")
