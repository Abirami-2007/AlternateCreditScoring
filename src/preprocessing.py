# src/preprocessing.py
# ================================================================
# STEP 4 — Preprocessing + Feature Selection
# Run: python src/preprocessing.py
# Output: outputs/final_80_features.csv
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATHS, ALL_FEATURES, FEATURE_GROUPS
from sklearn.preprocessing  import RobustScaler
from sklearn.impute          import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif


def load_merged():
    print("Loading merged dataset...")
    df = pd.read_csv(PATHS["output_merged"], low_memory=False)
    print(f"  Shape: {df.shape} | Default rate: {df['TARGET'].mean():.2%}")
    return df


# ================================================================
# STEP 1 — Remove duplicates & bad rows
# ================================================================
def clean_data(df):
    print("\n[1] Cleaning data...")
    avail = [f for f in ALL_FEATURES if f in df.columns]

    before = len(df)
    df = df.dropna(subset=["TARGET"])
    df = df.drop_duplicates(subset=avail)
    print(f"  Removed {before - len(df):,} duplicate/null rows. Remaining: {len(df):,}")
    return df


# ================================================================
# STEP 2 — Outlier Capping (Winsorization)
# ================================================================
def handle_outliers(df):
    print("\n[2] Winsorizing outliers (1st–99th percentile)...")
    avail = [f for f in ALL_FEATURES if f in df.columns and
             df[f].dtype in [np.float64, np.int64, "float32", "int32"]]
    df = df.copy()
    clipped = 0
    for col in avail:
        lo, hi = df[col].quantile([0.01, 0.99])
        if lo < hi:
            before = df[col].clip(lo, hi)
            df[col] = before
            clipped += 1
    print(f"  Winsorized {clipped} features")
    return df


# ================================================================
# STEP 3 — Impute Missing Values
# ================================================================
def impute_missing(df):
    print("\n[3] Imputing missing values...")
    avail = [f for f in ALL_FEATURES if f in df.columns]
    miss_rates = df[avail].isnull().mean()

    no_miss    = [f for f in avail if miss_rates[f] == 0]
    # Only impute features that have AT LEAST one real value
    can_impute = [f for f in avail if 0 < miss_rates[f] < 1.0]
    # Features that are 100% empty — just fill with 0
    all_empty  = [f for f in avail if miss_rates[f] == 1.0]

    print(f"  Complete (0% missing):     {len(no_miss)} features")
    print(f"  Median impute:             {len(can_impute)} features")
    print(f"  All empty → fill 0:        {len(all_empty)} features")
    if all_empty:
        print(f"  Empty features: {all_empty}")

    df = df.copy()

    # Fill completely empty features with 0
    if all_empty:
        df[all_empty] = df[all_empty].fillna(0)

    # Median impute the rest
    if can_impute:
        imp = SimpleImputer(strategy="median")
        df[can_impute] = imp.fit_transform(df[can_impute])

    remaining = df[avail].isnull().sum().sum()
    print(f"  Missing values remaining:  {remaining}")
    return df


# ================================================================
# STEP 4 — Feature Selection (keep best 80)
# ================================================================
def select_features(df):
    print("\n[4] Feature selection...")
    avail = [f for f in ALL_FEATURES if f in df.columns]

    # Ensure at least one from each category is included
    must_have = []
    for grp, feats in FEATURE_GROUPS.items():
        grp_avail = [f for f in feats if f in avail]
        must_have.extend(grp_avail[:2])   # at least 2 per category
    must_have = list(dict.fromkeys(must_have))

    # Score remaining features with f_classif
    remaining = [f for f in avail if f not in must_have]
    if remaining:
        X_sel = df[remaining].fillna(0)
        y     = df["TARGET"]
        sel   = SelectKBest(f_classif, k=min(len(remaining), 80-len(must_have)))
        sel.fit(X_sel, y)
        scores      = pd.Series(sel.scores_, index=remaining).sort_values(ascending=False)
        selected_extra = scores.head(80 - len(must_have)).index.tolist()
    else:
        selected_extra = []

    final_features = must_have + [f for f in selected_extra if f not in must_have]
    final_features = final_features[:80]

    print(f"  Selected {len(final_features)} features")
    for grp, feats in FEATURE_GROUPS.items():
        grp_sel = [f for f in feats if f in final_features]
        print(f"    {grp:<25} {len(grp_sel)} features")

    return final_features


# ================================================================
# STEP 5 — Scale Features
# ================================================================
def scale_features(df, feature_cols):
    print("\n[5] Scaling with RobustScaler...")
    df = df.copy()
    scaler = RobustScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    joblib.dump(scaler, "models/feature_scaler.pkl")
    print(f"  ✅ Scaler saved: models/feature_scaler.pkl")
    return df, scaler


# ================================================================
# PLOT — Preprocessing Report
# ================================================================
def plot_preprocessing_report(df_before, df_after, feature_cols):
    print("\n📊 Generating preprocessing report plots...")
    os.makedirs("outputs/eda_plots", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Preprocessing Report", fontsize=14, fontweight="bold")

    # Missing values before vs after
    miss_before = df_before[feature_cols].isnull().mean() * 100
    miss_after  = df_after[feature_cols].isnull().mean() * 100
    axes[0,0].scatter(miss_before, miss_after, alpha=0.6, color="#2196F3")
    axes[0,0].plot([0,100],[0,100], "r--", linewidth=1)
    axes[0,0].set_xlabel("Missing % Before")
    axes[0,0].set_ylabel("Missing % After")
    axes[0,0].set_title("Missing Values: Before vs After Imputation")

    # Feature value distribution sample
    sample_feat = [f for f in feature_cols if df_before[f].notna().sum() > 1000][:3]
    for i, feat in enumerate(sample_feat[:3]):
        ax = [axes[0,1], axes[1,0], axes[1,1]][i]
        ax.hist(df_before[feat].dropna().clip(
                    df_before[feat].quantile(0.01),
                    df_before[feat].quantile(0.99)),
                bins=40, alpha=0.5, label="Before", color="#F44336")
        ax.hist(df_after[feat].dropna(),
                bins=40, alpha=0.5, label="After Scaling", color="#2196F3")
        ax.set_title(f"{feat}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("outputs/eda_plots/08_preprocessing_report.png", dpi=150)
    plt.close()
    print("  ✅ Saved: outputs/eda_plots/08_preprocessing_report.png")


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("ALTERNATIVE CREDIT SCORING — PREPROCESSING")
    print("=" * 55)

    os.makedirs("models", exist_ok=True)

    df = load_merged()
    df = clean_data(df)
    df_before = df.copy()
    df = handle_outliers(df)
    df = impute_missing(df)

    final_features = select_features(df)
    df_before_scale = df.copy()

    df, scaler = scale_features(df, final_features)

    plot_preprocessing_report(df_before, df, final_features)

    # Save final dataset
    meta_cols = ["source_id", "source_dataset", "TARGET"]
    save_cols = meta_cols + final_features
    available_save = [c for c in save_cols if c in df.columns]
    df[available_save].to_csv(PATHS["output_final"], index=False)
    print(f"\n💾 Saved: {PATHS['output_final']}")

    # Save feature list
    with open("outputs/selected_features.txt", "w") as f:
        for feat in final_features:
            f.write(feat + "\n")
    print(f"💾 Saved: outputs/selected_features.txt")

    print(f"\n{'='*55}")
    print(f"✅ PREPROCESSING COMPLETE")
    print(f"   Rows:     {len(df):,}")
    print(f"   Features: {len(final_features)}")
    print(f"   Default rate: {df['TARGET'].mean():.2%}")
    print(f"\n   Next step: python src/model_training.py")
