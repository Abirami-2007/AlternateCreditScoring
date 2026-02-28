# src/eda.py
# ================================================================
# STEP 3 — Exploratory Data Analysis
# Run: python src/eda.py
# Outputs: outputs/eda_plots/ folder + outputs/eda_summary.txt
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATHS, ALL_FEATURES, FEATURE_GROUPS

# Plot style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
PLOT_DIR = "outputs/eda_plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def load_data():
    print("Loading merged dataset...")
    df = pd.read_csv(PATHS["output_merged"], low_memory=False)
    print(f"  Shape: {df.shape}")
    print(f"  Default rate: {df['TARGET'].mean():.2%}")
    return df


# ================================================================
# PLOT 1 — Dataset Overview
# ================================================================
def plot_dataset_overview(df):
    print("\nPlot 1: Dataset Overview...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dataset Overview", fontsize=16, fontweight="bold")

    # 1A: Row count by dataset
    counts = df["source_dataset"].value_counts()
    axes[0,0].bar(counts.index, counts.values,
                  color=["#2196F3","#4CAF50","#FF9800","#9C27B0"])
    axes[0,0].set_title("Rows per Dataset")
    axes[0,0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0,0].text(i, v + 500, f"{v:,}", ha="center", fontsize=10)

    # 1B: Default rate per dataset
    dr = df.groupby("source_dataset")["TARGET"].mean() * 100
    bars = axes[0,1].bar(dr.index, dr.values,
                          color=["#2196F3","#4CAF50","#FF9800","#9C27B0"])
    axes[0,1].set_title("Default Rate per Dataset (%)")
    axes[0,1].set_ylabel("Default Rate %")
    axes[0,1].axhline(df["TARGET"].mean()*100, color="red",
                       linestyle="--", label=f"Overall: {df['TARGET'].mean():.1%}")
    axes[0,1].legend()
    for bar, val in zip(bars, dr.values):
        axes[0,1].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.3, f"{val:.1f}%", ha="center")

    # 1C: Overall class distribution
    target_counts = df["TARGET"].value_counts()
    axes[1,0].pie(target_counts.values,
                   labels=["No Default","Default"],
                   colors=["#2196F3","#F44336"],
                   autopct="%1.1f%%", startangle=90,
                   explode=(0, 0.05))
    axes[1,0].set_title("Overall Class Distribution")

    # 1D: New user vs existing users
    new_users = (df["has_loan_history"] == 0).sum()
    exist_users = (df["has_loan_history"] == 1).sum()
    axes[1,1].bar(["Existing Credit Users","New Users (No History)"],
                   [exist_users, new_users], color=["#2196F3","#FF9800"])
    axes[1,1].set_title("New vs Existing Credit Users")
    axes[1,1].set_ylabel("Count")
    for i, v in enumerate([exist_users, new_users]):
        axes[1,1].text(i, v + 200, f"{v:,}\n({v/len(df):.1%})", ha="center")

    plt.tight_layout()
    path = f"{PLOT_DIR}/01_dataset_overview.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")


# ================================================================
# PLOT 2 — Missing Values Heatmap
# ================================================================
def plot_missing_values(df):
    print("\n Plot 2: Missing Values...")
    avail = [f for f in ALL_FEATURES if f in df.columns]
    miss  = df.groupby("source_dataset")[avail].apply(lambda x: x.isnull().mean())

    fig, axes = plt.subplots(1, 2, figsize=(22, 6))
    fig.suptitle("Missing Value Analysis", fontsize=14, fontweight="bold")

    # Heatmap by dataset
    sns.heatmap(miss, ax=axes[0], cmap="YlOrRd", vmin=0, vmax=1,
                cbar_kws={"label":"Missing Rate"},
                linewidths=0.3, linecolor="white")
    axes[0].set_title("Missing Rate by Dataset × Feature\n(Red=100% missing, White=0%)")
    axes[0].set_xlabel("Features")
    axes[0].set_ylabel("Dataset")
    axes[0].tick_params(axis="x", rotation=90, labelsize=6)

    # Overall missing rate bar chart (top 30 most missing)
    overall_miss = df[avail].isnull().mean().sort_values(ascending=False).head(30)
    colors = ["#F44336" if v > 0.5 else "#FF9800" if v > 0.2 else "#4CAF50"
              for v in overall_miss.values]
    axes[1].barh(overall_miss.index, overall_miss.values * 100, color=colors)
    axes[1].set_title("Top 30 Features by Missing Rate")
    axes[1].set_xlabel("Missing %")
    axes[1].axvline(40, color="red", linestyle="--", label="40% threshold")
    axes[1].legend()

    plt.tight_layout()
    path = f"{PLOT_DIR}/02_missing_values.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")


# ================================================================
# PLOT 3 — Target Variable Analysis
# ================================================================
def plot_target_analysis(df):
    print("\n Plot 3: Target Analysis...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Default Rate Analysis", fontsize=14, fontweight="bold")

    # Age group default rate
    df["age_group"] = pd.cut(df["age_years"].dropna(),
                              bins=[0,25,35,45,55,65,100],
                              labels=["<25","25-35","35-45","45-55","55-65","65+"])
    age_dr = df.groupby("age_group", observed=True)["TARGET"].mean() * 100
    axes[0].bar(age_dr.index.astype(str), age_dr.values, color="#2196F3")
    axes[0].set_title("Default Rate by Age Group")
    axes[0].set_xlabel("Age Group")
    axes[0].set_ylabel("Default Rate %")
    for i, v in enumerate(age_dr.values):
        axes[0].text(i, v+0.2, f"{v:.1f}%", ha="center", fontsize=9)

    # Employment default rate
    emp_dr = df.groupby("employed_flag")["TARGET"].mean() * 100
    axes[1].bar(["Unemployed","Employed"],
                 [emp_dr.get(0, 0), emp_dr.get(1, 0)],
                 color=["#F44336","#4CAF50"])
    axes[1].set_title("Default Rate: Employed vs Unemployed")
    axes[1].set_ylabel("Default Rate %")

    # Loan history default rate
    if "has_loan_history" in df.columns:
        hist_dr = df.groupby("has_loan_history")["TARGET"].mean() * 100
        axes[2].bar(["No Loan History\n(New User)","Has Loan History"],
                     [hist_dr.get(0, 0), hist_dr.get(1, 0)],
                     color=["#FF9800","#2196F3"])
        axes[2].set_title("Default Rate: New vs Existing Users")
        axes[2].set_ylabel("Default Rate %")

    plt.tight_layout()
    path = f"{PLOT_DIR}/03_target_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")


# ================================================================
# PLOT 4 — Feature Distributions (key features)
# ================================================================
def plot_feature_distributions(df):
    print("\n Plot 4: Feature Distributions...")

    key_features = [
        "credit_to_income", "age_years", "inst_late_payment_rate",
        "ext_source_mean", "cc_utilization_mean", "income_total",
        "employment_years", "bureau_debt_to_credit_ratio",
    ]
    avail = [f for f in key_features if f in df.columns and df[f].notna().sum() > 100]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Key Feature Distributions (Default vs No Default)",
                 fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for i, feat in enumerate(avail[:8]):
        ax = axes[i]
        no_def = df[df["TARGET"]==0][feat].dropna()
        yes_def= df[df["TARGET"]==1][feat].dropna()

        # Clip at 99th percentile for readability
        cap = df[feat].quantile(0.99)
        no_def  = no_def.clip(upper=cap)
        yes_def = yes_def.clip(upper=cap)

        ax.hist(no_def,  bins=50, alpha=0.6, label="No Default",
                color="#2196F3", density=True)
        ax.hist(yes_def, bins=50, alpha=0.6, label="Default",
                color="#F44336", density=True)
        ax.set_title(feat.replace("_"," ").title(), fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = f"{PLOT_DIR}/04_feature_distributions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")


# ================================================================
# PLOT 5 — Correlation Heatmap
# ================================================================
def plot_correlation(df):
    print("\n Plot 5: Correlation Heatmap...")
    avail = [f for f in ALL_FEATURES if f in df.columns and
             df[f].dtype in [np.float64, np.int64]][:30]

    corr = df[avail + ["TARGET"]].corr()

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    fig.suptitle("Feature Correlation Analysis", fontsize=14, fontweight="bold")

    # Full correlation heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=axes[0], mask=mask, cmap="RdBu_r",
                vmin=-1, vmax=1, center=0,
                square=True, linewidths=0.3,
                cbar_kws={"shrink":0.7},
                annot=False)
    axes[0].set_title("Feature Correlation Matrix (Top 30 Features)")
    axes[0].tick_params(axis="x", rotation=90, labelsize=7)
    axes[0].tick_params(axis="y", labelsize=7)

    # Correlation with TARGET
    target_corr = corr["TARGET"].drop("TARGET").sort_values()
    colors = ["#F44336" if v > 0 else "#2196F3" for v in target_corr.values]
    axes[1].barh(range(len(target_corr)), target_corr.values, color=colors)
    axes[1].set_yticks(range(len(target_corr)))
    axes[1].set_yticklabels(target_corr.index, fontsize=8)
    axes[1].set_title("Feature Correlation with TARGET\n(Red=positive/risky, Blue=negative/safe)")
    axes[1].set_xlabel("Correlation")
    axes[1].axvline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    path = f"{PLOT_DIR}/05_correlation_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")


# ================================================================
# PLOT 6 — Alternative Signals for New Users
# ================================================================
def plot_alternative_signals(df):
    print("\n Plot 6: Alternative Signals (New User Focus)...")
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Alternative Credit Signals — New User Scoring Power",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # Savings vs Default Rate
    ax1 = fig.add_subplot(gs[0,0])
    if "has_savings" in df.columns and df["has_savings"].notna().sum() > 100:
        sav_dr = df.groupby("has_savings")["TARGET"].mean() * 100
        ax1.bar(["No Savings","Has Savings"],
                 [sav_dr.get(0,0), sav_dr.get(1,0)],
                 color=["#F44336","#4CAF50"])
        ax1.set_title("Default Rate:\nSavings Account")
        ax1.set_ylabel("Default Rate %")

    # Checking Account vs Default Rate
    ax2 = fig.add_subplot(gs[0,1])
    if "checking_account_ordinal" in df.columns and df["checking_account_ordinal"].notna().sum() > 100:
        chk = df.dropna(subset=["checking_account_ordinal"])
        chk_dr = chk.groupby("checking_account_ordinal")["TARGET"].mean() * 100
        ax2.bar(["No Account","< 0","0-200","200+"],
                 [chk_dr.get(0,0), chk_dr.get(1,0), chk_dr.get(2,0), chk_dr.get(3,0)],
                 color=["#F44336","#FF9800","#FF9800","#4CAF50"])
        ax2.set_title("Default Rate:\nChecking Account Balance Tier")
        ax2.set_ylabel("Default Rate %")

    # Employment Stability vs Default Rate
    ax3 = fig.add_subplot(gs[0,2])
    if "employment_stability" in df.columns and df["employment_stability"].notna().sum() > 100:
        emp_labels = {0:"Unemployed",1:"<1yr",2:"1-3yr",3:"4-7yr",4:"7+yr"}
        emp_dr = df.groupby("employment_stability")["TARGET"].mean() * 100
        ax3.bar([emp_labels.get(k,str(k)) for k in emp_dr.index],
                 emp_dr.values, color="#2196F3")
        ax3.set_title("Default Rate:\nEmployment Stability")
        ax3.set_ylabel("Default Rate %")
        ax3.tick_params(axis="x", rotation=30)

    # Payment consistency vs Default
    ax4 = fig.add_subplot(gs[1,0])
    if "inst_late_payment_rate" in df.columns:
        df["late_bucket"] = pd.cut(
            df["inst_late_payment_rate"].clip(0,0.5),
            bins=[0,0.05,0.1,0.2,0.3,0.5],
            labels=["0-5%","5-10%","10-20%","20-30%","30-50%"])
        late_dr = df.groupby("late_bucket", observed=True)["TARGET"].mean() * 100
        ax4.bar(late_dr.index.astype(str), late_dr.values, color="#FF9800")
        ax4.set_title("Default Rate:\nLate Payment Rate Bucket")
        ax4.set_ylabel("Default Rate %")

    # External source score distribution
    ax5 = fig.add_subplot(gs[1,1])
    if "ext_source_mean" in df.columns and df["ext_source_mean"].notna().sum() > 100:
        no_def  = df[df["TARGET"]==0]["ext_source_mean"].dropna()
        yes_def = df[df["TARGET"]==1]["ext_source_mean"].dropna()
        ax5.hist(no_def,  bins=40, alpha=0.6, label="No Default",
                 color="#2196F3", density=True)
        ax5.hist(yes_def, bins=40, alpha=0.6, label="Default",
                 color="#F44336", density=True)
        ax5.set_title("External Source Score\nDistribution by Target")
        ax5.set_xlabel("External Score (mean)")
        ax5.legend()

    # Score gap: New users vs Existing
    ax6 = fig.add_subplot(gs[1,2])
    ext_new = df[df["ext_source_count"]==0]["ext_source_mean"] if "ext_source_count" in df.columns else pd.Series()
    ext_exist= df[df.get("ext_source_count",pd.Series(dtype=float))>0]["ext_source_mean"] if "ext_source_count" in df.columns else pd.Series()
    ax6.bar(["New Users\n(No ext score)","Existing Users\n(Has ext score)"],
             [df[df.get("has_loan_history",pd.Series(1,index=df.index))==0]["TARGET"].mean()*100,
              df[df.get("has_loan_history",pd.Series(1,index=df.index))==1]["TARGET"].mean()*100],
             color=["#FF9800","#2196F3"])
    ax6.set_title("Default Rate:\nNew vs Existing Credit Users")
    ax6.set_ylabel("Default Rate %")

    plt.tight_layout()
    path = f"{PLOT_DIR}/06_alternative_signals.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")


# ================================================================
# PLOT 7 — Feature Coverage by Dataset
# ================================================================
def plot_feature_coverage(df):
    print("\n Plot 7: Feature Coverage by Dataset...")
    avail = [f for f in ALL_FEATURES if f in df.columns]
    coverage = df.groupby("source_dataset")[avail].apply(
        lambda x: (x.notna().mean() * 100))

    fig, ax = plt.subplots(figsize=(22, 6))
    im = ax.imshow(coverage.values, cmap="YlGn", aspect="auto", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="Coverage %")
    ax.set_xticks(range(len(avail)))
    ax.set_xticklabels(avail, rotation=90, fontsize=6)
    ax.set_yticks(range(len(coverage.index)))
    ax.set_yticklabels(coverage.index)
    ax.set_title("Feature Coverage by Dataset (Green=100% available, Yellow=partial)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{PLOT_DIR}/07_feature_coverage.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")


# ================================================================
# TEXT SUMMARY REPORT
# ================================================================
def generate_text_report(df):
    avail = [f for f in ALL_FEATURES if f in df.columns]
    miss  = df[avail].isnull().mean()

    lines = []
    lines.append("=" * 60)
    lines.append("ALTERNATIVE CREDIT SCORING — EDA SUMMARY REPORT")
    lines.append("=" * 60)

    lines.append("\n[1] DATASET OVERVIEW")
    for ds in df["source_dataset"].unique():
        sub = df[df["source_dataset"] == ds]
        lines.append(f"  {ds:<30} {len(sub):>8,} rows  default={sub['TARGET'].mean():.2%}")
    lines.append(f"\n  TOTAL: {len(df):,} rows | Default rate: {df['TARGET'].mean():.2%}")
    lines.append(f"  Features available: {len(avail)}/80")

    lines.append("\n[2] CLASS IMBALANCE")
    lines.append(f"  No Default: {(df['TARGET']==0).sum():,} ({(df['TARGET']==0).mean():.1%})")
    lines.append(f"  Default:    {(df['TARGET']==1).sum():,} ({(df['TARGET']==1).mean():.1%})")
    lines.append(f"  Imbalance ratio: {(df['TARGET']==0).sum()/(df['TARGET']==1).sum():.1f}:1")

    lines.append("\n[3] NEW USER ANALYSIS")
    new_mask = df.get("has_loan_history", pd.Series(1, index=df.index)) == 0
    lines.append(f"  New users (no loan history): {new_mask.sum():,} ({new_mask.mean():.1%})")
    if new_mask.sum() > 0:
        lines.append(f"  New user default rate: {df[new_mask]['TARGET'].mean():.2%}")
        lines.append(f"  Existing user default rate: {df[~new_mask]['TARGET'].mean():.2%}")

    lines.append("\n[4] TOP 15 MOST MISSING FEATURES")
    for feat, rate in miss.sort_values(ascending=False).head(15).items():
        bar = "█" * int(rate * 20)
        lines.append(f"  {feat:<45} {rate:>6.1%}  {bar}")

    lines.append("\n[5] FEATURES WITH 0% MISSING (COMPLETE)")
    complete = miss[miss == 0].index.tolist()
    for f in complete:
        lines.append(f"   {f}")

    lines.append("\n[6] KEY STATISTICS")
    for feat in ["age_years","credit_to_income","inst_late_payment_rate","ext_source_mean"]:
        if feat in df.columns and df[feat].notna().sum() > 10:
            lines.append(f"\n  {feat}:")
            lines.append(f"    mean={df[feat].mean():.3f} | median={df[feat].median():.3f} | "
                        f"std={df[feat].std():.3f} | "
                        f"missing={df[feat].isnull().mean():.1%}")

    report = "\n".join(lines)
    with open("outputs/eda_summary.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\n Saved: outputs/eda_summary.txt")


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("ALTERNATIVE CREDIT SCORING — EDA")
    print("=" * 55)

    df = load_data()

    plot_dataset_overview(df)
    plot_missing_values(df)
    plot_target_analysis(df)
    plot_feature_distributions(df)
    plot_correlation(df)
    plot_alternative_signals(df)
    plot_feature_coverage(df)
    generate_text_report(df)

    print(f"\n EDA Complete!")
    print(f"   All plots saved to: {PLOT_DIR}/")
    print(f"   Summary saved to:   outputs/eda_summary.txt")
    print(f"\n   Next step: python src/preprocessing.py")
