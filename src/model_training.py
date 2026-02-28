# src/model_training.py
# ================================================================
# STEP 5 — Model Training + SHAP + Credit Score Generation
# Run: python src/model_training.py
# Output: models/credit_score_model.pkl + outputs/scores.csv
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import shap
import joblib
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATHS, RANDOM_STATE, TEST_SIZE, SCORE_MIN, SCORE_MAX

from sklearn.model_selection  import train_test_split
from sklearn.metrics          import (roc_auc_score, classification_report,
                                      roc_curve, confusion_matrix)
from sklearn.preprocessing    import MinMaxScaler
from imblearn.over_sampling   import SMOTE

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/eda_plots", exist_ok=True)

LGBM_PARAMS = {
    "objective":         "binary",
    "metric":            "auc",
    "boosting_type":     "gbdt",
    "n_estimators":      1000,
    "learning_rate":     0.03,
    "num_leaves":        31,
    "max_depth":         -1,
    "min_child_samples": 50,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
    "verbose":           -1,
}


def load_data():
    print("=" * 55)
    print("STEP 1 — Loading Data")
    print("=" * 55)
    df = pd.read_csv(PATHS["output_final"], low_memory=False)
    meta_cols    = ["source_id", "source_dataset", "TARGET"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X    = df[feature_cols]
    y    = df["TARGET"]
    meta = df[meta_cols]
    print(f"  Rows: {len(df):,} | Features: {len(feature_cols)} | Default rate: {y.mean():.2%}")
    return X, y, meta, feature_cols


def split_and_balance(X, y):
    print("\n" + "=" * 55)
    print("STEP 2 — Split + SMOTE")
    print("=" * 55)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE: {len(X_bal):,} rows | Default rate: {y_bal.mean():.2%}")
    return X_bal, X_test, y_bal, y_test


def train_model(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 55)
    print("STEP 3 — Training LightGBM")
    print("=" * 55)
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(200)])
    print(f"  Best iteration: {model.best_iteration_}")
    return model


def evaluate(model, X_test, y_test):
    print("\n" + "=" * 55)
    print("STEP 4 — Evaluation")
    print("=" * 55)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    auc  = roc_auc_score(y_test, y_prob)
    gini = 2 * auc - 1
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ks   = max(tpr - fpr)
    print(f"  AUC:  {auc:.4f} {'✅ Excellent' if auc>0.80 else '✅ Good' if auc>0.75 else '⚠️ Fair'}")
    print(f"  Gini: {gini:.4f}")
    print(f"  KS:   {ks:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Default','Default'])}")

    # ROC curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color="steelblue", lw=2,
             label=f"AUC={auc:.3f} | Gini={gini:.3f} | KS={ks:.3f}")
    plt.plot([0,1],[0,1],"r--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("outputs/eda_plots/09_roc_curve.png", dpi=150)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Default","Default"],
                yticklabels=["No Default","Default"])
    plt.title("Confusion Matrix"); plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("outputs/eda_plots/10_confusion_matrix.png", dpi=150)
    plt.close()
    print("   Saved ROC + Confusion Matrix plots")
    return y_prob, auc, gini, ks


def prob_to_score(prob_default):
    prob     = np.clip(prob_default, 1e-6, 1 - 1e-6)
    log_odds = np.log((1 - prob) / prob)
    scaler   = MinMaxScaler(feature_range=(SCORE_MIN, SCORE_MAX))
    scores   = scaler.fit_transform(log_odds.reshape(-1,1)).flatten()
    return np.round(scores).astype(int), scaler


def get_band(score):
    if score >= 750:   return "Excellent"
    elif score >= 700: return "Very Good"
    elif score >= 650: return "Good"
    elif score >= 600: return "Fair"
    elif score >= 550: return "Poor"
    else:              return "Very Poor"


def generate_scores(model, X, meta):
    print("\n" + "=" * 55)
    print("STEP 5 — Credit Score Generation")
    print("=" * 55)
    prob           = model.predict_proba(X)[:, 1]
    scores, scaler = prob_to_score(prob)
    results        = meta.copy().reset_index(drop=True)
    results["prob_default"] = prob.round(4)
    results["credit_score"] = scores
    results["score_band"]   = [get_band(s) for s in scores]
    print(f"  Mean: {scores.mean():.0f} | Median: {np.median(scores):.0f}")
    print(f"\n  Band distribution:")
    print(results["score_band"].value_counts().to_string())

    # Score distribution
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    results["credit_score"].hist(bins=50, color="steelblue", edgecolor="white")
    plt.title("Credit Score Distribution"); plt.xlabel("Score"); plt.ylabel("Count")
    plt.subplot(1,2,2)
    order  = ["Very Poor","Poor","Fair","Good","Very Good","Excellent"]
    colors = ["#B71C1C","#E64A19","#F57F17","#F9A825","#388E3C","#2E7D32"]
    counts = results["score_band"].value_counts().reindex(order, fill_value=0)
    plt.bar(counts.index, counts.values, color=colors)
    plt.title("Score Band Distribution"); plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("outputs/eda_plots/11_score_distribution.png", dpi=150)
    plt.close()
    results.to_csv("outputs/credit_scores.csv", index=False)
    print("   Saved: outputs/credit_scores.csv")
    return results, scaler


def run_shap(model, X_test, feature_names):
    print("\n" + "=" * 55)
    print("STEP 6 — SHAP Explainability")
    print("=" * 55)
    print("  Computing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    X_sample    = X_test.iloc[:2000] if len(X_test) > 2000 else X_test
    shap_values = explainer.shap_values(X_sample)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(10,8))
    shap.summary_plot(sv, X_sample, feature_names=feature_names,
                      plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig("outputs/eda_plots/12_shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10,8))
    shap.summary_plot(sv, X_sample, feature_names=feature_names,
                      max_display=20, show=False)
    plt.tight_layout()
    plt.savefig("outputs/eda_plots/13_shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    top = pd.Series(np.abs(sv).mean(axis=0), index=feature_names).sort_values(ascending=False)
    print(f"  Top 10 features by SHAP:")
    for i,(f,v) in enumerate(top.head(10).items(),1):
        print(f"    {i:2d}. {f:<40} {v:.4f}")
    print("   Saved SHAP plots")
    return explainer


def save_bundle(model, explainer, feature_names, scaler, auc, gini, ks):
    print("\n" + "=" * 55)
    print("STEP 7 — Saving Model")
    print("=" * 55)
    bundle = {
        "model":         model,
        "explainer":     explainer,
        "feature_names": feature_names,
        "score_scaler":  scaler,
        "metrics":       {"auc": auc, "gini": gini, "ks": ks},
        "score_min":     SCORE_MIN,
        "score_max":     SCORE_MAX,
    }
    joblib.dump(bundle, PATHS["model_path"])
    print(f"   Saved: {PATHS['model_path']}")


if __name__ == "__main__":
    print("=" * 55)
    print("ALTERNATIVE CREDIT SCORING — MODEL TRAINING")
    print("=" * 55)

    X, y, meta, feature_names = load_data()
    X_train, X_test, y_train, y_test = split_and_balance(X, y)
    model = train_model(X_train, X_test, y_train, y_test)
    y_prob, auc, gini, ks = evaluate(model, X_test, y_test)
    _, scaler = generate_scores(model, X, meta)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    explainer = run_shap(model, X_test_df, feature_names)
    save_bundle(model, explainer, feature_names, scaler, auc, gini, ks)

    print("\n DONE! Run: streamlit run app.py")

# Add to src/model_training.py
# Run: pip install optuna first

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def tune_hyperparameters(X_train, X_test, y_train, y_test):
    print("\nRunning Optuna hyperparameter search (50 trials)...")
    
    def objective(trial):
        params = {
            "objective":         "binary",
            "metric":            "auc",
            "verbose":           -1,
            "random_state":      42,
            "n_estimators":      1000,
            "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":      trial.suggest_int("bagging_freq", 1, 10),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        return roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    print(f"  Best AUC:    {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params
