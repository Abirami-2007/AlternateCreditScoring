# config.py
# ================================================================
# CENTRAL CONFIGURATION — edit BASE_DATA_PATH only
# ================================================================
import os

# ── Set this to where you downloaded your datasets ──
BASE_DATA_PATH = "data"   # relative to project root

PATHS = {
    # Home Credit
    "hc_main":          f"{BASE_DATA_PATH}/home_credit/application_train.csv",
    "hc_bureau":        f"{BASE_DATA_PATH}/home_credit/bureau.csv",
    "hc_bureau_bal":    f"{BASE_DATA_PATH}/home_credit/bureau_balance.csv",
    "hc_installments":  f"{BASE_DATA_PATH}/home_credit/installments_payments.csv",
    "hc_pos":           f"{BASE_DATA_PATH}/home_credit/POS_CASH_balance.csv",
    "hc_cc":            f"{BASE_DATA_PATH}/home_credit/credit_card_balance.csv",
    "hc_prev":          f"{BASE_DATA_PATH}/home_credit/previous_application.csv",
    # Give Me Some Credit
    "gmsc":             f"{BASE_DATA_PATH}/give_me_some_credit/cs-training.csv",
    # Lending Club
    "lc":               f"{BASE_DATA_PATH}/lending_club/loan.csv",
    # PKDD Czech
    "pkdd_loan":        f"{BASE_DATA_PATH}/pkdd_czech/loan.csv",
    "pkdd_trans":       f"{BASE_DATA_PATH}/pkdd_czech/trans.csv",
    "pkdd_account":     f"{BASE_DATA_PATH}/pkdd_czech/account.csv",
    # Outputs
    "output_merged":    "outputs/unified_credit_dataset.csv",
    "output_final":     "outputs/final_80_features.csv",
    "output_eda":       "outputs/eda_report.html",
    "model_path":       "models/credit_score_model.pkl",
}

RANDOM_STATE = 42
TEST_SIZE    = 0.20
SCORE_MIN    = 300
SCORE_MAX    = 900

# 80 Features grouped by category
FEATURE_GROUPS = {
    "income_loan": [
        "income_total", "credit_amount", "annuity_amount",
        "credit_to_income", "annuity_to_income", "credit_to_goods",
        "income_per_person", "credit_per_month", "installment_rate_pct",
        "loan_amount_to_goods",
    ],
    "alternative_signals": [
        "ext_source_1", "ext_source_2", "ext_source_3",
        "ext_source_mean", "ext_source_std", "ext_source_min", "ext_source_count",
        "checking_account_ordinal", "savings_account_ordinal",
        "has_savings", "savings_above_500", "stability_score",
        "docs_submission_rate", "has_checking_account",
    ],
    "payment_behaviour": [
        "inst_late_payment_rate", "inst_early_payment_rate", "inst_on_time_rate",
        "inst_payment_consistency", "inst_days_late_mean", "inst_days_late_max",
        "inst_payment_ratio_mean", "inst_payment_ratio_std",
        "pos_dpd_rate", "pos_dpd_occurrences", "pos_sk_dpd_max", "pos_completion_ratio",
        "cc_payment_rate_mean", "cc_payment_rate_std", "cc_dpd_mean", "cc_dpd_max",
    ],
    "credit_history": [
        "credit_history_ordinal", "has_past_delays", "has_critical_account",
        "bureau_debt_to_credit_ratio", "bureau_credit_day_overdue_max",
        "bureau_credit_sum_overdue", "bureau_bad_status_months",
        "bureau_dpd_months_mean", "bureau_active_credit_ratio",
        "prev_approval_rate", "prev_refusal_rate", "prev_total_applications",
        "has_loan_history",
    ],
    "stability_demographics": [
        "age_years", "employment_years", "employed_flag", "employment_stability",
        "is_employed", "senior_employee", "present_residence_since",
        "own_car_flag", "own_realty_flag", "owns_property", "owns_house",
        "has_telephone", "region_rating", "live_work_same_region",
        "family_members", "children_count",
    ],
    "credit_utilisation": [
        "cc_utilization_mean", "cc_utilization_max",
        "cc_atm_to_total_drawing", "cc_drawing_rate_mean",
        "existing_credits_count", "total_enquiries", "recent_enquiries_1yr",
        "debt_ratio",
    ],
    "loan_structure": [
        "name_contract_type", "loan_purpose_high_risk", "loan_purpose_encoded",
        "has_guarantor", "has_co_applicant", "has_other_installments",
        "other_install_bank",
    ],
}

ALL_FEATURES = []
seen = set()
for grp in FEATURE_GROUPS.values():
    for f in grp:
        if f not in seen:
            ALL_FEATURES.append(f)
            seen.add(f)
