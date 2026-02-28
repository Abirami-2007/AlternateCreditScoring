# backend/main.py
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import joblib, numpy as np, pandas as pd
import jwt, time, os, io, warnings, logging
warnings.filterwarnings("ignore")

# ── Terminal logger ───────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("zenith")

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
DIM    = "\033[2m"

def log_score(score, band, prob, t_feat, t_model, t_shap, t_total, mode, user_email):
    band_color = GREEN if score>=700 else YELLOW if score>=600 else RED
    bar_len     = int((score-300)/600 * 30)
    bar         = "█" * bar_len + "░" * (30-bar_len)

    log.info(f"\n{BOLD}{'─'*58}{RESET}")
    log.info(f"{BOLD}  ZENITH CREDIT SCORE  ·  {mode}{RESET}  {DIM}({user_email}){RESET}")
    log.info(f"{'─'*58}")
    log.info(f"  Score   {band_color}{BOLD}{score:>4}{RESET}  [{band_color}{bar}{RESET}]  {band_color}{band}{RESET}")
    log.info(f"  Default {YELLOW}{prob:.1f}%{RESET} probability")
    log.info(f"{'─'*58}")
    log.info(f"  {CYAN}Timing Breakdown:{RESET}")
    log.info(f"    Feature engineering  {t_feat*1000:>7.2f} ms")
    log.info(f"    {BOLD}Model prediction     {t_model*1000:>7.2f} ms  ← LightGBM inference{RESET}")
    log.info(f"    SHAP explanation     {t_shap*1000:>7.2f} ms")
    log.info(f"    {BOLD}{'─'*36}{RESET}")
    log.info(f"    {GREEN}Total response time  {t_total*1000:>7.2f} ms{RESET}")
    log.info(f"{'─'*58}\n")

def log_batch(n_total, n_ok, t_total_ms, avg_ms):
    log.info(f"\n{BOLD}{'─'*58}{RESET}")
    log.info(f"{BOLD}  ZENITH BATCH SCORING{RESET}")
    log.info(f"{'─'*58}")
    log.info(f"  Applicants   {GREEN}{n_ok}/{n_total} scored successfully{RESET}")
    log.info(f"  {BOLD}Total time   {t_total_ms:.1f} ms{RESET}")
    log.info(f"  Avg per row  {avg_ms:.1f} ms  ← {1000/max(avg_ms,1):.0f} scores/second throughput")
    log.info(f"{'─'*58}\n")

app = FastAPI(title="ZENITH Credit Scoring API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

SECRET_KEY = "zenith_credit_2024_secret"
ALGORITHM  = "HS256"
security   = HTTPBearer()

USERS = {
    "admin@zenith.com":   {"password":"admin123",   "name":"Admin User",     "role":"admin"},
    "analyst@zenith.com": {"password":"analyst123", "name":"Credit Analyst", "role":"analyst"},
    "demo@zenith.com":    {"password":"demo123",    "name":"Demo User",      "role":"viewer"},
}

MODEL_BUNDLE = None
def get_model():
    global MODEL_BUNDLE
    if MODEL_BUNDLE is None:
        for path in ["../models/credit_score_model.pkl", "models/credit_score_model.pkl"]:
            if os.path.exists(path):
                MODEL_BUNDLE = joblib.load(path)
                break
    return MODEL_BUNDLE


# ── AUTH ──────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: str
    password: str

def create_token(email, name, role):
    payload = {"sub":email,"name":name,"role":role,"exp":time.time()+86400}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        return jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/api/auth/login")
def login(req: LoginRequest):
    user = USERS.get(req.email)
    if not user or user["password"] != req.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(req.email, user["name"], user["role"])
    return {"token":token,"name":user["name"],"role":user["role"],"email":req.email}

@app.get("/api/auth/me")
def me(user=Depends(verify_token)):
    return user


# ── FEATURE BUILDER ───────────────────────────────────────────────
def build_features(inp: dict, feature_names: list) -> pd.DataFrame:
    r  = {f: np.nan for f in feature_names}
    ai = float(inp.get("annual_income", 480000))
    la = float(inp.get("loan_amount",   250000))

    def n(k, d=0):
        try: return float(inp.get(k, d))
        except: return float(d)
    def b(k, d=False):
        v = inp.get(k, d)
        if isinstance(v, bool): return v
        return str(v).lower() in ["true","1","yes"]

    r["income_total"]          = ai
    r["credit_amount"]         = la
    r["annuity_amount"]        = n("monthly_payment",8500)
    r["credit_to_income"]      = la/(ai+1)
    r["annuity_to_income"]     = n("monthly_payment",8500)*12/(ai+1)
    r["income_per_person"]     = ai/(n("family_size",3)+1)
    r["credit_per_month"]      = la/(n("loan_term",36)+1)
    r["installment_rate_pct"]  = n("dti",25)
    r["debt_ratio"]            = n("dti",25)/100
    r["loan_amount_to_goods"]  = la/(la*1.05+1)
    r["credit_to_goods"]       = r["loan_amount_to_goods"]

    r["has_savings"]               = int(b("has_savings",True))
    r["savings_above_500"]         = int(n("savings_amount",25000)>500)
    r["has_checking_account"]      = int(b("has_checking",True))
    r["checking_account_ordinal"]  = n("checking_tier",3)
    r["savings_account_ordinal"]   = n("savings_tier",3)
    r["docs_submission_rate"]      = n("docs_rate",0.85)
    is_new                         = b("is_new_user",False)
    r["ext_source_count"]          = 0 if is_new else 2
    r["ext_source_mean"]           = np.nan if is_new else 0.55
    r["stability_score"]           = (n("checking_tier",3)+n("savings_tier",3)+
                                       n("employment_stability",3)+int(b("owns_property")))

    lr = n("late_payment_rate",0.05)
    ml = n("max_days_late",0)
    r["inst_late_payment_rate"]    = lr
    r["inst_on_time_rate"]         = 1-lr
    r["inst_payment_consistency"]  = 1/(lr+.01)
    r["inst_days_late_max"]        = ml
    r["inst_payment_ratio_mean"]   = 1-lr*.3
    r["inst_payment_ratio_std"]    = lr*.2
    r["inst_early_payment_rate"]   = max(0,.1-lr)
    r["inst_days_late_mean"]       = ml*.4
    r["pos_dpd_rate"]              = lr*.5
    r["pos_dpd_occurrences"]       = lr*24
    r["pos_sk_dpd_max"]            = ml
    r["pos_completion_ratio"]      = 1-lr*.3
    r["cc_payment_rate_mean"]      = 1-lr*.5
    r["cc_payment_rate_std"]       = lr*.1
    r["cc_dpd_mean"]               = ml*.2
    r["cc_dpd_max"]                = ml
    r["cc_utilization_mean"]       = n("cc_utilization",0.30)
    r["cc_utilization_max"]        = min(n("cc_utilization",0.30)*1.3,1)
    r["cc_drawing_rate_mean"]      = n("cc_utilization",0.30)*.8
    r["cc_atm_to_total_drawing"]   = .2

    r["credit_history_ordinal"]    = n("credit_history",4)
    r["has_past_delays"]           = int(b("has_past_delays"))
    r["has_critical_account"]      = int(ml>60)
    r["has_loan_history"]          = 0 if is_new else 1
    r["prev_approval_rate"]        = n("approval_rate",0.80)
    r["prev_refusal_rate"]         = 1-n("approval_rate",0.80)
    r["prev_total_applications"]   = n("prev_applications",2)
    r["bureau_debt_to_credit_ratio"]   = n("dti",25)/100
    r["bureau_credit_day_overdue_max"] = ml
    r["bureau_bad_status_months"]      = int(lr*24)
    r["bureau_credit_sum_overdue"]     = la*lr*.1
    r["bureau_dpd_months_mean"]        = lr*3
    r["bureau_active_credit_ratio"]    = min(n("existing_credits",2)/10,1)

    r["age_years"]               = n("age",30)
    r["employment_years"]        = n("employment_years",4)
    r["employed_flag"]           = int(b("employed",True))
    r["is_employed"]             = r["employed_flag"]
    r["employment_stability"]    = n("employment_stability",3)
    r["senior_employee"]         = int(n("employment_years",4)>=4)
    r["own_realty_flag"]         = int(b("owns_property"))
    r["owns_house"]              = r["own_realty_flag"]
    r["owns_property"]           = r["own_realty_flag"]
    r["own_car_flag"]            = 0
    r["family_members"]          = n("family_size",3)
    r["children_count"]          = max(0,n("family_size",3)-1)
    r["has_telephone"]           = 1
    r["region_rating"]           = n("region_rating",1)
    r["live_work_same_region"]   = 1
    r["existing_credits_count"]  = n("existing_credits",2)
    r["total_enquiries"]         = n("enquiries",1)
    r["recent_enquiries_1yr"]    = n("enquiries",1)

    r["loan_purpose_high_risk"]  = int(b("high_risk_purpose"))
    r["loan_purpose_encoded"]    = int(b("high_risk_purpose"))
    r["has_guarantor"]           = int(b("has_guarantor"))
    r["has_co_applicant"]        = 0
    r["has_other_installments"]  = int(n("existing_credits",2)>0)
    r["other_install_bank"]      = 0
    r["name_contract_type"]      = 0

    return pd.DataFrame([{f: r.get(f,np.nan) for f in feature_names}])


def prob_to_score(p, lo=300, hi=900):
    p  = np.clip(p,1e-6,1-1e-6)
    od = np.clip(np.log((1-p)/p),-4,4)
    return int(round(lo+(od+4)/8*(hi-lo)))

def get_band(s):
    if s>=750: return "Excellent"
    if s>=700: return "Very Good"
    if s>=650: return "Good"
    if s>=600: return "Fair"
    if s>=550: return "Poor"
    return "Very Poor"

def get_rec(s):
    if s>=700: return "Approve"
    if s>=600: return "Conditional"
    return "Reject"

LABELS = {
    "credit_to_income":"Loan-to-income ratio","annuity_to_income":"Monthly payment burden",
    "inst_late_payment_rate":"Late payment rate","has_savings":"Savings account",
    "savings_account_ordinal":"Savings balance","checking_account_ordinal":"Checking account health",
    "stability_score":"Life stability score","employment_stability":"Job stability",
    "own_realty_flag":"Property ownership","cc_utilization_mean":"Credit card utilization",
    "has_loan_history":"Loan history","credit_history_ordinal":"Credit history quality",
    "bureau_debt_to_credit_ratio":"Debt-to-credit ratio","inst_payment_consistency":"Payment consistency",
    "ext_source_mean":"External credit score","age_years":"Age","debt_ratio":"Debt ratio",
    "total_enquiries":"Credit enquiries","income_per_person":"Income per family member",
    "loan_purpose_high_risk":"Loan purpose risk",
}


# ── SINGLE SCORE ──────────────────────────────────────────────────
class ScoreRequest(BaseModel):
    is_new_user: bool = False
    annual_income: float = 480000
    loan_amount: float = 250000
    monthly_payment: float = 8500
    loan_term: int = 36
    dti: float = 25
    has_checking: bool = True
    checking_tier: int = 3
    has_savings: bool = True
    savings_amount: float = 25000
    savings_tier: int = 3
    age: int = 30
    family_size: int = 3
    employed: bool = True
    employment_years: float = 4
    employment_stability: int = 3
    owns_property: bool = False
    region_rating: int = 1
    high_risk_purpose: bool = False
    has_guarantor: bool = False
    existing_credits: int = 2
    cc_utilization: float = 0.30
    enquiries: int = 1
    late_payment_rate: float = 0.05
    max_days_late: int = 0
    has_past_delays: bool = False
    credit_history: int = 4
    approval_rate: float = 0.80
    prev_applications: int = 2
    docs_rate: float = 0.85

@app.post("/api/score")
def calculate_score(req: ScoreRequest, user=Depends(verify_token)):
    bundle = get_model()
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run model_training.py first.")

    model=bundle["model"]; explainer=bundle["explainer"]
    feature_names=bundle["feature_names"]; metrics=bundle["metrics"]

    t0    = time.perf_counter()
    X     = build_features(req.__dict__, feature_names)
    t_feat= time.perf_counter()-t0

    t1    = time.perf_counter()
    prob  = float(model.predict_proba(X)[0,1])
    t_mod = time.perf_counter()-t1

    t2        = time.perf_counter()
    shap_vals = explainer.shap_values(X)
    t_shap    = time.perf_counter()-t2

    sv=shap_vals[1][0] if isinstance(shap_vals,list) else shap_vals[0]
    pairs=sorted(zip(feature_names,sv),key=lambda x:abs(x[1]),reverse=True)
    pos,neg=[],[]
    for feat,val in pairs[:20]:
        if abs(val)<.001: continue
        label=LABELS.get(feat,feat.replace("_"," ").title())
        if val<0: pos.append(label)
        else:     neg.append(label)
        if len(pos)>=4 and len(neg)>=4: break

    score    = prob_to_score(prob)
    t_total  = time.perf_counter()-t0
    total_ms = t_total*1000
    ai,la    = req.annual_income,req.loan_amount

    # Print timing to terminal
    log_score(score, get_band(score), prob*100,
              t_feat, t_mod, t_shap, t_total,
              'NEW USER' if req.is_new_user else 'GENERAL',
              user.get('sub','unknown'))

    tips=[]
    if la/max(ai,1)>3:            tips.append(f"Reduce loan amount — ratio is {la/max(ai,1):.1f}x (target <3x)")
    if req.cc_utilization>.3:     tips.append(f"Lower CC utilization — {req.cc_utilization:.0%} (target <30%)")
    if req.late_payment_rate>.05: tips.append(f"Improve payment punctuality — {req.late_payment_rate:.1%} late (target <5%)")
    if not req.has_savings:       tips.append("Open a savings account — strong positive signal")
    if req.enquiries>5:           tips.append(f"Reduce credit enquiries — {req.enquiries} detected (target <5)")

    return {
        "score":score,"band":get_band(score),"prob_default":round(prob*100,2),
        "recommendation":get_rec(score),
        "positive_factors":pos[:4],"negative_factors":neg[:4],
        "improvement_tips":tips,
        "timing":{"total_ms":round(total_ms,1),"feature_ms":round(t_feat*1000,2),
                  "model_ms":round(t_mod*1000,2),"shap_ms":round(t_shap*1000,1)},
        "key_metrics":{"credit_to_income":round(la/max(ai,1),2),
                       "payment_burden":round(req.monthly_payment*12/max(ai,1)*100,1),
                       "cc_utilization":round(req.cc_utilization*100,1),
                       "late_payment_rate":round(req.late_payment_rate*100,1)},
        "model_metrics":metrics,
    }


# ── BATCH SCORE ───────────────────────────────────────────────────
@app.post("/api/batch-score")
async def batch_score(file: UploadFile = File(...), user=Depends(verify_token)):
    bundle = get_model()
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    model=bundle["model"]; feature_names=bundle["feature_names"]

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    if len(df) > 500:
        raise HTTPException(status_code=400, detail="Max 500 rows per batch.")

    df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]

    scores,bands,probs,recs,times,statuses=[],[],[],[],[],[]
    t_batch = time.perf_counter()

    for _,row in df.iterrows():
        t_row = time.perf_counter()
        try:
            X    = build_features(row.to_dict(), feature_names)
            prob = float(model.predict_proba(X)[0,1])
            s    = prob_to_score(prob)
            scores.append(s);  bands.append(get_band(s))
            probs.append(round(prob*100,2)); recs.append(get_rec(s))
            times.append(round((time.perf_counter()-t_row)*1000,1))
            statuses.append("success")
        except Exception as e:
            scores.append(None); bands.append(None)
            probs.append(None);  recs.append("Error")
            times.append(None);  statuses.append(f"error: {e}")

    total_ms = round((time.perf_counter()-t_batch)*1000,1)
    ok_times  = [t for t in times if t is not None]
    avg_ms    = sum(ok_times)/len(ok_times) if ok_times else 0
    n_ok      = statuses.count("success")

    # Print batch timing to terminal
    log_batch(len(df), n_ok, total_ms, avg_ms)

    df["credit_score"]              = scores
    df["score_band"]                = bands
    df["default_probability_pct"]   = probs
    df["recommendation"]            = recs
    df["scoring_time_ms"]           = times
    df["status"]                    = statuses

    ok = df[df["status"]=="success"]
    summary = pd.DataFrame([{
        col: "" for col in df.columns} | {
        "credit_score":            f"=== SUMMARY: {len(ok)}/{len(df)} scored ===",
        "score_band":              f"Avg score: {ok['credit_score'].mean():.0f}" if len(ok) else "N/A",
        "default_probability_pct": f"Avg default prob: {ok['default_probability_pct'].mean():.1f}%" if len(ok) else "N/A",
        "recommendation":          f"Approve:{(ok['recommendation']=='Approve').sum()} | Conditional:{(ok['recommendation']=='Conditional').sum()} | Reject:{(ok['recommendation']=='Reject').sum()}",
        "scoring_time_ms":         f"Total batch: {total_ms}ms | Avg/row: {ok['scoring_time_ms'].mean():.1f}ms" if len(ok) else "N/A",
    }])

    out = pd.concat([df, summary], ignore_index=True)
    buf = io.BytesIO()
    out.to_csv(buf, index=False)
    buf.seek(0)

    return StreamingResponse(buf, media_type="text/csv",
        headers={"Content-Disposition":
                 f"attachment; filename=zenith_scores_{int(time.time())}.csv"})


# ── TEMPLATE DOWNLOAD ─────────────────────────────────────────────
@app.get("/api/batch-template")
def download_template(user=Depends(verify_token)):
    sample = pd.DataFrame([
        # Row 1 — Good existing user
        dict(annual_income=480000,loan_amount=250000,monthly_payment=8500,loan_term=36,dti=25,
             has_checking=True,checking_tier=3,has_savings=True,savings_amount=25000,savings_tier=3,
             age=30,family_size=3,employed=True,employment_years=4,employment_stability=3,
             owns_property=False,region_rating=1,high_risk_purpose=False,has_guarantor=False,
             existing_credits=2,cc_utilization=0.30,enquiries=1,late_payment_rate=0.05,
             max_days_late=0,has_past_delays=False,credit_history=4,approval_rate=0.80,
             prev_applications=2,docs_rate=0.85,is_new_user=False),
        # Row 2 — Excellent existing user
        dict(annual_income=720000,loan_amount=180000,monthly_payment=6000,loan_term=36,dti=15,
             has_checking=True,checking_tier=4,has_savings=True,savings_amount=80000,savings_tier=4,
             age=42,family_size=4,employed=True,employment_years=12,employment_stability=4,
             owns_property=True,region_rating=1,high_risk_purpose=False,has_guarantor=False,
             existing_credits=1,cc_utilization=0.15,enquiries=0,late_payment_rate=0.0,
             max_days_late=0,has_past_delays=False,credit_history=4,approval_rate=1.0,
             prev_applications=3,docs_rate=1.0,is_new_user=False),
        # Row 3 — High risk
        dict(annual_income=240000,loan_amount=320000,monthly_payment=12000,loan_term=36,dti=55,
             has_checking=True,checking_tier=1,has_savings=False,savings_amount=0,savings_tier=0,
             age=25,family_size=2,employed=True,employment_years=1,employment_stability=1,
             owns_property=False,region_rating=3,high_risk_purpose=True,has_guarantor=False,
             existing_credits=5,cc_utilization=0.85,enquiries=8,late_payment_rate=0.30,
             max_days_late=90,has_past_delays=True,credit_history=1,approval_rate=0.40,
             prev_applications=6,docs_rate=0.60,is_new_user=False),
        # Row 4 — New user with good savings
        dict(annual_income=360000,loan_amount=120000,monthly_payment=4000,loan_term=36,dti=20,
             has_checking=True,checking_tier=3,has_savings=True,savings_amount=30000,savings_tier=3,
             age=27,family_size=1,employed=True,employment_years=3,employment_stability=2,
             owns_property=False,region_rating=1,high_risk_purpose=False,has_guarantor=False,
             existing_credits=0,cc_utilization=0.20,enquiries=1,late_payment_rate=0.0,
             max_days_late=0,has_past_delays=False,credit_history=4,approval_rate=0.5,
             prev_applications=0,docs_rate=0.90,is_new_user=True),
        # Row 5 — Very good with guarantor
        dict(annual_income=600000,loan_amount=200000,monthly_payment=7000,loan_term=36,dti=22,
             has_checking=True,checking_tier=4,has_savings=True,savings_amount=55000,savings_tier=3,
             age=35,family_size=3,employed=True,employment_years=8,employment_stability=4,
             owns_property=True,region_rating=1,high_risk_purpose=False,has_guarantor=True,
             existing_credits=2,cc_utilization=0.25,enquiries=2,late_payment_rate=0.02,
             max_days_late=0,has_past_delays=False,credit_history=4,approval_rate=0.90,
             prev_applications=2,docs_rate=0.95,is_new_user=False),
    ])
    buf = io.BytesIO()
    sample.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(buf, media_type="text/csv",
        headers={"Content-Disposition":"attachment; filename=zenith_batch_template.csv"})


@app.get("/api/health")
def health():
    bundle=get_model()
    return {"status":"ok","model_loaded":bundle is not None,
            "metrics":bundle["metrics"] if bundle else {}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
