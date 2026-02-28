# ZENITH — Alternative Credit Scoring System

> Score every applicant fairly — including those with no prior loan history — using 80 engineered features trained on 1.8 million real loan records.

---

## 🏗️ Architecture

```
zenith/
├── backend/          ← FastAPI + LightGBM model
│   ├── main.py       ← REST API with JWT auth
│   └── requirements.txt
├── frontend/         ← React UI
│   ├── src/
│   │   ├── pages/    ← Login, Home, Calculate, Result
│   │   ├── components/
│   │   └── api.js    ← Axios API layer
│   └── package.json
├── src/              ← ML pipeline scripts
│   ├── download_data.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── preprocessing.py
│   └── model_training.py
├── config.py
├── requirements.txt
└── README.md
```

---

##  Quick Start

### 1 — Clone
```bash
git clone https://github.com/YOUR_USERNAME/zenith-credit-scoring.git
cd zenith-credit-scoring
```

### 2 — Train the Model (run once)
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Download datasets manually from Kaggle into data/ folder
# Then run the ML pipeline:
python src/feature_engineering.py
python src/preprocessing.py
python src/model_training.py
# → saves models/credit_score_model.pkl
```

### 3 — Start Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# API running at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 4 — Start Frontend
```bash
cd frontend
npm install
npm start
# UI running at http://localhost:3000
```

---

##  Demo Credentials

| Email | Password | Role |
|---|---|---|
| demo@zenith.com | demo123 | Viewer |
| analyst@zenith.com | analyst123 | Analyst |
| admin@zenith.com | admin123 | Admin |

---

## Model Performance

| Metric | Value | Interpretation |
|---|---|---|
| AUC | 0.80+ | Excellent separation of defaulters |
| Gini | 0.60+ | 60% better than random |
| KS | 0.40+ | Strong score separation |

---

## Datasets Used

| Dataset | Rows | Source |
|---|---|---|
| Home Credit Default Risk | 307,511 | Kaggle |
| Give Me Some Credit | 150,000 | Kaggle |
| Lending Club | 1,348,132 | Kaggle |
| PKDD Czech Financial | 682 | Kaggle |
| **Total** | **1,806,325** | |

---

## Feature Categories (80 total)

1. Income & Loan (10) — affordability ratios
2. Alternative Signals (14) — savings, bank account, external scores
3. Payment Behaviour (16) — installment, POS, credit card patterns
4. Credit History (13) — bureau, previous applications
5. Stability & Demographics (16) — employment, age, property
6. Credit Utilisation (8) — CC utilization, enquiries
7. Loan Structure (7) — purpose, guarantor, contract type

---

## New User Mode

Users with zero loan history are scored using alternative signals only:
- Bank account balance tier
- Savings account level
- Employment stability (0–4 scale)
- Payment consistency from bank transactions
- Document submission rate

LightGBM handles missing loan-history features natively via learned default bin directions — new users are never penalised for the absence of loan history.

---

## Tech Stack

**ML Pipeline:** Python, LightGBM, SHAP, scikit-learn, pandas  
**Backend:** FastAPI, JWT auth, Pydantic, Uvicorn  
**Frontend:** React 18, React Router v6, Framer Motion, CSS Modules  
**Charts:** Canvas API (gauge), Recharts  

---

## 👥 Team

Built as an alternative credit scoring prototype for financial inclusion — enabling creditworthy individuals without traditional loan histories to access fair credit assessments.
