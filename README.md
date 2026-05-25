# Cost-Aware Churn Prediction

Churn prediction optimized for business value, not raw accuracy.

> **Status:** in progress. Live demo link, key findings, and the full write-up will appear here once the project completes.

## The problem

Banks lose existing customers at a steady rate. Retaining a customer is cheaper than acquiring a new one, but retention offers cost money, success rates are imperfect, and not every churner is worth saving. Most predictive churn models stop at "who is likely to leave?" They optimize ROC AUC, pick the default 0.5 threshold, and call it done.

That answer is incomplete. The right decision depends on dollars: how much a retention offer costs, how much a saved customer is worth, and how often an offer actually works. With those numbers in hand, a different threshold maximizes expected value. With them honestly wrong, a near-optimal threshold can still be recoverable — sensitivity analysis tells you which assumptions actually matter.

This project is built around that framing.

## Approach

- Train a churn classifier on 10,127 credit-card customers from the BankChurners dataset (~16% churn rate).
- Build a cost matrix that maps every prediction outcome (true positive, false positive, false negative, true negative) to its dollar consequence.
- Sweep the decision threshold and find the one that maximizes total expected value. Compare it to the default-threshold and naive blanket-offer baselines.
- Run a sensitivity analysis on every cost assumption (offer cost, customer LTV, offer success rate) to test how robust the recommendation is to changes in the inputs.
- Wrap the result in an interactive Streamlit dashboard where viewers can move the assumptions and watch the optimum shift.

## Dataset

[BankChurners](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) — 10,127 credit-card customers, 23 columns. Two `Naive_Bayes_Classifier_*` columns are artifacts from the original uploader and are dropped on load.

The raw data is not committed to this repository. To reproduce: download `BankChurners.csv` from the link above and place it at `data/raw/BankChurners.csv`.

## Project structure

```
.
├── app/                       # Streamlit dashboard
├── data/
│   ├── raw/                   # Raw dataset (gitignored, download separately)
│   └── processed/             # Derived data used by the dashboard
├── models/                    # Trained model artifacts
├── notebooks/
│   ├── figures/               # Saved plots for the README and dashboard
│   ├── 01_eda.ipynb           # Exploratory data analysis (Phase 2)
│   ├── 02_modeling.ipynb      # Modeling and interpretation (Phase 4)
│   └── 03_cost_analysis.ipynb # Cost analysis and sensitivity (Phase 5)
├── src/                       # Reusable Python code
├── requirements.txt
└── README.md
```

## How to run locally

```bash
git clone https://github.com/mkhlndrv/cost-aware-churn-prediction.git
cd cost-aware-churn-prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Download BankChurners.csv from Kaggle and place it at data/raw/
jupyter lab
```

The Streamlit dashboard will be runnable with `streamlit run app/app.py` once Phase 6 lands.

## What's next

- [ ] EDA notebook
- [ ] Modeling notebook (baseline + XGBoost + SHAP)
- [ ] Cost analysis notebook
- [ ] Streamlit dashboard
- [ ] Live deployment + final business-memo README
