# Cost-Aware Churn Prediction

Churn prediction optimized for business value, not raw accuracy.

## The problem

Banks lose existing customers at a steady rate. Retaining a customer is cheaper than acquiring a new one, but retention offers cost money, success rates are imperfect, and not every churner is worth saving. Most predictive churn models stop at "who is likely to leave?" They optimize ROC AUC, pick the default 0.5 threshold, and call it done.

That answer is incomplete. The right decision depends on dollars: how much a retention offer costs, how much a saved customer is worth, and how often an offer actually works. With those numbers in hand, a different threshold maximizes expected value. With them honestly wrong, sensitivity analysis tells you which assumptions actually matter.

This project is built around that framing.

## Approach

- Train a churn classifier on 10,127 credit-card customers from the BankChurners dataset (16% churn rate): a class-weighted logistic regression baseline, then a tuned XGBoost pipeline, interpreted with SHAP.
- Build a cost matrix that maps every prediction outcome (true positive, false positive, false negative, true negative) to its dollar consequence, relative to doing nothing.
- Sweep the decision threshold across [0, 1] on the held-out test set and find the cutoff that maximizes total expected value. Compare it to default-threshold, do-nothing, and blanket-offer baselines, then recompute per customer segment.
- Run a sensitivity analysis on every cost assumption (offer cost, customer LTV, offer success rate) to test how robust the recommendation is.
- Wrap the result in an interactive Streamlit dashboard where viewers can move the assumptions and watch the optimum shift.

## Results

### The model

On the validation set (1,519 customers; the split is 70/15/15 stratified):

| model | ROC-AUC | PR-AUC | recall (churners) | precision (churners) |
|---|---|---|---|---|
| logistic regression, class-weighted | 0.916 | 0.709 | 0.836 | 0.528 |
| XGBoost, default | 0.995 | 0.976 | 0.926 | 0.919 |
| XGBoost, tuned | 0.995 | 0.978 | 0.934 | 0.905 |

Grid search (27 combinations, 5-fold CV) moved PR-AUC by 0.002; the model was near its ceiling before tuning. The tuned pipeline (300 trees, depth 3, learning rate 0.15) is what gets persisted and used downstream. SHAP attribution shows the model leans on engagement signals: transaction count, transaction amount, and revolving balance dominate, which matches the EDA finding that low-activity customers churn at many times the rate of high-activity ones.

### The decision

At the stated assumptions ($50 offer cost, $500 customer LTV, 30% offer success rate), a true positive is worth +$100 in expectation and a false positive costs $50. Sweeping the threshold on the held-out test set (1,520 customers):

| strategy | total EV | per 1,000 customers |
|---|---|---|
| optimal threshold (0.445) | +$21,050 | +$13,849 |
| default threshold (0.5) | +$20,500 | +$13,487 |
| do nothing | $0 | $0 |
| blanket offer to everyone | -$39,400 | -$25,921 |

Read honestly: running a targeted program at all is worth about $14k per 1,000 customers over doing nothing, while the cost-aware threshold adds only $550 over the default 0.5 at these midpoint assumptions. Blanket offering destroys value because false positives are not free. Per-segment thresholds (tenure or activity tiers) add at most a further 1.7%.

### The sensitivity analysis

The optimum is only as stable as the inputs. Across offer cost $20 to $100 the optimal threshold moves from 0.21 to 0.82; across LTV $200 to $1,000 it moves from 0.82 down to 0.21. The offer success rate matters most: below roughly 10%, where the expected recovered LTV (10% of $500) no longer covers the $50 offer, no threshold is profitable and the right decision is to not run the program. That input is also the one the dataset contains no evidence about, so it is the one to measure first (for example with a small randomized holdout) before deploying anything like this.

## Dataset

[BankChurners](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers): 10,127 credit-card customers, 23 columns. The `CLIENTNUM` identifier and two `Naive_Bayes_Classifier_*` columns (added by the original uploader and trained on the target itself) are dropped on load.

The raw data is not committed to this repository. To run the notebooks, download `BankChurners.csv` from the link above and place it at `data/raw/BankChurners.csv`.

## Project structure

```
.
├── app/app.py                 # Streamlit dashboard: cost sliders, EV curve, customer ranking
├── data/
│   ├── raw/                   # Raw dataset (gitignored, download separately)
│   └── processed/             # Held-out test set (committed, used by the dashboard)
├── models/                    # Tuned XGBoost pipeline (committed)
├── notebooks/
│   ├── figures/               # Saved plots
│   ├── 01_eda.ipynb           # EDA and customer segmentation
│   ├── 02_modeling.ipynb      # Baseline, tuned XGBoost, SHAP interpretation
│   └── 03_cost_analysis.ipynb # Cost matrix, threshold sweep, sensitivity analysis
├── requirements.txt
└── README.md
```

## How to run

```bash
git clone https://github.com/mkhlndrv/cost-aware-churn-prediction.git
cd cost-aware-churn-prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dashboard runs from the committed model and test set, no data download needed:

```bash
streamlit run app/app.py
```

The notebooks need the raw CSV (see Dataset above) and run in order: 01 EDA, 02 modeling (writes `models/xgb_tuned.joblib` and `data/processed/test_set.csv`), 03 cost analysis.

## Limitations

- The three dollar inputs are industry-benchmark assumptions, not measurements. The dataset has no record of past retention offers, so the 30% success rate in particular is borrowed, and the analysis shows it is the input the conclusion is most exposed to.
- At the midpoint assumptions the cost-aware optimum (0.445) lands close to the default 0.5 and is worth only $550 more on this test set. The framework earns its keep under other input values, not at the midpoint.
- LTV is assumed uniform across customers; segment-specific values would change the per-segment thresholds.
- The split is random and stratified, and the data is a single snapshot. A production model would need temporal validation before any of the dollar figures could be trusted.

License: MIT.
