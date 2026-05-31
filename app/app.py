"""Cost-Aware Churn Prediction — Streamlit dashboard."""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

st.set_page_config(
    page_title="Cost-Aware Churn Prediction",
    layout="wide",
)

plt.style.use("seaborn-v0_8-whitegrid")


# --- Data layer (cached) --------------------------------------------------

@st.cache_resource
def load_model():
    """Load the tuned XGBoost pipeline once and share it across sessions."""
    return joblib.load("models/xgb_tuned.joblib")


@st.cache_data
def load_test_set():
    """Test set augmented with tenure and activity segments from the Phase 3 EDA."""
    test = pd.read_csv("data/processed/test_set.csv")
    test["tenure_segment"] = pd.qcut(
        test["Months_on_book"], q=3, labels=["New", "Established", "Long-tenured"]
    )
    test["activity_segment"] = pd.qcut(
        test["Total_Trans_Ct"], q=3, labels=["Low", "Medium", "High"]
    )
    return test


@st.cache_data
def compute_probs():
    """Predicted churn probability for every customer in the test set."""
    model = load_model()
    test = load_test_set()
    feature_cols = [c for c in test.columns if c not in {"churned", "tenure_segment", "activity_segment"}]
    return model.predict_proba(test[feature_cols])[:, 1]


@st.cache_data
def build_threshold_sweep():
    """Confusion-matrix counts at every threshold from 0 to 1. Independent of cost assumptions."""
    y = load_test_set()["churned"].values
    probs = compute_probs()
    thresholds = np.linspace(0, 1, 1001)

    rows = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
        rows.append({"threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return pd.DataFrame(rows)


model = load_model()
test_set = load_test_set()
probs = compute_probs()
sweep = build_threshold_sweep()


# --- Helpers --------------------------------------------------------------

def build_cost_matrix(offer_cost, customer_ltv, offer_success_rate):
    """Expected dollar value of each confusion-matrix cell, relative to no intervention."""
    return {
        "TN": 0,
        "FP": -offer_cost,
        "FN": 0,
        "TP": offer_success_rate * customer_ltv - offer_cost,
    }


# --- Header ---------------------------------------------------------------
st.title("Cost-Aware Churn Prediction")
st.markdown(
    "Turn churn predictions into expected-value decisions. "
    "Move the cost assumptions in the sidebar to see how the optimal targeting threshold shifts."
)


# --- Sidebar (cost-assumption controls + segment filters) -----------------
st.sidebar.header("Cost assumptions")

offer_cost = st.sidebar.slider("Offer cost ($)", 20, 100, 50, 5)
st.sidebar.caption(
    "Per-customer cost of the retention offer — the incentive itself plus call-center, "
    "list, and channel overhead."
)

customer_ltv = st.sidebar.slider("Customer lifetime value ($)", 200, 1000, 500, 50)
st.sidebar.caption(
    "Net present value of keeping a customer who would otherwise churn — projected "
    "revenue minus servicing cost over their expected remaining tenure."
)

success_rate_pct = st.sidebar.slider("Offer success rate (%)", 10, 50, 30, 1)
offer_success_rate = success_rate_pct / 100
st.sidebar.caption(
    "Probability that a retention offer actually prevents a customer who was going to churn from leaving."
)

st.sidebar.divider()
st.sidebar.subheader("Filter customer table")

tenure_filter = st.sidebar.selectbox(
    "Tenure segment", ["All", "New", "Established", "Long-tenured"]
)
activity_filter = st.sidebar.selectbox(
    "Activity segment", ["All", "Low", "Medium", "High"]
)

st.sidebar.divider()
st.sidebar.caption(f"Loaded {len(test_set):,} test customers and tuned XGBoost model.")


# --- Live computation -----------------------------------------------------

cost_matrix = build_cost_matrix(offer_cost, customer_ltv, offer_success_rate)
sweep_ev = sweep["tp"] * cost_matrix["TP"] + sweep["fp"] * cost_matrix["FP"]

optimal_idx = sweep_ev.idxmax()
optimal_threshold = float(sweep.loc[optimal_idx, "threshold"])
optimal_ev = float(sweep_ev[optimal_idx])

# Per-customer expected value of intervening at current assumptions.
intervention_ev_per_customer = probs * (offer_success_rate * customer_ltv) - offer_cost


# --- Main panel: Headline metrics ----------------------------------------

st.header("Headline metrics")
metric_cols = st.columns(3)
metric_cols[0].metric("Optimal threshold", f"{optimal_threshold:.3f}")
metric_cols[1].metric("Expected value at optimum", f"${optimal_ev:,.0f}")
metric_cols[2].metric(
    "Gap vs do-nothing",
    f"${optimal_ev:,.0f}",
    help="Total expected dollar value the cost-aware targeting program adds over not running it at all.",
)


# --- Main panel: EV curve -------------------------------------------------

st.header("Expected value by decision threshold")
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(sweep["threshold"], sweep_ev, linewidth=2)
ax.axhline(0, color="gray", linestyle=":", linewidth=1, label="do-nothing baseline ($0)")
ax.axvline(
    optimal_threshold, color="#2ca02c", linestyle="--", linewidth=1.5,
    label=f"optimal threshold = {optimal_threshold:.3f}",
)
ax.axvline(0.5, color="#d62728", linestyle="--", linewidth=1.5, label="default threshold = 0.5")
ax.set_xlabel("Decision threshold (predicted churn probability cutoff)")
ax.set_ylabel("Total expected value on test set ($)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend(loc="lower right")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)


# --- Main panel: Customer ranking ----------------------------------------

st.header("Top customers by intervention value")
st.markdown(
    "Customers ranked by the expected dollar value of intervening on them at the current "
    "assumptions. Filter by segment from the sidebar."
)

customer_view = test_set.copy()
customer_view["predicted_churn_prob"] = probs
customer_view["intervention_ev"] = intervention_ev_per_customer

if tenure_filter != "All":
    customer_view = customer_view[customer_view["tenure_segment"] == tenure_filter]
if activity_filter != "All":
    customer_view = customer_view[customer_view["activity_segment"] == activity_filter]

customer_view = customer_view.sort_values("intervention_ev", ascending=False).reset_index(drop=True)

display_cols = [
    "predicted_churn_prob",
    "intervention_ev",
    "Customer_Age",
    "Months_on_book",
    "Total_Trans_Ct",
    "Card_Category",
    "tenure_segment",
    "activity_segment",
]

st.dataframe(
    customer_view[display_cols].head(25),
    column_config={
        "predicted_churn_prob": st.column_config.NumberColumn("Predicted churn prob", format="%.3f"),
        "intervention_ev": st.column_config.NumberColumn("Intervention EV ($)", format="$%.2f"),
        "Customer_Age": "Age",
        "Months_on_book": "Tenure (mo.)",
        "Total_Trans_Ct": "Transactions (12 mo.)",
        "Card_Category": "Card tier",
        "tenure_segment": "Tenure tier",
        "activity_segment": "Activity tier",
    },
    hide_index=True,
)

n_positive = int((customer_view["intervention_ev"] > 0).sum())
st.caption(
    f"{len(customer_view):,} customers match current filters; "
    f"{n_positive:,} have positive intervention EV at the current assumptions."
)


# --- Footer (assumptions and disclosures) ---------------------------------

st.markdown("---")
with st.expander("About the cost assumptions"):
    st.markdown(
        f"**Currently displayed values:** offer cost \\${offer_cost}, "
        f"customer LTV \\${customer_ltv}, offer success rate {success_rate_pct}%.\n\n"
        "The dashboard uses three inputs: **offer cost** (per attempted save), **customer lifetime "
        "value** (per successful save), and **offer success rate** (probability an offer prevents a "
        "churn). Midpoint defaults are drawn from credit-card retention industry benchmarks; the "
        "values themselves are not measurements from the dataset and should be treated as illustrative."
    )
