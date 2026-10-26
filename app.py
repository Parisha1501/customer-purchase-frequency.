import io
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_DIR / "purchase_frequency_model.pkl"
FEATURES = ["PreviousPurchases", "DaysSinceLastPurchase"]
TARGET_DEFAULT = "Next30DaysPurchases"

st.set_page_config(
    page_title="Customer Purchase Frequency",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Simple, clean styling
st.markdown("""
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2.5rem;}
    .hero {padding: 1.3rem 1.5rem; border-radius: 16px; background: linear-gradient(120deg,#17324d,#246b75); color: white; margin-bottom: 1.2rem;}
    .hero h1 {color: white; margin: 0 0 .35rem 0;}
    .hero p {color: #e8f4f6; margin: 0;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); padding: 12px 16px; border-radius: 12px;}
    .small-note {font-size: .88rem; opacity: .8;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_default_model(path_string, modified_time):
    # Only load the supplied local model file; don't load arbitrary model files from CSV uploads.
    return joblib.load(path_string)

def get_default_model():
    if not DEFAULT_MODEL_PATH.exists():
        return None, "No default model file found beside app.py."
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = load_default_model(str(DEFAULT_MODEL_PATH), DEFAULT_MODEL_PATH.stat().st_mtime)
        warning_text = " ".join(str(w.message) for w in caught)
        return model, warning_text
    except Exception as exc:
        return None, f"Could not load the saved model: {exc}"

def normalize_prediction_model(model):
    """Return a model-compatible dataframe, respecting known feature names."""
    expected = list(getattr(model, "feature_names_in_", FEATURES))
    missing = [c for c in expected if c not in FEATURES]
    if missing:
        raise ValueError(
            "The loaded model expects different feature columns: "
            + ", ".join(expected)
            + ". This website is configured for "
            + ", ".join(FEATURES)
            + ". Retrain a model using the training tab."
        )
    return expected

def predict_values(model, frame):
    expected = normalize_prediction_model(model)
    X = frame[expected].copy()
    predictions = np.asarray(model.predict(X), dtype=float).reshape(-1)
    # Purchase counts cannot be negative; retain the raw value separately for transparency.
    return predictions, np.maximum(0, predictions)

def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")

def safe_numeric_dataset(df, features, target=None):
    needed = list(features) + ([target] if target else [])
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise ValueError("Missing columns: " + ", ".join(missing_cols))
    clean = df[needed].copy()
    for c in needed:
        clean[c] = pd.to_numeric(clean[c], errors="coerce")
    before = len(clean)
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
    dropped = before - len(clean)
    if clean.empty:
        raise ValueError("No usable rows remain after converting selected columns to numbers and removing missing values.")
    return clean, dropped

model, model_warning = get_default_model()
if "active_model" not in st.session_state:
    st.session_state.active_model = model
    st.session_state.model_source = "Provided model" if model is not None else "No model loaded"
if "training_results" not in st.session_state:
    st.session_state.training_results = None

st.markdown("""
<div class="hero">
  <h1>🛍️ Customer Purchase Frequency</h1>
  <p>Predict future purchasing activity, test model quality, and train a model on your own transaction data.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Model status")
    active_model = st.session_state.active_model
    if active_model is not None:
        st.success(f"Active model: {st.session_state.model_source}")
        st.caption(f"Model type: {type(active_model).__name__}")
    else:
        st.warning("No model loaded. Use the Training & Testing tab to train one.")
    if model_warning:
        st.caption("Model compatibility note: " + model_warning)
    st.divider()
    st.caption("Expected prediction features")
    st.code("PreviousPurchases\nDaysSinceLastPurchase")
    st.caption("Prediction horizon: next 30 days")

tab_predict, tab_batch, tab_train, tab_data, tab_about = st.tabs([
    "🔮 Predict", "📂 Batch prediction", "🧪 Training & testing", "📊 Data explorer", "ℹ️ Guide"
])

with tab_predict:
    st.subheader("Predict for one customer")
    st.write("Enter a customer's recent purchase activity. The estimate is the number of purchases expected over the next 30 days.")
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        with st.form("single_prediction_form"):
            previous_purchases = st.number_input(
                "Purchases in the previous 90 days", min_value=0, max_value=500, value=3, step=1
            )
            days_since_last_purchase = st.number_input(
                "Days since the last purchase", min_value=0, max_value=1000, value=10, step=1
            )
            submitted = st.form_submit_button("Predict purchase frequency", type="primary", use_container_width=True)
    with right:
        st.info("The model uses `PreviousPurchases` and `DaysSinceLastPurchase`. Estimates are model outputs, not guarantees.")
        st.markdown("**Example input**")
        st.dataframe(pd.DataFrame([{
            "PreviousPurchases": 3,
            "DaysSinceLastPurchase": 10
        }]), hide_index=True, use_container_width=True)
    if submitted:
        active_model = st.session_state.active_model
        if active_model is None:
            st.error("There is no active model. Train one in the Training & testing tab or place purchase_frequency_model.pkl next to app.py.")
        else:
            try:
                one = pd.DataFrame([{
                    "PreviousPurchases": previous_purchases,
                    "DaysSinceLastPurchase": days_since_last_purchase,
                }])
                raw, nonnegative = predict_values(active_model, one)
                estimate = float(nonnegative[0])
                st.markdown("### Prediction result")
                a, b, c = st.columns(3)
                a.metric("Estimated purchases (next 30 days)", f"{round(estimate):,}")
                b.metric("Model estimate", f"{raw[0]:.2f}")
                c.metric("Recent purchases (90 days)", f"{previous_purchases:,}")
                st.caption("The displayed whole-number estimate is rounded from the model output. Negative estimates are floored at zero for reporting.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with tab_batch:
    st.subheader("Predict for many customers")
    st.write("Upload a CSV with the two feature columns. Extra columns are preserved in the output.")
    template = pd.DataFrame({
        "CustomerID": ["C001", "C002", "C003"],
        "PreviousPurchases": [3, 8, 1],
        "DaysSinceLastPurchase": [10, 4, 45],
    })
    st.download_button("Download example CSV", csv_bytes(template), "customer_prediction_template.csv", "text/csv")
    uploaded_batch = st.file_uploader("Upload customer CSV", type=["csv"], key="batch_csv")
    if uploaded_batch:
        try:
            batch_df = pd.read_csv(uploaded_batch)
            st.write(f"Rows: **{len(batch_df):,}** · Columns: **{len(batch_df.columns)}**")
            st.dataframe(batch_df.head(10), use_container_width=True)
            if all(c in batch_df.columns for c in FEATURES):
                if st.button("Run batch predictions", type="primary"):
                    if st.session_state.active_model is None:
                        st.error("No active model. Train or load one first.")
                    else:
                        numeric = batch_df.copy()
                        for c in FEATURES:
                            numeric[c] = pd.to_numeric(numeric[c], errors="coerce")
                        invalid = numeric[FEATURES].isna().any(axis=1)
                        if invalid.any():
                            st.warning(f"{int(invalid.sum())} row(s) have missing or non-numeric feature values. They will have blank predictions.")
                        valid = ~invalid
                        result = batch_df.copy()
                        result["PredictedPurchasesNext30Days"] = np.nan
                        result["RawModelEstimate"] = np.nan
                        if valid.any():
                            raw, clipped = predict_values(st.session_state.active_model, numeric.loc[valid, FEATURES])
                            result.loc[valid, "RawModelEstimate"] = raw
                            result.loc[valid, "PredictedPurchasesNext30Days"] = np.rint(clipped).astype(int)
                        st.session_state.batch_result = result
            else:
                st.error("Required columns not found. The CSV must include: " + ", ".join(FEATURES))
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
    if "batch_result" in st.session_state:
        st.markdown("### Results")
        st.dataframe(st.session_state.batch_result.head(100), use_container_width=True)
        st.download_button(
            "Download predictions CSV",
            csv_bytes(st.session_state.batch_result),
            "customer_purchase_predictions.csv",
            "text/csv",
            type="primary",
        )

with tab_train:
    st.subheader("Train and test a regression model")
    st.write("Upload a labeled historical dataset. Select two input features and the column that records the actual number of purchases in the following 30 days.")
    st.warning("For a fair test, the model is evaluated on a held-out test split that is not used to fit the model. Your dataset needs a target column containing actual future purchase counts.")
    training_file = st.file_uploader("Upload historical training CSV", type=["csv"], key="training_csv")
    if training_file:
        try:
            raw_df = pd.read_csv(training_file)
            st.write(f"Dataset: **{len(raw_df):,} rows × {len(raw_df.columns)} columns**")
            st.dataframe(raw_df.head(8), use_container_width=True)
            columns = list(raw_df.columns)
            default_feature_indices = [columns.index(c) for c in FEATURES if c in columns]
            f1_default = columns.index("PreviousPurchases") if "PreviousPurchases" in columns else 0
            f2_default = columns.index("DaysSinceLastPurchase") if "DaysSinceLastPurchase" in columns else min(1, len(columns)-1)
            target_default_index = columns.index(TARGET_DEFAULT) if TARGET_DEFAULT in columns else len(columns)-1
            choices = st.columns(3)
            with choices[0]:
                feature_1 = st.selectbox("Feature 1", columns, index=f1_default)
            with choices[1]:
                feature_2 = st.selectbox("Feature 2", columns, index=f2_default)
            with choices[2]:
                target_col = st.selectbox("Target (actual future purchases)", columns, index=target_default_index)
            test_size = st.slider("Test data percentage", min_value=10, max_value=40, value=20, step=5)
            random_seed = st.number_input("Random seed (reproducible split)", min_value=0, max_value=999999, value=42, step=1)
            if feature_1 == feature_2:
                st.error("Choose two different feature columns.")
            elif target_col in [feature_1, feature_2]:
                st.error("The target must be different from both feature columns.")
            elif st.button("Train model and evaluate", type="primary"):
                try:
                    cleaned, dropped = safe_numeric_dataset(raw_df, [feature_1, feature_2], target_col)
                    if len(cleaned) < 5:
                        raise ValueError("At least 5 valid rows are needed for a train/test split.")
                    X = cleaned[[feature_1, feature_2]].copy()
                    X.columns = FEATURES  # Standardize names so the trained model works in the prediction tabs.
                    y = cleaned[target_col]
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size/100, random_state=int(random_seed)
                    )
                    trained = LinearRegression()
                    trained.fit(X_train, y_train)
                    y_pred = trained.predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                    r2 = r2_score(y_test, y_pred) if len(y_test) >= 2 else float("nan")
                    test_results = pd.DataFrame({
                        "ActualPurchases": y_test.to_numpy(),
                        "PredictedRaw": y_pred,
                        "PredictedNonNegativeRounded": np.rint(np.maximum(0, y_pred)).astype(int),
                        "AbsoluteError": np.abs(y_test.to_numpy() - y_pred),
                    }, index=y_test.index).sort_index()
                    st.session_state.training_results = {
                        "model": trained, "features": FEATURES.copy(), "source_features": [feature_1, feature_2], "target": target_col,
                        "mae": mae, "rmse": rmse, "r2": r2, "rows": len(cleaned),
                        "dropped": dropped, "train_rows": len(X_train), "test_rows": len(X_test),
                        "test_results": test_results, "X_test": X_test, "y_test": y_test, "y_pred": y_pred,
                    }
                    st.session_state.active_model = trained
                    st.session_state.model_source = "Newly trained model"
                    st.success("Training and evaluation completed. The new model is now active.")
                except Exception as exc:
                    st.error(f"Training failed: {exc}")
        except Exception as exc:
            st.error(f"Could not read training CSV: {exc}")
    result = st.session_state.training_results
    if result:
        st.markdown("### Test-set performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", f"{result['mae']:.3f}")
        m2.metric("RMSE", f"{result['rmse']:.3f}")
        m3.metric("R²", f"{result['r2']:.3f}" if np.isfinite(result["r2"]) else "N/A")
        m4.metric("Test rows", f"{result['test_rows']:,}")
        st.caption(f"Valid rows: {result['rows']:,} · Rows removed for missing/non-numeric values: {result['dropped']:,} · Training rows: {result['train_rows']:,}")
        st.markdown("**How to read these metrics:** lower MAE/RMSE usually means smaller prediction errors; R² closer to 1 generally indicates a better fit. A negative R² means the model performs worse than predicting the test-set mean.")
        st.markdown("### Actual vs predicted on test data")
        plot_df = result["test_results"]
        fig, ax = plt.subplots()
        ax.scatter(plot_df["ActualPurchases"], plot_df["PredictedRaw"], alpha=0.75)
        low = min(plot_df["ActualPurchases"].min(), plot_df["PredictedRaw"].min())
        high = max(plot_df["ActualPurchases"].max(), plot_df["PredictedRaw"].max())
        ax.plot([low, high], [low, high], linestyle="--")
        ax.set_xlabel("Actual purchases")
        ax.set_ylabel("Predicted purchases")
        ax.set_title("Held-out test set")
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(plot_df, use_container_width=True)
        st.download_button("Download test-set results", csv_bytes(plot_df.reset_index(names="RowIndex")), "model_test_results.csv", "text/csv")
        model_bytes = io.BytesIO()
        joblib.dump(result["model"], model_bytes)
        st.download_button("Download trained model (.pkl)", model_bytes.getvalue(), "purchase_frequency_model_trained.pkl", "application/octet-stream", type="primary")
        st.info("The model is trained with the standard prediction feature names. Your uploaded CSV columns can have different names; the training tab maps the selected columns to the standard names for later prediction.")

with tab_data:
    st.subheader("Explore a dataset")
    st.write("Upload any CSV to inspect data quality and numeric patterns.")
    explorer_file = st.file_uploader("Upload CSV for exploration", type=["csv"], key="explorer_csv")
    if explorer_file:
        try:
            data = pd.read_csv(explorer_file)
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", f"{data.shape[0]:,}")
            c2.metric("Columns", f"{data.shape[1]:,}")
            c3.metric("Duplicate rows", f"{int(data.duplicated().sum()):,}")
            st.markdown("### Preview")
            st.dataframe(data.head(50), use_container_width=True)
            st.markdown("### Missing values")
            missing = data.isna().sum().rename("MissingValues").to_frame()
            missing["MissingPercent"] = (missing["MissingValues"] / max(len(data), 1) * 100).round(2)
            st.dataframe(missing, use_container_width=True)
            st.markdown("### Summary statistics")
            st.dataframe(data.describe(include="all").transpose(), use_container_width=True)
            numeric_data = data.select_dtypes(include=np.number)
            if not numeric_data.empty:
                st.markdown("### Numeric correlation")
                st.dataframe(numeric_data.corr(numeric_only=True).round(3), use_container_width=True)
                col = st.selectbox("Choose numeric column to visualize", list(numeric_data.columns))
                fig, ax = plt.subplots()
                ax.hist(numeric_data[col].dropna(), bins=25)
                ax.set_title(f"Distribution: {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Rows")
                st.pyplot(fig)
                plt.close(fig)
        except Exception as exc:
            st.error(f"Could not explore this file: {exc}")

with tab_about:
    st.subheader("How to use this website")
    st.markdown("""
1. **Predict:** Use the existing `purchase_frequency_model.pkl` to estimate purchases for one customer.
2. **Batch prediction:** Upload a CSV containing `PreviousPurchases` and `DaysSinceLastPurchase`. Download the result CSV.
3. **Training & testing:** Upload historical labeled data, choose two input features and a target containing actual future purchases, then train and evaluate a fresh Linear Regression model.
4. **Data explorer:** Review dataset size, missing data, descriptive statistics, correlations and a numeric distribution.

### Important data requirement
For model training, each row should represent a customer at a historical prediction point. The target column must contain the **actual number of purchases in the following 30 days**. Do not use a target that includes information from after the prediction date in the input features, because that causes data leakage.

### Current model
The supplied model expects:
- `PreviousPurchases`: purchases in the previous 90 days
- `DaysSinceLastPurchase`: days since the last purchase

The supplied app predicts a 30-day purchase estimate and floors negative outputs at zero for display. The quality of the model depends on the data used to train it.
""")
