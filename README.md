# Customer Purchase Frequency Prediction Website

A Streamlit web app for single-customer prediction, batch CSV prediction, model training/testing, and dataset exploration.

## Files
- `app.py` — website application
- `purchase_frequency_model.pkl` — your supplied model (copy it into this folder)
- `requirements.txt` — Python packages
- `customer_prediction_template.csv` — example batch input

## Run locally (Windows)
1. Install Python 3.10 or newer.
2. Put `app.py`, `requirements.txt`, and `purchase_frequency_model.pkl` in the same folder.
3. Open Command Prompt/PowerShell in that folder.
4. Run:

   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   python -m pip install -r requirements.txt
   streamlit run app.py
   ```

5. The app opens in your browser, usually at `http://localhost:8501`.

## Run locally (macOS/Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Training data format
Upload a CSV where each row has:
- two numeric feature columns (by default `PreviousPurchases` and `DaysSinceLastPurchase`)
- one numeric target column representing actual purchases during the next 30 days

The training tab allows you to select feature and target columns. It creates a random train/test split and reports MAE, RMSE, and R² on held-out test data.

## Notes
- The supplied pickle model was serialized with a different scikit-learn version than the environment used during development. Prediction was checked in the development environment, but for long-term reliability, retrain and save the model using the same scikit-learn version listed in your environment.
- Only load pickle/joblib model files from sources you trust. They can execute code when deserialized.
- This is a local development app, not a production-hardened multi-user service.
