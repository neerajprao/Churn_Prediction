import sys
import os
import joblib
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"BASE_DIR: {BASE_DIR}")

print("Loading XGBoost...", flush=True)
from xgboost import XGBClassifier
xgb_model = XGBClassifier()
# This matches app.py exactly where BASE_DIR in app.py is .../front_end
xgb_model.load_model(os.path.join(BASE_DIR, 'front_end', 'models', 'xgboost_model.json'))
print("XGBoost success", flush=True)
