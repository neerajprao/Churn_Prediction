import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from xgboost import XGBClassifier
from lightgbm import Booster
from catboost import CatBoostClassifier

# ---------------- Load ML Models ----------------
xgb_model = XGBClassifier()
xgb_model.load_model('/Users/neerajprao/Downloads/ML-vs-DL-main/front_end/models/xgboost_model.json')

rf_model = joblib.load('/Users/neerajprao/Downloads/ML-vs-DL-main/front_end/models/random_forest.pkl')

lgbm_booster = Booster(model_file='/Users/neerajprao/Downloads/ML-vs-DL-main/front_end/models/lightgbm_model.txt')
lgbm_model = lgbm_booster

catboost_model = CatBoostClassifier()
catboost_model.load_model('/Users/neerajprao/Downloads/ML-vs-DL-main/front_end/models/catboost_model.cbm')

# ---------------- Define DL Model Architectures ----------------
class NodeModel(nn.Module):
    def __init__(self):
        super(NodeModel, self).__init__()
        self.tree_layer = nn.ModuleDict({
            'hidden_layers': nn.ModuleList([
                nn.Linear(13, 128),
                nn.Linear(128, 128),
                nn.Linear(128, 128),
                nn.Linear(128, 128),
                nn.Linear(128, 128),
                nn.Linear(128, 128),
            ]),
            'output_layer': nn.Linear(128, 56)
        })
        self.fc1 = nn.Linear(56, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        tree_features = x
        for layer in self.tree_layer['hidden_layers']:
            tree_features = F.relu(layer(tree_features))
        tree_output = self.tree_layer['output_layer'](tree_features)
        fc_features = F.relu(self.fc1(tree_output))
        fc_features = F.relu(self.fc2(fc_features))
        fc_output = self.fc3(fc_features)
        return fc_output

class SaintModel(nn.Module):
    def __init__(self):
        super(SaintModel, self).__init__()
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class TabTransformerModel(nn.Module):
    def __init__(self):
        super(TabTransformerModel, self).__init__()
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class TabNetModel(nn.Module):
    def __init__(self):
        super(TabNetModel, self).__init__()
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ---------------- Load models safely ----------------
available_models = {
    'xgb': True,
    'rf': True,
    'lgbm': True,
    'catboost': True,
    'node': False,
    'saint': False,
    'tabtransformer': False,
    'tabnet': False
}

try:
    node_model = NodeModel()
    node_model.load_state_dict(torch.load('/Users/neerajprao/Downloads/ML-vs-DL-main/front_end/models/node_model.pt'))
    node_model.eval()
    available_models['node'] = True
    print("NODE model loaded successfully")
except Exception as e:
    print(f"Error loading NODE model: {e}")
    node_model = None

try:
    saint_model = SaintModel()
    saint_model.load_state_dict(torch.load('/Users/neerajprao/Downloads/ML-vs-DL-main/front_end/models/saint_model.pt'), strict=False)
    saint_model.eval()
    available_models['saint'] = True
    print("SAINT model loaded successfully")
except Exception as e:
    print(f"Error loading SAINT model: {e}")
    saint_model = None

try:
    tabtransformer_model = TabTransformerModel()
    tabtransformer_model.load_state_dict(torch.load('/Users/neerajprao/Downloads/ML-vs-DL-main/front_end/models/tabtransformer_model.pt'), strict=False)
    tabtransformer_model.eval()
    available_models['tabtransformer'] = True
    print("TabTransformer model loaded successfully")
except Exception as e:
    print(f"Error loading TabTransformer model: {e}")
    tabtransformer_model = None

try:
    tabnet_model = TabNetModel()
    tabnet_model.load_state_dict(torch.load('/Users/neerajprao/Downloads/ML-vs-DL-main/front_end/models/tabnet_model.pt'), strict=False)
    tabnet_model.eval()
    available_models['tabnet'] = True
    print("TabNet model loaded successfully")
except Exception as e:
    print(f"Error loading TabNet model: {e}")
    tabnet_model = None

# ---------------- Input & Prediction ----------------
feature_names = [
    'CreditScore', 'Geography_France', 'Geography_Germany', 'Geography_Spain',
    'Gender_Female', 'Gender_Male', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
]

def get_user_input():
    print("\nPlease enter the following features:")
    values = []
    values.append(float(input("Credit Score (e.g., 650): ")))

    geo = input("Geography (France, Germany, Spain): ").strip().capitalize()
    values += [1 if geo == "France" else 0,
               1 if geo == "Germany" else 0,
               1 if geo == "Spain" else 0]

    gender = input("Gender (Male or Female): ").strip().capitalize()
    values += [1 if gender == "Female" else 0,
               1 if gender == "Male" else 0]

    values.append(float(input("Age (e.g., 35): ")))
    values.append(int(input("Tenure (e.g., 5): ")))
    values.append(float(input("Balance (e.g., 75000.0): ")))
    values.append(int(input("Number of Products (1-4): ")))
    values.append(int(input("Has Credit Card (1 = Yes, 0 = No): ")))
    values.append(int(input("Is Active Member (1 = Yes, 0 = No): ")))
    values.append(float(input("Estimated Salary (e.g., 65000.0): ")))

    return np.array(values).reshape(1, -1)


def predict_lgbm(model, data):
    prob = model.predict(data)[0]
    return 1 if prob > 0.5 else 0

def predict_dl_models(user_tensor):
    if len(user_tensor.shape) == 2 and user_tensor.shape[0] == 1:
        tensor_for_models = user_tensor.squeeze(0)
    else:
        tensor_for_models = user_tensor
        
    print("\n--- Deep Learning Model Predictions ---")

    if available_models['node']:
        try:
            with torch.no_grad():
                node_output = node_model(tensor_for_models)
                probabilities = F.softmax(node_output, dim=0)
                node_pred = torch.argmax(probabilities).item()
            print("NODE Model:", node_pred)
        except Exception as e:
            print(f"Error in NODE prediction: {e}")
    else:
        print("NODE Model: Not available")

    if available_models['saint']:
        try:
            with torch.no_grad():
                saint_output = saint_model(tensor_for_models)
            saint_pred = 1 if torch.sigmoid(saint_output).item() > 0.5 else 0
            print("SAINT Model:", saint_pred)
        except Exception as e:
            print(f"Error in SAINT prediction: {e}")
    else:
        print("SAINT Model: Not available")

    if available_models['tabtransformer']:
        try:
            with torch.no_grad():
                tabtransformer_output = tabtransformer_model(tensor_for_models)
            tabtransformer_pred = 1 if torch.sigmoid(tabtransformer_output).item() > 0.5 else 0
            print("TabTransformer Model:", tabtransformer_pred)
        except Exception as e:
            print(f"Error in TabTransformer prediction: {e}")
    else:
        print("TabTransformer Model: Not available")

    if available_models['tabnet']:
        try:
            with torch.no_grad():
                tabnet_output = tabnet_model(tensor_for_models)
            tabnet_pred = 1 if torch.sigmoid(tabnet_output).item() > 0.5 else 0
            print("TabNet Model:", tabnet_pred)
        except Exception as e:
            print(f"Error in TabNet prediction: {e}")
    else:
        print("TabNet Model: Not available")

def predict_all_models(user_input):
    user_tensor = torch.tensor(user_input, dtype=torch.float32)

    print("\n--- Machine Learning Model Predictions ---")
    try:
        print("XGBoost:", xgb_model.predict(user_input)[0])
        print("Random Forest:", rf_model.predict(user_input)[0])
        print("LightGBM:", predict_lgbm(lgbm_model, user_input))
        print("CatBoost:", catboost_model.predict(user_input)[0])
    except Exception as e:
        print(f"Error in ML model prediction: {e}")

    predict_dl_models(user_tensor)

def show_example_values():
    examples = {
        'CreditScore': '650 (range: 300-850)',
        'Geography': 'France, Germany, or Spain',
        'Gender': 'Female or Male',
        'Age': '35 (range: 18-100)',
        'Tenure': '5 (range: 0-10 years)',
        'Balance': '75000.00 (account balance)',
        'NumOfProducts': '2 (range: 1-4 products)',
        'HasCrCard': '1 (Yes) or 0 (No)',
        'IsActiveMember': '1 (Yes) or 0 (No)',
        'EstimatedSalary': '65000.00 (annual salary)'
    }
    print("\nExample values for reference:")
    for feature, example in examples.items():
        print(f"{feature}: {example}")
    print()

def main():
    print("Bank Customer Churn Prediction")
    print("=============================")
    print("This program predicts whether a bank customer is likely to leave the bank.")
    print("0 = Customer stays, 1 = Customer churns (leaves)")

    while True:
        show_example_values()
        user_input = get_user_input()
        predict_all_models(user_input)
        cont = input("\nDo you want to make another prediction? (y/n): ").strip().lower()
        if cont != 'y':
            print("Thank you for using the Bank Customer Churn Prediction tool.")
            break

if __name__ == '__main__':
    main()
