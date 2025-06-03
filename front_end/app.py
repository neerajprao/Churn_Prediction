import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from xgboost import XGBClassifier
from lightgbm import Booster
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

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
        return self.fc3(fc_features)

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

# ---------------- Load Models ----------------
def load_models():
    global xgb_model, rf_model, lgbm_model, catboost_model
    global node_model, saint_model, tabtransformer_model, tabnet_model
    global available_models

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

    # Load XGBoost model
    try:
        xgb_model = XGBClassifier()
        xgb_model.load_model('models/xgboost_model.json')
    except Exception as e:
        print(f"Error loading XGBoost model: {e}")
        available_models['xgb'] = False

    # Load Random Forest model
    try:
        rf_model = joblib.load('models/random_forest.pkl')
    except Exception as e:
        print(f"Error loading Random Forest model: {e}")
        available_models['rf'] = False

    # Load LightGBM model
    try:
        lgbm_model = Booster(model_file='models/lightgbm_model.txt')
    except Exception as e:
        print(f"Error loading LightGBM model: {e}")
        available_models['lgbm'] = False

    # Load CatBoost model
    try:
        catboost_model = CatBoostClassifier()
        catboost_model.load_model('models/catboost_model.cbm')
    except Exception as e:
        print(f"Error loading CatBoost model: {e}")
        available_models['catboost'] = False

    # Load NODE model
    try:
        node_model = NodeModel()
        node_model.load_state_dict(torch.load('models/node_model.pt'))
        node_model.eval()
        available_models['node'] = True
        print("NODE model loaded successfully")
    except Exception as e:
        print(f"Error loading NODE model: {e}")
        node_model = None

    # Load SAINT model
    try:
        saint_model = SaintModel()
        saint_model.load_state_dict(torch.load('models/saint_model.pt'), strict=False)
        saint_model.eval()
        available_models['saint'] = True
        print("SAINT model loaded successfully")
    except Exception as e:
        print(f"Error loading SAINT model: {e}")
        saint_model = None

    # Load TabTransformer model
    try:
        tabtransformer_model = TabTransformerModel()
        tabtransformer_model.load_state_dict(torch.load('models/tabtransformer_model.pt'), strict=False)
        tabtransformer_model.eval()
        available_models['tabtransformer'] = True
        print("TabTransformer model loaded successfully")
    except Exception as e:
        print(f"Error loading TabTransformer model: {e}")
        tabtransformer_model = None

    # Load TabNet model
    try:
        # First try loading using TabNetModel (our custom implementation)
        try:
            tabnet_model = TabNetModel()
            tabnet_model.load_state_dict(torch.load('models/tabnet_model.pt'), strict=False)
            tabnet_model.eval()
            available_models['tabnet'] = True
            print("TabNet model loaded successfully (custom implementation)")
        except Exception as e:
            print(f"Error loading custom TabNet model: {e}")
            # Try loading using TabNetClassifier
            tabnet_model = TabNetClassifier()
            # Load the TabNet model from the .pt file
            saved_model = torch.load('models/tabnet_model.pt')
            # If the saved model is a state dict
            if isinstance(saved_model, dict):
                tabnet_model.load_state_dict(saved_model)
            # If the saved model is the entire model
            else:
                tabnet_model = saved_model
            tabnet_model.eval()  # Set to evaluation mode
            available_models['tabnet'] = True
            print("TabNet model loaded successfully (TabNetClassifier)")
    except Exception as e:
        print(f"Error loading all versions of TabNet model: {e}")
        available_models['tabnet'] = False
        tabnet_model = None

    return available_models

# Function to make predictions with LightGBM model
def predict_lgbm(model, data):
    prob = model.predict(data)[0]
    return 1 if prob > 0.5 else 0

# Function to prepare input data
def prepare_input_data(form_data):
    credit_score = float(form_data.get('creditScore'))
    age = float(form_data.get('age'))
    tenure = float(form_data.get('tenure'))
    balance = float(form_data.get('balance'))
    num_products = float(form_data.get('numProducts'))
    has_card = float(form_data.get('hasCard'))
    is_active = float(form_data.get('isActive'))
    salary = float(form_data.get('salary'))
    geography = form_data.get('geography')
    gender = form_data.get('gender')

    geography_france = 1 if geography == 'France' else 0
    geography_germany = 1 if geography == 'Germany' else 0
    geography_spain = 1 if geography == 'Spain' else 0
    gender_female = 1 if gender == 'Female' else 0
    gender_male = 1 if gender == 'Male' else 0

    input_data = np.array([
        credit_score, geography_france, geography_germany, geography_spain,
        gender_female, gender_male, age, tenure, balance, num_products,
        has_card, is_active, salary
    ]).reshape(1, -1)

    return input_data

# ---------------- Flask Routes ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models')
def models():
    return render_template('models.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = prepare_input_data(data)
        input_tensor = torch.tensor(input_data, dtype=torch.float32)

        predictions = {}

        # Make predictions with available ML models
        if available_models['xgb']:
            predictions['xgb'] = int(xgb_model.predict(input_data)[0])
        else:
            predictions['xgb'] = "Not available"

        if available_models['rf']:
            predictions['rf'] = int(rf_model.predict(input_data)[0])
        else:
            predictions['rf'] = "Not available"

        if available_models['lgbm']:
            predictions['lgbm'] = predict_lgbm(lgbm_model, input_data)
        else:
            predictions['lgbm'] = "Not available"

        if available_models['catboost']:
            predictions['catboost'] = int(catboost_model.predict(input_data)[0])
        else:
            predictions['catboost'] = "Not available"

        # Make predictions with available DL models
        if available_models['node']:
            with torch.no_grad():
                node_output = node_model(input_tensor)
                probabilities = F.softmax(node_output, dim=1)
                predictions['node'] = int(torch.argmax(probabilities).item())
        else:
            predictions['node'] = "Not available"

        if available_models['saint']:
            with torch.no_grad():
                saint_output = saint_model(input_tensor)
                predictions['saint'] = 1 if torch.sigmoid(saint_output).item() > 0.5 else 0
        else:
            predictions['saint'] = "Not available"

        if available_models['tabtransformer']:
            with torch.no_grad():
                tabtransformer_output = tabtransformer_model(input_tensor)
                predictions['tabtransformer'] = 1 if torch.sigmoid(tabtransformer_output).item() > 0.5 else 0
        else:
            predictions['tabtransformer'] = "Not available"

        if available_models['tabnet']:
            try:
                # Check if it's our custom TabNetModel implementation
                if isinstance(tabnet_model, TabNetModel):
                    with torch.no_grad():
                        tabnet_output = tabnet_model(input_tensor)
                        predictions['tabnet'] = 1 if torch.sigmoid(tabnet_output).item() > 0.5 else 0
                # Otherwise assume it's TabNetClassifier implementation
                else:
                    if hasattr(tabnet_model, 'predict'):
                        tab_output = tabnet_model.predict(input_data)
                        predictions['tabnet'] = int(tab_output[0] > 0.5)
                    else:
                        with torch.no_grad():
                            tab_output = tabnet_model(input_tensor)
                            if isinstance(tab_output, tuple):  # Some TabNet implementations return multiple values
                                tab_output = tab_output[0]
                            predictions['tabnet'] = int(torch.sigmoid(tab_output).item() > 0.5)
            except Exception as e:
                print(f"Error making prediction with TabNet: {e}")
                predictions['tabnet'] = "Error in prediction"
        else:
            predictions['tabnet'] = "Not available"

        return jsonify({'success': True, 'predictions': predictions})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    load_models()
    app.run(debug=True)