# Churn Predictor

## Overview

Churn Predictor is a powerful web-based application that predicts customer churn using a combination of traditional machine learning and advanced deep learning models. It is designed to help businesses retain customers by identifying individuals at high risk of leaving based on their profile and behavioral attributes.

## Features

- Predicts customer churn using:
  - Classical Machine Learning: Random Forest, XGBoost, LightGBM, CatBoost
  - Deep Learning: SAINT, NODE, TabNet, TabTransformer
- Model comparison with accuracy metrics
- Interactive UI for inputting customer details
- Clean API and modular Flask backend
- Built using Python, HTML, CSS, JavaScript, Flask

## Project Structure

```
churn-predictor/
│
├── dataset/
│   └── predict-bank-churn.csv   # Dataset for training and evaluation
│
├── front_end/                   # Flask application
│   ├── app.py                   # Main Flask application file
│   ├── main.py                  # Entry point to run the Flask app
│   ├── models/                  # Pre-trained ML/DL models (.pt, .pkl)
│   ├── static/                  # CSS/JS files and images
│   │   ├── script.js
│   │   ├── style.css
│   │   └── images/
│   │       └── IMG.png
│   └── templates/               # HTML templates
│       ├── index.html
│       └── models.html
│
├── results/
│   └── BankChurn_Eval_Results.xlsx # Evaluation results of the models
│
├── notebooks/                   # EDA, preprocessing, training notebooks (Optional - Add if you have them)
│
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Input Features

The models use the following customer features:

- CreditScore
- Geography: France, Germany, Spain (One-hot encoded)
- Gender: Male, Female (One-hot encoded)
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard (0 or 1)
- IsActiveMember (0 or 1)
- EstimatedSalary

## Models Used

### Classical ML

- Random Forest
- XGBoost
- LightGBM
- CatBoost

### Deep Learning

- SAINT (Self-Attention & Intersample Attention)
- NODE (Neural Oblivious Decision Ensembles)
- TabNet
- TabTransformer

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/churn-predictor.git
cd churn-predictor
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Flask app:

```bash
python front_end/main.py
```

5. Open your browser and go to [http://localhost:5000](http://localhost:5000)

## Sample Output

- Predicted churn probability (0 to 1)
- Final classification: Will Churn or Will Not Churn
- Option to view model-specific predictions

## Future Enhancements

- Add Explainability with SHAP or LIME
- Deploy on Streamlit Cloud or Hugging Face Spaces
- Integrate real-time data ingestion from a CRM or database

## Author

**Neeraj P Rao**  
Third-Year CSE Student at RV University  
Machine Learning, Quantum AI, and Full-Stack Enthusiast

## License

This project is licensed under the MIT License.

## Sources

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [PyTorch Tabular Models](https://pytorch-tabular.readthedocs.io/)
