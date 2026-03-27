# src/predict_pipeline.py

import pandas as pd
import joblib
import os

# Load pipeline
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(BASE_DIR,"models")
os.makedirs(model_dir,exist_ok=True)
model = joblib.load(model_dir)
model_path = os.path.join(model_dir,"churn_pipeline.pkl")
joblib.dump(best_model,model_path)

def predict(input_dict):
    df = pd.DataFrame([input_dict])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return prediction, probability


# Test
if __name__ == "__main__":

    sample_input = {
        'tenure': 12,
        'MonthlyCharges': 2000,
        'TotalCharges': 5000,
        'Contract': 'Month-to-month',
        'InternetService': 'Fiber optic',
        'PaymentMethod': 'Electronic check'
    }

    pred, prob = predict(sample_input)

    print(f"Prediction: {pred}")
    print(f"Probability: {prob:.2f}")