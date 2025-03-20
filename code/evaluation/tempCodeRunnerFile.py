# code/evaluation/8_performance_evaluation.py
import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
from config import DATA_PATHS, MODEL_PATHS

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def evaluate_models():
    datasets = ['crosscheck', 'studentlife']

    for dataset in datasets:
        print(f"📂 Processing dataset: {dataset}")

        # Load dataset
        data_path = DATA_PATHS.get(f'{dataset}_processed')
        if not os.path.exists(data_path):
            print(f"❌ Dataset not found: {data_path}")
            continue

        df = pd.read_csv(data_path)

        # Ensure dataset has required columns
        if 'study_id' not in df.columns or 'ema_score' not in df.columns:
            print(f"❌ Missing required columns in {dataset} dataset.")
            continue

        # Load model
        model_path = MODEL_PATHS[dataset] / 'best_model.pkl'
        if not os.path.exists(model_path):
            print(f"❌ Model not found for {dataset}.")
            continue

        model = joblib.load(model_path)
        print(f"✅ Loaded model from: {model_path}")

        X = df.drop(columns=['study_id', 'ema_score'], errors='ignore')
        y = df['ema_score']
        groups = df['study_id']

        logo = LeaveOneGroupOut()
        unique_subjects = df['study_id'].unique()

        if len(unique_subjects) < 2:
            print(f"❌ Not enough subjects for LOSO cross-validation in {dataset}.")
            continue

        print(f"📊 Found {len(unique_subjects)} unique subjects for LOSO.")

        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            print(f"📈 Accuracy: {acc:.4f}")

if __name__ == '__main__':
    evaluate_models()
