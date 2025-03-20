#code\modeling\6_individual_model_training.py

import sys
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import DATA_PATHS, MODEL_PATHS
from model_utils import create_pipeline

def train_individual_models():
    datasets = ['crosscheck', 'studentlife']

    for dataset in datasets:
        print(f"🚀 Training model for dataset: {dataset}")

        try:
            df = pd.read_csv(DATA_PATHS[f'{dataset}_processed'])
        except FileNotFoundError:
            print(f"❌ Error: Processed dataset not found for {dataset}. Skipping...")
            continue

        drop_cols = ['study_id', 'date', 'eureka_id']
        drop_cols = [col for col in drop_cols if col in df.columns]
        X = df.drop(columns=drop_cols + ['ema_score'])
        y = df['ema_score']

        if len(np.unique(y)) < 2:
            print(f"❌ Error: Not enough class variety in {dataset}. Skipping...")
            continue

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
        except ValueError:
            print(f"⚠️ Warning: Stratification failed for {dataset}. Disabling stratification.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=None
            )

        threshold = np.median(y_train)
        y_train = np.where(y_train > threshold, 1, 0)
        y_test = np.where(y_test > threshold, 1, 0)

        ros = RandomOverSampler(random_state=42)
        X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

        pipeline = create_pipeline(
            RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        )

        pipeline.fit(X_train_resampled, y_train_resampled)
        print(f"✅ Model trained for {dataset}!")

        MODEL_PATHS[dataset].mkdir(parents=True, exist_ok=True)
        model_path = MODEL_PATHS[dataset] / 'best_model.pkl'
        joblib.dump(pipeline, model_path)
        print(f"💾 Model saved at: {model_path}")

if __name__ == '__main__':
    train_individual_models()