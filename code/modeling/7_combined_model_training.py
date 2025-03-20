# code\modeling\7_combined_model_training.py


import os
import sys
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

def train_combined_model():
    print("🚀 Training Combined Model...")

    data_path = DATA_PATHS.get('combined')

    if not data_path or not os.path.exists(data_path):
        print(f"❌ Dataset not found: {data_path}")
        return

    df = pd.read_csv(data_path)

    if 'ema_score' not in df.columns:
        print("❌ Target column 'ema_score' not found in dataset. Check preprocessing.")
        return

    print("Class Distribution Before Handling:")
    print(df['ema_score'].value_counts())

    value_counts = df['ema_score'].value_counts()
    single_sample_classes = value_counts[value_counts == 1].index

    if len(single_sample_classes) > 0:
        print(f"⚠️ Removing rows with single samples of classes: {single_sample_classes.tolist()}")
        df = df[~df['ema_score'].isin(single_sample_classes)]
        print("Class Distribution After Handling:")
        print(df['ema_score'].value_counts())

    X = df.drop(columns=['study_id', 'date', 'ema_score'], errors='ignore')
    y = df['ema_score']

    if y.nunique() < 2:
        print("❌ Not enough unique values in target variable for stratification.")
        return

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    except ValueError as e:
        print(f"❌ Train-test split failed: {e}")
        return

    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    pipeline = create_pipeline(RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    pipeline.fit(X_train_resampled, y_train_resampled)

    print("✅ Model trained successfully!")

    model_dir = MODEL_PATHS.get('combined')
    if not model_dir:
        print("❌ Error: MODEL_PATHS['combined'] is not set. Check config.py")
        return

    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'combined_model.pkl')
    joblib.dump(pipeline, model_path)

    print(f"💾 Model saved at: {model_path}")

if __name__ == '__main__':
    train_combined_model()