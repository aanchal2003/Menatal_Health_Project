# code/evaluation/8_performance_evaluation.py

import sys
import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_PATHS, MODEL_PATHS, BASE_DIR

def loso_evaluation():
    results = []
    print("🚀 Starting LOSO evaluation...")

    for dataset in ['crosscheck', 'studentlife', 'combined']:
        try:
            print(f"\n📂 Processing dataset: {dataset}")
            df = pd.read_csv(DATA_PATHS[f'{dataset}_processed'])
            model_path = MODEL_PATHS[dataset] / ('combined_model.pkl' if dataset == 'combined' else 'best_model.pkl')

            if not model_path.exists():
                print(f"❌ Model not found: {model_path}")
                continue

            print(f"✅ Loaded model from: {model_path}")
            model = joblib.load(model_path)
            subjects = df['study_id'].unique()

            accuracies = []
            f1_scores = []

            for subject in subjects:
                train = df[df['study_id'] != subject]
                test = df[df['study_id'] == subject]

                # Relaxed constraints: Reduced minimum samples to 1
                if len(test) < 1 or len(train) < 1:
                    continue

                X_train = train.drop(columns=['ema_score', 'study_id'], errors='ignore')
                y_train = train['ema_score']
                X_test = test.drop(columns=['ema_score', 'study_id'], errors='ignore')
                y_test = test['ema_score']

                if len(y_train.unique()) < 2:
                    continue

                # Relaxed constraints: Reduced minimum minority class size to 1
                y_counts = y_train.value_counts()
                if y_counts.min() < 1:
                    continue

                try:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    acc = accuracy_score(y_test, preds)
                    f1 = f1_score(y_test, preds, average='weighted')

                    accuracies.append(acc)
                    f1_scores.append(f1)

                except Exception as e:
                    print(f"⚠️ Error with subject {subject}: {e}")
                    continue

            if accuracies:
                avg_accuracy = sum(accuracies) / len(accuracies)
                avg_f1 = sum(f1_scores) / len(f1_scores)
                print(f"📊 Average Accuracy for {dataset}: {avg_accuracy:.4f}")
                print(f"📊 Average F1-score for {dataset}: {avg_f1:.4f}")
                results.append({
                    'dataset': dataset,
                    'accuracy': avg_accuracy,
                    'f1': avg_f1
                })
            else:
                print(f"⚠️ No valid evaluations for {dataset}.")

        except Exception as e:
            print(f"❌ Dataset {dataset} error: {e}")
            continue

    results_dir = BASE_DIR / 'results/metrics'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'loso_average_results.csv'

    pd.DataFrame(results).to_csv(results_path, index=False)
    print("\n📊 Evaluation completed! Results saved to:", results_path)

if __name__ == '__main__':
    loso_evaluation()