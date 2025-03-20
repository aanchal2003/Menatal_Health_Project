# Menatal_Health_Project

Project Title: Mental Health Prediction from Sensor and EMA Data

Description:

This project aims to predict mental health scores (derived from EMA stress and mood levels) using data collected from smartphone sensors and Ecological Momentary Assessments (EMA). It encompasses data preprocessing, feature analysis, model training, and evaluation, utilizing datasets from the CrossCheck and StudentLife studies.

Key Features:

Data Preprocessing: Cleans and prepares raw sensor and EMA data for analysis, handling missing values and inconsistencies.
Feature Analysis: Performs feature comparison and selection to identify relevant predictors.
Model Training: Trains machine learning models (Random Forest) on individual and combined datasets.
Model Evaluation: Evaluates model performance using Leave-One-Group-Out (LOGO) cross-validation, providing accuracy and F1-score metrics.
Combined Dataset Analysis: Combines datasets to potentially enhance model performance.

Project Structure:

code/: Contains Python scripts for:
preprocessing/: Data cleaning and preparation.
feature_analysis/: Feature comparison and selection.
modeling/: Model training.
evaluation/: Model performance evaluation.
config.py: Stores data and model paths, and feature settings.
models/: Stores trained model files.
data/: Stores raw and processed datasets.
results/: Stores evaluation metrics and feature selection results.

Dependencies:

pandas
scikit-learn
joblib
imblearn
seaborn
scipy
pathlib
Usage:


Install dependencies: pip install -r requirements.txt (create a requirements.txt file)
Ensure data is placed in the data/ directory.
Run the scripts in the code/ directory in sequential order.

Key Files and Their Function:
1_preprocess_crosscheck.py: Preprocesses the CrossCheck dataset.
2_preprocess_studentlife.py: Preprocesses the StudentLife dataset.
3_combine_datasets.py: Combines the preprocessed datasets.
4_feature_comparison.py: Compares features between datasets.
5_feature_selection.py: Selects relevant features using RFECV.
6_individual_model_training.py: Trains models on individual datasets.
7_combined_model_training.py: Trains a model on the combined dataset.
8_performance_evaluation.py: Evaluates model performance.
config.py: Holds directory paths and feature settings.

Evaluation Metrics:
Accuracy
F1-score (weighted)

Datasets are taken online.
