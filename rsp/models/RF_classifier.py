import optuna
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from joblib import Parallel, delayed, cpu_count
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score  # Changed from accuracy_score
from sklearn.ensemble import RandomForestClassifier
import os
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.simplefilter("ignore", UserWarning)  # Suppress all user warnings globally

# Set random seed for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)

seed = 42
set_seeds(seed)

# Get optimal number of cores for M2 Pro Max
NUM_CORES = cpu_count()

def process_fold_rf_classifier(train_idx, val_idx, X, y, param):
    """
    Train and validate a Random Forest Classifier on a given fold.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    X_fold_train, X_fold_val = X[train_idx], X[val_idx]
    y_fold_train, y_fold_val = y[train_idx], y[val_idx]

    model = RandomForestClassifier(**param, random_state=42, class_weight={0: 2.0, 1: 1.0}, n_jobs=-1)
    model.fit(X_fold_train, y_fold_train)

    preds = model.predict(X_fold_val)
    f1 = f1_score(y_fold_val, preds)  # Changed to F1 score

    return f1, preds, val_idx  # Return F1 score instead of accuracys

def rf_objective_classifier(trial, X, y, n_splits):
    """
    Define the objective function for Optuna hyperparameter optimization of Random Forest Classifier.
    """
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),  # Number of trees
        'max_depth': trial.suggest_int('max_depth', 3, 10),  # Depth of trees
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),  # Min samples to split
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),  # Min samples per leaf
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),  # Feature selection
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])  # Use bootstrap sampling
    }

    # Use StratifiedKFold for classification
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Process folds in parallel using joblib
    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(process_fold_rf_classifier)(train_idx, val_idx, X, y, param) 
        for train_idx, val_idx in skf.split(X, y)
    )

    # Extract F1 scores per fold and predictions
    cv_f1_scores, predictions_per_fold, indices_per_fold = zip(*results)

    # Reconstruct CV predictions in the correct order
    all_predictions = np.zeros_like(y, dtype=int)
    for preds, val_idx in zip(predictions_per_fold, indices_per_fold):
        all_predictions[val_idx] = preds  

    # Compute statistics
    mean_f1 = np.mean(cv_f1_scores)
    std_f1 = np.std(cv_f1_scores)

    # Store additional metadata in Optuna trial
    trial.set_user_attr("cv_f1_per_fold", cv_f1_scores)
    trial.set_user_attr("std_f1", std_f1)
    trial.set_user_attr("cv_predictions", all_predictions)  

    return mean_f1  # Maximize F1 score

# Example usage:
# study = optuna.create_study(direction="maximize")
# study.optimize(lambda trial: rf_objective_classifier(trial, X, y, n_splits=5), n_trials=50, n_jobs=-1)