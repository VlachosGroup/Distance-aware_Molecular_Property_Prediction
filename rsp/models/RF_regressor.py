import optuna
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from joblib import Parallel, delayed, cpu_count
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import os
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.simplefilter("ignore", UserWarning)  # Suppress all user warnings globally


def set_seeds(seed=42):
    np.random.seed(seed)

seed = 42
set_seeds(seed)

# Get optimal number of cores for M2 Pro Max
NUM_CORES = cpu_count()

def process_fold_rf(train_idx, val_idx, X, y, param):
    """
    Train and validate a Random Forest model on a given fold.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    X_fold_train, X_fold_val = X[train_idx], X[val_idx]
    y_fold_train, y_fold_val = y[train_idx], y[val_idx]

    model = RandomForestRegressor(**param, random_state=42, n_jobs=-1)
    model.fit(X_fold_train, y_fold_train)

    preds = model.predict(X_fold_val)
    rmse = np.sqrt(mean_squared_error(y_fold_val, preds))  # Return RMSE for this fold

    return rmse, preds, val_idx  # Return RMSE, predictions, and indices

def rf_objective(trial, X, y, n_splits):
    """
    Define the objective function for Optuna hyperparameter optimization of Random Forest.
    """
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),  # Number of trees
        'max_depth': trial.suggest_int('max_depth', 3, 10),  # Depth of trees
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),  # Min samples to split
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),  # Min samples per leaf
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),  # Feature selection
        'bootstrap': False  
    }

    # Initialize K-Fold Cross-Validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Process folds in parallel using joblib
    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(process_fold_rf)(train_idx, val_idx, X, y, param) 
        for train_idx, val_idx in kf.split(X)
    )

 # Extract RMSE per fold and predictions
    cv_rmse_scores, predictions_per_fold, indices_per_fold = zip(*results)

    # Reconstruct CV predictions in the correct order
    all_predictions = np.zeros_like(y, dtype=float)
    for preds, val_idx in zip(predictions_per_fold, indices_per_fold):
        all_predictions[val_idx] = preds  

    # Compute statistics
    mean_rmse = np.mean(cv_rmse_scores)
    std_rmse = np.std(cv_rmse_scores)

    # Store additional metadata in Optuna trial
    trial.set_user_attr("cv_rmse_per_fold", cv_rmse_scores)
    trial.set_user_attr("std_rmse", std_rmse)
    trial.set_user_attr("cv_predictions", all_predictions)  

    return mean_rmse  # Minimize this