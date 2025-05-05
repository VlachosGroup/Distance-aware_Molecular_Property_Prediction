import optuna
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import multiprocessing as mp
from functools import partial
from joblib import Parallel, delayed, cpu_count
import os
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import pickle
import warnings

warnings.simplefilter("ignore", UserWarning)  # Suppress all user warnings globally


def set_seeds(seed=42):
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
set_seeds(seed)


# Get optimal number of cores for M2 Pro Max
NUM_CORES = cpu_count()

# Device configuration
device = torch.device("cpu")
print(f"Using device: {device}")

def process_fold_xgb(train_idx, val_idx, X, y, param):
    if isinstance(X, pd.DataFrame):
        X = X.values
    X_fold_train, X_fold_val = X[train_idx], X[val_idx]
    y_fold_train, y_fold_val = y[train_idx], y[val_idx]

    model = xgb.XGBRegressor(**param)
    model.fit(
        X_fold_train, 
        y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    preds = model.predict(X_fold_val)
    rmse = np.sqrt(mean_squared_error(y_fold_val, preds))

    return rmse, preds, val_idx  # Return RMSE, predictions, and indices

def xgboost_objective(trial, X, y, n_splits):
    param = {
        'tree_method': 'hist',
        'nthread': NUM_CORES // n_splits,  # Divide cores among folds
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.5, 0.7, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.5, 0.7, 1.0]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 1, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 10.0),  # New!
        'max_leaves': trial.suggest_int('max_leaves', 10, 100),  # New!
        'grow_policy': 'lossguide'
    }

    # Initialize K-Fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Process folds in parallel using joblib
    results = Parallel(n_jobs=NUM_CORES, prefer="processes")(
        delayed(process_fold_xgb)(train_idx, val_idx, X, y, param) 
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
    trial.set_user_attr("best_fold_rmse", np.min(cv_rmse_scores))
    trial.set_user_attr("worst_fold_rmse", np.max(cv_rmse_scores))
    trial.set_user_attr("cv_predictions", all_predictions)  

    return mean_rmse  # Minimize this
