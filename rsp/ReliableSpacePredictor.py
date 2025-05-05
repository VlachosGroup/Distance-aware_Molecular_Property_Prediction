import numpy as np
import pandas as pd
from rdkit import Chem
import warnings
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_score,  cross_val_predict, KFold, train_test_split
from sklearn.svm import OneClassSVM
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, UnivariateSpline
from joblib import Parallel, delayed
from datetime import datetime
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
import optuna
from optuna.trial import Trial
from optuna.exceptions import TrialPruned
import multiprocessing as mp
from functools import partial
import os
from concurrent.futures import ProcessPoolExecutor
import pickle
from .Conformal_Predictors import *
from .models.NN_model import Net,train_final_nn,process_fold_nn, nn_objective, predict_with_model
from .models.XGB_model import xgboost_objective
from .models.RF_regressor import rf_objective
from .models.RF_classifier import rf_objective_classifier
# from xgboost import XGBRegressor
import matplotlib.pyplot as plt #####
from .utils import *

def print_progress(study, trial, n_trials):
    """Print progress of optimization"""
    try:
        print(f"Trial {trial.number + 1}/{n_trials} completed | Best score so far: {study.best_value:.4f}")
    except:
        print(f"Trial {trial.number + 1}/{n_trials} completed")

class ReliableSpacePredictor:
    def __init__(self, radius=2, n_bits=2048, cv_folds=10, n_trials_distance=30, n_trials_model_optimization=100, threshold_range = None, model_type='rf', model_params=None):
        self.radius = radius
        self.n_bits = n_bits
        self.cv_folds = cv_folds
        self.pca = None
        self.X_3d_base = None
        self.optimal_weights = None
        self.optimal_threshold = None
        self.optimal_lambda = None
        self.centroid = None
        self.distances = None
        self.best_model_study = None
        self.best_feature_mask = None
        self.best_score = np.inf
        self.best_rmse_per_fold = None
        self.best_mean_rmse = None
        self.best_rmse_std = None
        self.best_loss = None
        self.best_cv_predictions = None
        self.X_RSP_optimized = None
        self.y_RSP_optimized = None
        self.model_type = model_type
        self.model_params = model_params or {}
        self.best_model = None
        self.best_dcp = None
        self.selected_indices_base = None
        self.selected_indices_new = None
        self.n_trials_distance = n_trials_distance
        self.n_trials_model_optimization = n_trials_model_optimization
        self.threshold_range = threshold_range
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_fingerprints(self, smiles_list):
        """Generate Morgan fingerprints for visualization"""
        print("Generating fingerprints...")
        fps = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, self.n_bits)
                fps.append(list(fp.ToBitString()))
        return np.array(fps, dtype=int)

    def define_space(self, smiles_list, descriptors, y):
        # 1) Store descriptors if needed
        if isinstance(descriptors, pd.DataFrame):
            self.descriptors = descriptors.copy()
        else:
            self.descriptors = pd.DataFrame(descriptors)
        
        print("Fitting base dataset...")
        fps = self.generate_fingerprints(smiles_list)
        
        # 2) Kernel PCA (no shift step)
        from sklearn.decomposition import KernelPCA
        self.pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1, random_state=42)
        X_2d = self.pca.fit_transform(fps)  # shape: (n_samples,2)
        
        # 3) Scale y using y's own std
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        y_scaled = (y - self.y_mean) / self.y_std

        # 4) Combine to 3D space
        X_3d = np.column_stack([X_2d, y_scaled])
        self.X_3d_base = X_3d

        # 5) Compute centroid (no shift needed to avoid negative coords)
        self.centroid = np.mean(X_3d, axis=0)

        return X_3d

    def scaler(self, y):
        # scale new y values the same way
        y_scaled = (y - self.y_mean) / self.y_std
        return y_scaled

    def weighted_distance(self, point, weights):
        """Calculate weighted distance between point and centroid"""
        return np.sqrt(sum(w * (p - c)**2 for w, p, c in zip(weights, point, self.centroid)))

    def get_distances(self, X_new, weights):
        """Get distances to centroid"""
        distances = np.array([
            self.weighted_distance(point, weights)
            for point in X_new
        ])
        return distances

    def remove_correlated_features(self, X, corr_cutoff=0.75):
        """Remove highly correlated features"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        cor_matrix = X.corr().abs()
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_cutoff)]
        X = X.drop(columns=to_drop, axis=1)
        
        return X

    def perform_feature_selection(self, X, y, n_top_features=30):
        """Two-step feature selection: importance ranking then RFE"""
        print("Performing feature selection...")
        if isinstance(X, pd.DataFrame):
            X_numpy = X.values
            feature_names = X.columns
        else:
            X_numpy = X
            feature_names = np.arange(X.shape[1])

        # we stick to RF for feature selection as it's a well estabilshed method (for stability)
        # Step 1: Select Top n Features based on RF importance
        rf_initial = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_initial.fit(X_numpy, y)
        importances = rf_initial.feature_importances_
        top_indices = np.argsort(importances)[::-1][:n_top_features]

        # Keep only top features
        X_reduced = X_numpy[:, top_indices]
        
        # Step 2: RFECV on reduced feature set
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        rfe = RFECV(
            estimator=rf,
            step=1,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        rfe.fit(X_reduced, y)

        # Get final mask and CV predictions
        final_mask = np.zeros(X_numpy.shape[1], dtype=bool)
        final_mask[top_indices[rfe.support_]] = True

        return final_mask
    
    def optimize_model(self, X_train, y_train):
        """Optimize model hyperparameters using Optuna."""
        if isinstance(X_train, pd.DataFrame):
            X_numpy = X_train.values
        else:
            X_numpy = X_train

        study = optuna.create_study(direction='minimize')
        
        if self.model_type == 'nn':
            X_train_torch = torch.tensor(X_train.values, dtype=torch.float32)
            y_train_torch = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            study.optimize(lambda trial: nn_objective(trial, X_train_torch, y_train_torch, self.cv_folds), n_trials=self.n_trials_model_optimization)
            best_params = study.best_trial.params
            model = train_final_nn(best_params, X_train, y_train, 'whatever', self.device, save = False)
        
        elif self.model_type == 'xgb':  
            study.optimize(lambda trial: xgboost_objective(trial, X_train, y_train, self.cv_folds), n_trials=self.n_trials_model_optimization)
            best_params = study.best_trial.params
            model = xgb.XGBRegressor(**best_params)
            model.fit(X_train, y_train)

        else:  # RF
            study.optimize(lambda trial: rf_objective(trial, X_train, y_train, self.cv_folds), n_trials=self.n_trials_model_optimization)
            best_params = study.best_trial.params
            model = RandomForestRegressor(**best_params)
            model.fit(X_train, y_train)

        return model, best_params, study
    
    def objective(self, trial, X_new, X_new_desc, y_new, X_base, X_base_desc, y_base):
        """Objective function for Optuna optimization"""
        print(f"Starting trial {trial.number}")
        w0 = trial.suggest_float('w0', 0.1, 1.0)
        w1 = trial.suggest_float('w1', 0.1, 1.0)
        w2 = trial.suggest_float('w2', 0.1, 1.0)
        weights = [w0, w1, w2]

        if self.threshold_range != None:
            threshold_lower = self.threshold_range[0]
            threshold_upper = self.threshold_range[1]
            threshold = trial.suggest_float('threshold', threshold_lower, threshold_upper)
        else:
            threshold = trial.suggest_float('threshold', 4.0, 7.0)
        lambda_ = trial.suggest_float('lambda', 0.1, 0.5)  # Confidence interval scaling
        distances_base = self.get_distances(X_base, weights)
        distances_new = self.get_distances(X_new, weights)
        selected_indices_base = [i for i, dist in enumerate(distances_base) if dist < threshold]
        selected_indices_new = [i for i, dist in enumerate(distances_new) if dist < threshold]
        distances_combined = np.concatenate([distances_base[selected_indices_base], distances_new[selected_indices_new]])
        
        if len(selected_indices_base) > 0 and len(selected_indices_new) > 0:
            X_combined = pd.concat([X_base_desc.iloc[selected_indices_base].reset_index(drop=True), X_new_desc.iloc[selected_indices_new].reset_index(drop=True)]).reset_index(drop=True)
            original_columns = X_combined.columns.tolist()
            X_combined = self.remove_correlated_features(X_combined)
            remaining_columns = X_combined.columns.tolist()
            feature_map = {new_idx: original_columns.index(col) for new_idx, col in enumerate(remaining_columns)}
            X_combined = X_combined.values
            y_combined = np.concatenate([y_base[selected_indices_base], y_new[selected_indices_new]])
            feature_mask = self.perform_feature_selection(X_combined, y_combined)
            X_combined = X_combined[:, np.where(feature_mask)[0]] # keep only the selected features
            
            full_feature_mask = np.zeros(len(original_columns), dtype=bool)
            if feature_mask is not None:
                for i, is_selected in enumerate(feature_mask):
                    if is_selected:
                        original_idx = feature_map[i]
                        full_feature_mask[original_idx] = True
            
            # Train final model & collect CV predictions
            model, best_params, study = self.optimize_model(X_combined, y_combined)
            cv_rmse_per_fold = study.best_trial.user_attrs["cv_rmse_per_fold"]
            cv_rmse_std = study.best_trial.user_attrs["std_rmse"]
            cv_predictions = study.best_trial.user_attrs["cv_predictions"]
            errors_combined = y_combined - np.array(cv_predictions[:len(selected_indices_base)].tolist() + cv_predictions[len(selected_indices_base):].tolist())
            pearson_train = pearsonr(distances_combined, errors_combined)
            pearson_abs_train = pearsonr(distances_combined, abs(errors_combined))
            mean_rmse = study.best_value

            # Fit Distance-Based Conformal Predictor
            dcp = DistanceBasedConformalPredictor(confidence=0.95)
            dcp.fit(y_combined, cv_predictions, distances_combined)

            lower, upper = dcp.predict_interval(cv_predictions, lambda_, distances_combined)
            # Compute final coverage correctly
            coverage = dcp.get_coverage(y_combined, lower, upper)

            # loss = mean_rmse + beta * mean_rmse * abs(coverage - 0.95)
            loss = mean_rmse + mean_rmse * abs(coverage - 0.95)

            if loss < self.best_score:
                print(f"Best score updated to {loss} from {self.best_score}")
                self.best_model_study = study
                self.best_feature_mask = full_feature_mask
                self.best_rmse_per_fold = cv_rmse_per_fold
                self.best_mean_rmse = mean_rmse
                self.best_rmse_std = cv_rmse_std
                self.best_score = loss
                self.distances = distances_combined
                self.best_cv_predictions = cv_predictions
                self.X_RSP_optimized = X_combined
                self.y_RSP_optimized = y_combined
                self.best_model = model
                self.selected_indices_base = selected_indices_base
                self.selected_indices_new = selected_indices_new
                self.best_dcp = dcp
                self.model_params = best_params # model_params
                self.pearson_train = pearson_train
                self.pearson_abs_train = pearson_abs_train

                print("Parameters were updated accordingly")
                return loss
            return loss
        return float('inf')

    def select_data(self, new_smiles, new_descriptors, new_y, base_X_3d, base_y):
        """
        Identify relevant structure-property space among new data,
        then run an Optuna-based distance optimization to define in-domain.
        """
        print("Identifying the relevant structure-property space...")

        if not isinstance(new_descriptors, pd.DataFrame):
            new_descriptors = pd.DataFrame(new_descriptors)
                
        print("Processing new dataset...")
        new_fps = self.generate_fingerprints(new_smiles)
        
        # Transform new fingerprints using the same kernel PCA
        X_2d_new = self.pca.transform(new_fps)  # shape: (n_new, 2)
        
        # Scale new_y
        new_y_scaled = self.scaler(new_y)
        # Combine => (kpc1, kpc2, property)
        new_X_3d = np.column_stack([X_2d_new, new_y_scaled])
        
        print(f"Starting optimization with {self.n_trials_distance} trials...")
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: self.objective(
                trial, new_X_3d, new_descriptors, new_y,
                base_X_3d, self.descriptors, base_y
            ),
            n_trials=self.n_trials_distance,
            n_jobs=1,
            callbacks=[lambda study, trial: print_progress(study, trial, self.n_trials_distance)]
        )

        # Retrieve best parameters from the distance optimization
        self.optimal_weights = [study.best_params[f'w{i}'] for i in range(3)]
        self.optimal_threshold = study.best_params['threshold']
        self.optimal_lambda = study.best_params['lambda']
        
        print(f"Selected total of {len(self.selected_indices_base) + len(self.selected_indices_new)}"
            f" ({len(self.selected_indices_base)} samples and {len(self.selected_indices_new)} from "
            "base and new dataset respectively)")
        return study
    
    def define_space_from_target_only(self, target_smiles, training_smiles, training_descriptors, training_y):
        """
        Define structure-property space using target molecules for centroid,
        but only train on training data.
        """
        # Store descriptors
        if isinstance(training_descriptors, pd.DataFrame):
            self.descriptors = training_descriptors.copy()
        else:
            self.descriptors = pd.DataFrame(training_descriptors)
        
        # Generate fingerprints
        target_fps = self.generate_fingerprints(target_smiles)
        training_fps = self.generate_fingerprints(training_smiles)
        
        # Fit kernel PCA on combined data
        combined_fps = np.vstack([target_fps, training_fps])
        self.pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1, random_state=42)
        self.pca.fit(combined_fps)
        
        # Transform both datasets
        X_2d_target = self.pca.transform(target_fps)
        X_2d_training = self.pca.transform(training_fps)
        
        # Scale y using training data
        self.y_mean = np.mean(training_y)
        self.y_std = np.std(training_y)
        y_scaled = (training_y - self.y_mean) / self.y_std
        
        # Define centroid from target molecules (but use 0 for property dimension)
        self.centroid = np.zeros(3)
        self.centroid[:2] = np.mean(X_2d_target, axis=0)
        
        # Create 3D space for training data
        X_3d_training = np.column_stack([X_2d_training, y_scaled])
        self.X_3d_base = X_3d_training
        
        # Reset class attributes
        self.best_score = np.inf
        self.selected_indices_base = None
        self.selected_indices_new = None
        
        return X_3d_training

    def select_data_for_target_space(self, target_smiles, training_smiles, training_descriptors, training_y):
        """
        Modified version of select_data that uses target molecules only for centroid
        """
        # Define the space using target molecules for centroid
        base_X_3d = self.define_space_from_target_only(
            target_smiles, training_smiles, training_descriptors, training_y
        )
        
        # Run optimization
        print(f"Starting optimization with {self.n_trials_distance} trials...")
        study = optuna.create_study(direction='minimize')
        
        # Create a modified objective function specific to this case
        def target_space_objective(trial):
            w0 = trial.suggest_float('w0', 0.1, 1.0)
            w1 = trial.suggest_float('w1', 0.1, 1.0)
            w2 = trial.suggest_float('w2', 0.1, 1.0)
            weights = [w0, w1, w2]
            
            if self.threshold_range != None:
                threshold_lower = self.threshold_range[0]
                threshold_upper = self.threshold_range[1]
                threshold = trial.suggest_float('threshold', threshold_lower, threshold_upper)
            else:
                threshold = trial.suggest_float('threshold', 4.0, 7.0) 
            
            lambda_ = trial.suggest_float('lambda', 0.1, 0.5)
            
            # Calculate distances for the training data
            distances_base = self.get_distances(base_X_3d, weights)
            selected_indices_base = [i for i, dist in enumerate(distances_base) if dist < threshold]
            
            print(f"Trial {trial.number}: Selected {len(selected_indices_base)} molecules with threshold {threshold:.2f}")
            
            # Skip if too few samples selected
            if len(selected_indices_base) < 10:
                return float('inf')
            
            # Process the selected training data directly
            X_selected = self.descriptors.iloc[selected_indices_base].reset_index(drop=True)
            y_selected = training_y[selected_indices_base]
            
            # Remove correlated features
            original_columns = X_selected.columns.tolist()
            X_selected = self.remove_correlated_features(X_selected)
            remaining_columns = X_selected.columns.tolist()
            feature_map = {new_idx: original_columns.index(col) for new_idx, col in enumerate(remaining_columns)}
            
            # Feature selection
            feature_mask = self.perform_feature_selection(X_selected.values, y_selected)
            X_selected_filtered = X_selected.values[:, np.where(feature_mask)[0]]
            
            # Create full feature mask
            full_feature_mask = np.zeros(len(original_columns), dtype=bool)
            if feature_mask is not None:
                for i, is_selected in enumerate(feature_mask):
                    if is_selected:
                        original_idx = feature_map[i]
                        full_feature_mask[original_idx] = True
            
            # Train and evaluate model
            model, best_params, model_study = self.optimize_model(X_selected_filtered, y_selected)
            cv_rmse_per_fold = model_study.best_trial.user_attrs["cv_rmse_per_fold"]
            cv_rmse_std = model_study.best_trial.user_attrs["std_rmse"]
            cv_predictions = model_study.best_trial.user_attrs["cv_predictions"]
            mean_rmse = model_study.best_value
            
            # Errors and correlation
            errors = y_selected - np.array(cv_predictions)
            distances_selected = distances_base[selected_indices_base]
            pearson_train = pearsonr(distances_selected, errors)
            pearson_abs_train = pearsonr(distances_selected, abs(errors))
            
            # Fit conformal predictor
            dcp = DistanceBasedConformalPredictor(confidence=0.95)
            dcp.fit(y_selected, cv_predictions, distances_selected)
            
            # Calculate coverage
            lower, upper = dcp.predict_interval(cv_predictions, lambda_, distances_selected)
            coverage = dcp.get_coverage(y_selected, lower, upper)
            
            # Calculate loss
            loss = mean_rmse + mean_rmse * abs(coverage - 0.95)
            
            print(f"Trial {trial.number}: RMSE = {mean_rmse:.4f}, Coverage = {coverage:.4f}, Loss = {loss:.4f}")
            
            # Update best score and parameters if improved
            if loss < self.best_score:
                print(f"Best score updated to {loss} from {self.best_score}")
                self.best_model_study = model_study
                self.best_feature_mask = full_feature_mask
                self.best_rmse_per_fold = cv_rmse_per_fold
                self.best_mean_rmse = mean_rmse
                self.best_rmse_std = cv_rmse_std
                self.best_score = loss
                self.distances = distances_selected
                self.best_cv_predictions = cv_predictions
                self.X_RSP_optimized = X_selected_filtered
                self.y_RSP_optimized = y_selected
                self.best_model = model
                self.selected_indices_base = selected_indices_base
                self.selected_indices_new = []  # Empty, since we're not using new data
                self.best_dcp = dcp
                self.model_params = best_params
                self.pearson_train = pearson_train
                self.pearson_abs_train = pearson_abs_train
            
            return loss
        
        # Run optimization with the target-specific objective
        study.optimize(
            target_space_objective,
            n_trials=self.n_trials_distance,
            n_jobs=1,
            callbacks=[lambda study, trial: print_progress(study, trial, self.n_trials_distance)]
        )
        
        # Retrieve best parameters
        self.optimal_weights = [study.best_params[f'w{i}'] for i in range(3)]
        self.optimal_threshold = study.best_params['threshold']
        self.optimal_lambda = study.best_params['lambda']
        
        print(f"Selected {len(self.selected_indices_base)} samples from training dataset")
        return study

def method_validation(
    df,
    X,
    y,
    threshold_range = None,
    model_type='rf',
    model_params=None,
    n_seeds=10,
    n_folds=10,
    n_trials_model_optimization=100,
    n_trials_distance=100,
    save=True
):
    """
    Perform a validation procedure that compares:
    1) A "global" model (trained on all training data)
    2) A "reliable-space" model (RSP) that leverages your ReliableSpacePredictor

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least a 'Smiles' column.
    X : pd.DataFrame
        Feature matrix.
    y : array-like
        Target array.
    model_type : str, optional
        Which model type to evaluate {'rf', 'xgb', 'nn'}, by default 'rf'.
    model_params : dict, optional
        Additional parameters for the RSP or model usage, by default None.
    n_seeds : int, optional
        Number of random seeds for repeated train-test splits, by default 10.
    n_folds : int, optional
        Number of cross-validation folds, by default 10.
    n_trials_model_optimization : int, optional
        Number of Optuna trials for model hyperparameters, by default 100.
    n_trials_distance : int, optional
        Number of Optuna trials for RSP distance threshold optimization, by default 100.
    save : bool, optional
        Whether to save trained models, by default True.

    Returns
    -------
    results_global : list of dict
        Each element corresponds to a single seed’s "global" model results.
    results_RSP : list of dict
        Each element corresponds to a single seed’s "RSP" model results.
    """

    results_global = []
    results_RSP = []

    smis = df['Smiles'].to_numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in range(n_seeds):
        print(f"\nProcessing seed {seed + 1}/{n_seeds}")

        # ---------------------------------------------------------------------
        # 1) Train/Test split
        # ---------------------------------------------------------------------
        train_idx, test_idx = train_test_split(
            list(X.index), test_size=0.2, random_state=seed
        )
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y[train_idx]
        y_test = y[test_idx]

        smis_train = smis[train_idx]
        smis_test = smis[test_idx]

        # ---------------------------------------------------------------------
        # 2) Convert training SMILES to 3D embedding (KPC1, KPC2, scaled y)
        # ---------------------------------------------------------------------
        fps = []
        for smiles in smis_train:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fps.append(list(fp.ToBitString()))
        fps = np.array(fps, dtype=int)

        kpc = KernelPCA(n_components=2, kernel='rbf', gamma=0.1, random_state=42)
        X_2d = kpc.fit_transform(fps)

        # Scale y
        y_train_mean = np.mean(y_train)
        y_train_std = np.std(y_train)
        y_train_scaled = (y_train - y_train_mean) / y_train_std
        X_3d = np.column_stack([X_2d, y_train_scaled])

        # ---------------------------------------------------------------------
        # 3) Build NearestNeighbors & identify the densest region (base subset)
        # ---------------------------------------------------------------------
        k = int(0.2 * len(X_train))
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X_3d)
        distances_nn, neighbors_nn = nn.kneighbors(X_3d)

        # Sum distances to the k neighbors (excluding the point itself)
        sum_distances = np.sum(distances_nn[:, 1:], axis=1)
        best_seed_idx = np.argmin(sum_distances)  # densest point

        # 4) The base subset will be the densest point + its neighbors
        _, indices_coherent = nn.kneighbors(X_3d[best_seed_idx].reshape(1, -1))
        base_idx = indices_coherent[0]

        all_idx = set(range(len(X_train)))
        fusion_idx = list(all_idx - set(base_idx))

        X_train_base = X_train.iloc[base_idx].reset_index(drop=True)
        X_train_fusion = X_train.iloc[fusion_idx].reset_index(drop=True)
        y_train_base = y_train[base_idx]
        y_train_fusion = y_train[fusion_idx]
        smis_base = smis_train[base_idx]
        smis_fusion = smis_train[fusion_idx]

        # ---------------------------------------------------------------------
        # Train a "global" model on ALL training data
        # ---------------------------------------------------------------------
        print("Training global model...")

        # Step G1: remove correlated features
        X_train_uncorr = remove_corr_features(X_train)

        # Step G2: feature selection
        rfe_results_global = select_features_rfecv(
            X_train_uncorr, y_train, model_type='regressor', cv_folds=n_folds
        )
        selected_mask_global = rfe_results_global['selected_mask']
        selected_descriptors_global = X_train_uncorr.columns[selected_mask_global].to_list()

        # Step G3: subset train/test to selected features
        X_train_selected_global = X_train_uncorr[selected_descriptors_global]
        X_test_selected_global = X_test[selected_descriptors_global]

        # Step G4: Create an Optuna study & train
        study_global = optuna.create_study(direction='minimize')

        X_train_numpy_global = X_train_selected_global.values
        X_test_numpy_global = X_test_selected_global.values

        if model_type == 'nn':
            X_train_torch_global = torch.tensor(X_train_numpy_global, dtype=torch.float32)
            y_train_torch_global = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            X_test_torch_global = torch.tensor(X_test_numpy_global, dtype=torch.float32)

            # Optimize
            study_global.optimize(
                lambda trial: nn_objective(trial, X_train_torch_global, y_train_torch_global, n_folds),
                n_trials=n_trials_model_optimization
            )
            best_params_global = study_global.best_trial.params
            best_cv_predictions_global = study_global.best_trial.user_attrs["cv_predictions"]

            # Train final
            model_global = train_final_nn(
                best_params_global,
                X_train_torch_global,
                y_train_torch_global,
                f'nn_global_seed_{seed + 1}',
                device,
                save=save
            )
            batch_size_global = best_params_global['batch_size']
            y_test_pred_global = predict_with_model(
                model_global,
                X_test_torch_global,
                device=device,
                batch_size=batch_size_global
            )

        elif model_type == 'xgb':
            study_global.optimize(
                lambda trial: xgboost_objective(trial, X_train_numpy_global, y_train, n_folds),
                n_trials=n_trials_model_optimization
            )
            best_params_global = study_global.best_trial.params
            best_cv_predictions_global = study_global.best_trial.user_attrs["cv_predictions"]

            model_global = xgb.XGBRegressor(**best_params_global)
            model_global.fit(X_train_selected_global, y_train)
            y_test_pred_global = model_global.predict(X_test_numpy_global)

            if save:
                with open(f"xgb_global_seed_{seed + 1}.pkl", "wb") as file:
                    pickle.dump(model_global, file)
        else:
            # RF
            study_global.optimize(
                lambda trial: rf_objective(trial, X_train_numpy_global, y_train, n_folds),
                n_trials=n_trials_model_optimization
            )
            best_params_global = study_global.best_trial.params
            best_cv_predictions_global = study_global.best_trial.user_attrs["cv_predictions"]

            model_global = RandomForestRegressor(**best_params_global)
            model_global.fit(X_train_selected_global, y_train)
            y_test_pred_global = model_global.predict(X_test_numpy_global)

            if save:
                with open(f"rf_global_seed_{seed + 1}.pkl", "wb") as file:
                    pickle.dump(model_global, file)

        # Evaluate "global" performance
        best_trial_global = study_global.best_trial
        test_rmse_global = np.sqrt(mean_squared_error(y_test, y_test_pred_global))
        test_r2_global = r2_score(y_test, y_test_pred_global)

        # ---------------------------------------------------------------------
        # Train a "reliable-space" (RSP) model
        # ---------------------------------------------------------------------
        print("\nTraining a model after Reliable Space Identification...")

        RSP = ReliableSpacePredictor(
            cv_folds=n_folds,
            n_trials_distance=n_trials_distance,
            n_trials_model_optimization=n_trials_model_optimization,
            threshold_range=threshold_range,
            model_type=model_type,
            model_params=model_params
        )

        # Define the base space with the densest cluster
        base_X_3d = RSP.define_space(smis_base, X_train_base, y_train_base)

        # Then select data (base + fusion) inside the RSP domain
        study_RSP = RSP.select_data(
            smis_fusion,
            X_train_fusion,
            y_train_fusion,
            base_X_3d,
            y_train_base
        )
        best_trial_RSP = study_RSP.best_trial

        # Gather RSP training details
        smis_train_selected_RSP = np.concatenate([
            np.array(smis_base)[RSP.selected_indices_base],
            np.array(smis_fusion)[RSP.selected_indices_new]
        ])
        base_X_3d_selected = base_X_3d[RSP.selected_indices_base]

        # Pairwise distances among the base subset after RSP selection
        base_X_3d_selected_pdist = pdist(base_X_3d_selected)
        dict_base_subset_coherence = {
            'avg_pairwise_distance': np.mean(base_X_3d_selected_pdist),
            'max_pairwise_distance': np.max(base_X_3d_selected_pdist),
            'min_pairwise_distance': np.min(base_X_3d_selected_pdist),
            'std_pairwise_distance': np.std(base_X_3d_selected_pdist)
        }

        # Summaries from the RSP
        distances_train_RSP = RSP.distances
        train_rmse_RSP = RSP.best_mean_rmse
        model_RSP = RSP.best_model
        model_params_RSP = RSP.model_params
        n_original_train_data = X_train.shape[0]
        n_selected_train_data = RSP.X_RSP_optimized.shape[0]
        drop_rate = 1 - n_selected_train_data / n_original_train_data

        selected_descriptors_RSP = X_train_base.columns[RSP.best_feature_mask].to_list()

        # Predict on test set
        X_test_selected_RSP = X_test[selected_descriptors_RSP]
        X_test_selected_RSP_torch = torch.tensor(
            X_test_selected_RSP.values.astype(np.float32),
            dtype=torch.float32
        )

        if model_type == 'nn':
            if save:
                save_dict = {
                    'state_dict': model_RSP.state_dict(),
                    'params': model_params_RSP,
                    'input_size': len(selected_descriptors_RSP),
                    'final_rmse': train_rmse_RSP,
                    'training_metadata': {
                        'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'device': str(device),
                        'pytorch_version': torch.__version__
                    }
                }
                torch.save(save_dict, f'nn_RSP_seed_{seed + 1}.pt')
                print(f"Model saved to nn_RSP_seed_{seed + 1}.pt")

            batch_size_RSP = model_params_RSP['batch_size']
            y_test_pred_RSP = predict_with_model(
                model_RSP,
                X_test_selected_RSP_torch,
                device=device,
                batch_size=batch_size_RSP
            )
        elif model_type == 'xgb':
            y_test_pred_RSP = model_RSP.predict(X_test_selected_RSP)
            if save:
                with open(f"xgb_RSP_seed_{seed + 1}.pkl", "wb") as file:
                    pickle.dump(model_RSP, file)
        else:
            # RF
            y_test_pred_RSP = model_RSP.predict(X_test_selected_RSP)
            if save:
                with open(f"rf_RSP_seed_{seed + 1}.pkl", "wb") as file:
                    pickle.dump(model_RSP, file)

        test_rmse_RSP_overall = np.sqrt(mean_squared_error(y_test, y_test_pred_RSP))
        test_r2_RSP_overall = r2_score(y_test, y_test_pred_RSP)

        # ---------------------------------------------------------------------
        # Conformal Analysis & Classifier for domain approach
        # ---------------------------------------------------------------------
        dcp = RSP.best_dcp
        lambda_RSP = RSP.optimal_lambda
        Reff = RSP.optimal_threshold
        weights_distance = RSP.optimal_weights

        # Prepare test for conformal intervals
        fps_test = RSP.generate_fingerprints(smis_test)
        X_2d_test = RSP.pca.transform(fps_test)
        y_test_scaled = RSP.scaler(y_test)
        y_test_pred_RSP_scaled = RSP.scaler(y_test_pred_RSP)
        y_test_pred_global_scaled = RSP.scaler(y_test_pred_global)

        X_3d_test = np.column_stack([X_2d_test, y_test_scaled])
        X_3d_test_pred_RSP = np.column_stack([X_2d_test, y_test_pred_RSP_scaled])
        X_3d_test_pred_global = np.column_stack([X_2d_test, y_test_pred_global_scaled])
        X_3d_test_pred_global_df = pd.DataFrame({'PC1': X_3d_test_pred_global[:,0], 'PC2': X_3d_test_pred_global[:,1], 'y_hat': X_3d_test_pred_global[:,2]})

        distances_test = RSP.get_distances(X_3d_test, weights_distance)
        distances_test_pred_RSP = RSP.get_distances(X_3d_test_pred_RSP, weights_distance)
        distances_test_pred_global = RSP.get_distances(X_3d_test_pred_global, weights_distance)

        lower_dcp, upper_dcp = dcp.predict_interval(y_test_pred_RSP, lambda_RSP, distances_test_pred_RSP)
        coverage_dcp = dcp.get_coverage(y_test, lower_dcp, upper_dcp)

        fcp = FixedConformalPredictor(confidence=0.95)
        fcp.fit(RSP.y_RSP_optimized, RSP.best_cv_predictions)
        lower_fcp, upper_fcp = fcp.predict_interval(y_test_pred_RSP)

        # Prepare data for classifier
        fps_train = RSP.generate_fingerprints(smis_train)
        X_2d_train = RSP.pca.transform(fps_train)
        y_train_scaled = RSP.scaler(y_train)
        y_train_pred_scaled = RSP.scaler(best_cv_predictions_global)
        X_3d_train = np.column_stack([X_2d_train, y_train_scaled])
        X_3d_train_pred_global = np.column_stack([X_2d_train, y_train_pred_scaled])
        X_3d_train_pred_global_df = pd.DataFrame({'PC1': X_3d_train_pred_global[:,0], 'PC2': X_3d_train_pred_global[:,1], 'y_hat': X_3d_train_pred_global[:,2]})
        distances_train = RSP.get_distances(X_3d_train, weights_distance)
        distances_train_pred_global = RSP.get_distances(X_3d_train_pred_global, weights_distance)

        X_train_uncorr_overall = pd.concat([X_train_uncorr, X_3d_train_pred_global_df], axis = 1)
        X_test_overall = pd.concat([X_test, X_3d_test_pred_global_df], axis = 1)

        # Mark each training sample as inside/outside the domain
        y_train_labels = (distances_train <= Reff).astype(int)
        y_test_labels = (distances_test <= Reff).astype(int)

        # Feature selection for classifier
        rfe_results_classifier = select_features_rfecv(
            X_train_uncorr_overall,
            y_train_labels,
            model_type='classifier',
            cv_folds=n_folds
        )
        selected_mask_classifier = rfe_results_classifier['selected_mask']
        selected_descriptors_classifier = X_train_uncorr_overall.columns[selected_mask_classifier].to_list()

        X_train_selected_classifier = X_train_uncorr_overall[selected_descriptors_classifier]
        X_test_selected_classifier = X_test_overall[selected_descriptors_classifier]

        # Optimize classifier
        study_classifier = optuna.create_study(direction="maximize")
        study_classifier.optimize(
            lambda trial: rf_objective_classifier(
                trial,
                X_train_selected_classifier.values,
                y_train_labels,
                n_folds
            ),
            n_trials=n_trials_model_optimization
        )
        best_params_classifier = study_classifier.best_trial.params
        model_classifier = RandomForestClassifier(**best_params_classifier)
        model_classifier.fit(X_train_selected_classifier.values, y_train_labels)

        y_train_prob = model_classifier.predict_proba(X_train_selected_classifier.values)[:, 1]
        y_test_prob = model_classifier.predict_proba(X_test_selected_classifier.values)[:, 1]
        custom_thresh = 0.5
        y_test_labels_pred = (y_test_prob >= custom_thresh).astype(int)

        if save:
            with open(f"rf_classifier_seed_{seed + 1}.pkl", "wb") as file:
                pickle.dump(model_classifier, file)


        # Step 1: Build calibration DataFrame
        threshold_calibration_df = pd.DataFrame({
            'distance': distances_train_pred_global,
            'true_label': y_train_labels,
            'pred_prob': y_train_prob
        })

        # Step 2: Bin distances
        threshold_calibration_df['bin'] = pd.qcut(threshold_calibration_df['distance'], q=4, duplicates='drop')

        # Step 3: Calibrate threshold per bin
        # Use low percentile of true in-domain predictions → lenient near center, stricter farther out
        bin_stats = threshold_calibration_df.groupby('bin').apply(
            lambda g: pd.Series({
                'mean_distance': g['distance'].mean(),
                'threshold': np.percentile(g[g['true_label'] == 1]['pred_prob'], 10)
                            if not g[g['true_label'] == 1].empty else 0.5
            })
        ).reset_index(drop=True)

        x_bin = bin_stats['mean_distance'].values
        y_bin = bin_stats['threshold'].values
        
        min_d = np.min(distances_train_pred_global)
        max_d = np.max(distances_train_pred_global)
        
        study_thresh = optuna.create_study(direction='maximize')
        study_thresh.optimize(lambda trial: threshold_objective(
                trial,
                distances_train_pred_global, 
                min_d, 
                max_d, 
                y_train_prob, 
                y_train_labels), 
                n_trials=50)

        # Use optimized thresholds
        best_min_thresh = study_thresh.best_params['min_thresh']
        best_max_thresh = study_thresh.best_params['max_thresh']

        # Step 4: Fit a linear interpolation function and clip it
        thresholds_test = trust_threshold(distances_test_pred_global, min_d, max_d, min_thresh=best_min_thresh, max_thresh=best_max_thresh)
        thresholds_test = np.clip(thresholds_test, best_min_thresh, best_max_thresh) 

        # Step 5: Dynamic classification
        y_test_labels_pred_bin = (y_test_prob >= thresholds_test).astype(int)

        # ---------------------------------------------------------------------
        # Various coverage / confusion-matrix metrics
        # ---------------------------------------------------------------------
        # 1) Distance-based approach (global model pred. distances vs true)
        TP1 = FP1 = TN1 = FN1 = 0
        for i in range(len(distances_test)):
            dist_true = distances_test[i]
            dist_pred_global = distances_test_pred_global[i]
            if dist_true < Reff and dist_pred_global < Reff:
                TP1 += 1
            elif dist_true > Reff and dist_pred_global < Reff:
                FP1 += 1
            elif dist_true > Reff and dist_pred_global > Reff:
                TN1 += 1
            else:
                FN1 += 1
        distance_acc = (TP1 + TN1) / (TP1 + TN1 + FP1 + FN1)

        # 2) Classifier approach
        TP2 = FP2 = TN2 = FN2 = 0
        for i in range(len(distances_test)):
            dist_true = distances_test[i]
            pred_cls = y_test_labels_pred[i]
            if dist_true < Reff and pred_cls == 1:
                TP2 += 1
            elif dist_true > Reff and pred_cls == 1:
                FP2 += 1
            elif dist_true > Reff and pred_cls == 0:
                TN2 += 1
            else:
                FN2 += 1
        classifier_acc = (TP2 + TN2) / (TP2 + TN2 + FP2 + FN2)

        # 3) Distance-aware threshold Classifier approach
        TP3 = FP3 = TN3 = FN3 = 0
        for i in range(len(distances_test)):
            dist_true = distances_test[i]
            pred_cls = y_test_labels_pred_bin[i]
            if dist_true < Reff and pred_cls == 1:
                TP3 += 1
            elif dist_true > Reff and pred_cls == 1:
                FP3 += 1
            elif dist_true > Reff and pred_cls == 0:
                TN3 += 1
            else:
                FN3 += 1
        classifier_bin_acc = (TP3 + TN3) / (TP3 + TN3 + FP3 + FN3)

        # 4) Distance-based and Classifier approaches
        TP4 = FP4 = TN4 = FN4 = 0
        for i in range(len(distances_test)):
            dist_true = distances_test[i]
            dist_pred_global = distances_test_pred_global[i]
            pred_cls = y_test_labels_pred[i]
            combined_decision = (dist_pred_global < Reff) and (pred_cls == 1)
            if dist_true < Reff and combined_decision:
                TP4 += 1
            elif dist_true > Reff and combined_decision:
                FP4 += 1
            elif dist_true > Reff and not combined_decision:
                TN4 += 1
            else:
                FN4 += 1
        combined_acc_1 = (TP4 + TN4) / (TP4 + TN4 + FP4 + FN4)
        
        # 5) Distance-based and Distance-aware threshold Classifier approaches
        TP5 = FP5 = TN5 = FN5 = 0
        for i in range(len(distances_test)):
            dist_true = distances_test[i]
            dist_pred_global = distances_test_pred_global[i]
            pred_cls = y_test_labels_pred_bin[i]
            combined_decision = (dist_pred_global < Reff) and (pred_cls == 1)
            if dist_true < Reff and combined_decision:
                TP5 += 1
            elif dist_true > Reff and combined_decision:
                FP5 += 1
            elif dist_true > Reff and not combined_decision:
                TN5 += 1
            else:
                FN5 += 1
        combined_acc_2 = (TP5 + TN5) / (TP5 + TN5 + FP5 + FN5)
    
        # 6) Classifier and Distance-aware threshold Classifier approaches
        TP6 = FP6 = TN6 = FN6 = 0
        for i in range(len(distances_test)):
            dist_true = distances_test[i]
            pred_cls_1 = y_test_labels_pred[i]
            pred_cls_2 = y_test_labels_pred_bin[i]
            combined_decision = (pred_cls_1 == 1) and (pred_cls_2 == 1)
            if dist_true < Reff and combined_decision:
                TP6 += 1
            elif dist_true > Reff and combined_decision:
                FP6 += 1
            elif dist_true > Reff and not combined_decision:
                TN6 += 1
            else:
                FN6 += 1
        combined_acc_3 = (TP6 + TN6) / (TP6 + TN6 + FP6 + FN6)
    
        # 7) Combined
        TP7 = FP7 = TN7 = FN7 = 0
        for i in range(len(distances_test)):
            dist_true = distances_test[i]
            dist_pred_global = distances_test_pred_global[i]
            pred_cls_1 = y_test_labels_pred[i]
            pred_cls_2 = y_test_labels_pred_bin[i]
            combined_decision = (dist_pred_global < Reff) and (pred_cls_1 == 1) and (pred_cls_2 == 1)
            if dist_true < Reff and combined_decision:
                TP7 += 1
            elif dist_true > Reff and combined_decision:
                FP7 += 1
            elif dist_true > Reff and not combined_decision:
                TN7 += 1
            else:
                FN7 += 1
        combined_acc_4 = (TP7 + TN7) / (TP7 + TN7 + FP7 + FN7)


        # ---------------------------------------------------------------------
        # Summaries: Precision, TPR, FPR, coverage, etc.
        # ---------------------------------------------------------------------
        distance_precision = TP1 / (TP1 + FP1) if (TP1 + FP1) else np.nan
        classifier_precision = TP2 / (TP2 + FP2) if (TP2 + FP2) else np.nan
        classifier_bin_precision = TP3 / (TP3 + FP3) if (TP3 + FP3) else np.nan
        combined_precision_1 = TP4 / (TP4 + FP4) if (TP4 + FP4) else np.nan
        combined_precision_2 = TP5 / (TP5 + FP5) if (TP5 + FP5) else np.nan
        combined_precision_3 = TP6 / (TP6 + FP6) if (TP6 + FP6) else np.nan
        combined_precision_4 = TP7 / (TP7 + FP7) if (TP7 + FP7) else np.nan

        # True Positive Rate, False Positive Rate, and F1
        tpr_list = [
            TP1 / (TP1 + FN1) if (TP1 + FN1) else 0,
            TP2 / (TP2 + FN2) if (TP2 + FN2) else 0,
            TP3 / (TP3 + FN3) if (TP3 + FN3) else 0,
            TP4 / (TP4 + FN4) if (TP4 + FN4) else 0,
            TP5 / (TP5 + FN5) if (TP5 + FN5) else 0,
            TP6 / (TP6 + FN6) if (TP6 + FN6) else 0,
            TP7 / (TP7 + FN7) if (TP7 + FN7) else 0,
        ]
        fpr_list = [
            FP1 / (FP1 + TN1) if (FP1 + TN1) else 0,
            FP2 / (FP2 + TN2) if (FP2 + TN2) else 0,
            FP3 / (FP3 + TN3) if (FP3 + TN3) else 0,
            FP4 / (FP4 + TN4) if (FP4 + TN4) else 0,
            FP5 / (FP5 + TN5) if (FP5 + TN5) else 0,
            FP6 / (FP6 + TN6) if (FP6 + TN6) else 0,
            FP7 / (FP7 + TN7) if (FP7 + TN7) else 0,
        ]
        f1_scores = [
            2 * (tpr * (1 - fpr)) / (tpr + (1 - fpr)) if (tpr + (1 - fpr)) else 0
            for tpr, fpr in zip(tpr_list, fpr_list)
        ]
        
        indices_true_within_Reff = [
            i for i, d in enumerate(distances_test) if d < Reff
        ]
        indices_pred_distance_within_Reff = [ # distance-based
            i for i in range(len(distances_test))
            if distances_test_pred_global[i] < Reff
        ]
        indices_pred_clf_within_Reff = [ # regular RF classifier
            i for i in range(len(distances_test))
            if y_test_labels_pred[i] == 1
        ]
        indices_pred_clf_bin_within_Reff = [ # distance-aware clf
            i for i in range(len(distances_test))
            if y_test_labels_pred_bin[i] == 1
        ]
        indices_pred_combined_1_within_Reff = [ # distance-based and regular clf
            i for i in range(len(distances_test))
            if distances_test_pred_global[i] < Reff and y_test_labels_pred[i] == 1
        ]
        indices_pred_combined_2_within_Reff = [ # distance-based and distance-aware clf
            i for i in range(len(distances_test))
            if distances_test_pred_global[i] < Reff and y_test_labels_pred_bin[i] == 1
        ]
        indices_pred_combined_3_within_Reff = [ # regular clf and distance-aware clf
            i for i in range(len(distances_test))
            if y_test_labels_pred[i] == 1 and y_test_labels_pred_bin[i] == 1
        ]
        indices_pred_combined_4_within_Reff = [ # all three
            i for i in range(len(distances_test))
            if distances_test_pred_global[i] < Reff and y_test_labels_pred[i] == 1 and y_test_labels_pred_bin[i] == 1
        ]
        
        # Coverage
        coverage_vals = [
            len(indices_pred_distance_within_Reff)/len(y_test),                # coverage from distance-only approach
            len(indices_pred_clf_within_Reff)/len(y_test), 
            len(indices_pred_clf_bin_within_Reff)/len(y_test),
            len(indices_pred_combined_1_within_Reff)/len(y_test), 
            len(indices_pred_combined_2_within_Reff)/len(y_test), 
            len(indices_pred_combined_3_within_Reff)/len(y_test), 
            len(indices_pred_combined_4_within_Reff)/len(y_test),  
        ]

        classification_results = pd.DataFrame({
            'Approach': ['Distance', 'Classifier', 'Distance-aware Classifier', 'Combined 1', 'Combined 2', 'Combined 3', 'Combined 4'],
            'Accuracy': [distance_acc, classifier_acc, classifier_bin_acc, combined_acc_1, combined_acc_2, combined_acc_3, combined_acc_4],
            'Precision': [distance_precision, classifier_precision, classifier_bin_precision, combined_precision_1, combined_precision_2, combined_precision_3, combined_precision_4],
            'TPR': tpr_list,
            'FPR': fpr_list,
            'F1 Score': f1_scores,
            'Coverage': coverage_vals
        })

        # Performance by subsets
        subset_indices = {
            'true': indices_true_within_Reff,
            'distance': indices_pred_distance_within_Reff,
            'classifier': indices_pred_clf_within_Reff,
            'distance_bin': indices_pred_clf_bin_within_Reff,
            'combined_1': indices_pred_combined_1_within_Reff,
            'combined_2': indices_pred_combined_2_within_Reff,
            'combined_3': indices_pred_combined_3_within_Reff,
            'combined_4': indices_pred_combined_4_within_Reff
        }

        # RMSE & R²
        performance_by_subset = {}
        for name, idx in subset_indices.items():
            if len(idx) == 0:
                performance_by_subset[name] = {
                    'r2_global': np.nan, 'r2_RSP': np.nan,
                    'rmse_global': np.nan, 'rmse_RSP': np.nan
                }
                continue
            y_sub = y_test[idx]
            performance_by_subset[name] = {
                'r2_global': r2_score(y_sub, y_test_pred_global[idx]),
                'r2_RSP': r2_score(y_sub, y_test_pred_RSP[idx]),
                'rmse_global': np.sqrt(mean_squared_error(y_sub, y_test_pred_global[idx])),
                'rmse_RSP': np.sqrt(mean_squared_error(y_sub, y_test_pred_RSP[idx]))
            }

        # Conformal accuracy metrics
        conformal_accuracy_metrics = {}
        for name, idx in subset_indices.items():
            if len(idx) == 0:
                conformal_accuracy_metrics[name] = {
                    'coverage_dcp': np.nan,
                    'coverage_fcp': np.nan,
                    'dcp_interval_mean_width': np.nan,
                    'dcp_interval_width_std': np.nan,
                    'fcp_interval_mean_width': np.nan
                }
                continue
            y_sub = y_test[idx]
            conformal_accuracy_metrics[name] = {
                'coverage_dcp': np.mean((y_sub >= lower_dcp[idx]) & (y_sub <= upper_dcp[idx])),
                'coverage_fcp': np.mean((y_sub >= lower_fcp[idx]) & (y_sub <= upper_fcp[idx])),
                'dcp_interval_mean_width': np.mean(upper_dcp[idx] - lower_dcp[idx]),
                'dcp_interval_width_std': np.std(upper_dcp[idx] - lower_dcp[idx]),
                'fcp_interval_mean_width': np.mean(upper_fcp[idx] - lower_fcp[idx])
            }

        # ---------------------------------------------------------------------
        # Distance vs. error analysis
        # ---------------------------------------------------------------------

        idx_true = subset_indices['true']

        if len(idx_true) > 0:
            y_true_true = y_test[idx_true]
            y_pred_rsp_true = y_test_pred_RSP[idx_true]
            distances_true = distances_test[idx_true]

            errors_true = y_true_true - y_pred_rsp_true
            abs_errors_true = np.abs(errors_true)

            pearson_true = pearsonr(distances_true, errors_true)[0]
            pearson_true_abs = pearsonr(distances_true, abs_errors_true)[0]
        else:
            errors_true = abs_errors_true = distances_true = None
            pearson_true = pearson_true_abs = np.nan

        distance_error_analysis = {
            'pearson_train_abs': RSP.pearson_abs_train[0],
            'errors_test': errors_true,
            'distances_test': distances_true,
            'pearson_test_signed': pearson_true,
            'pearson_test_abs': pearson_true_abs
        }

        # ---------------------------------------------------------------------
        # Collect & Store in dictionaries
        # ---------------------------------------------------------------------
        # 1) Global dictionary (minimal duplication)
        global_dict = {
            'seed': seed,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'model_type': model_type,
            'model': model_global,
            'optuna_study': study_global,
            'best_trial': best_trial_global,
            'best_params': best_params_global,
            'feature_selection': {
                'selected_descriptors': selected_descriptors_global,
                'rfe_results': rfe_results_global,
            },
            'performance': {
            key: {
                'rmse': performance_by_subset[key]['rmse_global'],
                'r2': performance_by_subset[key]['r2_global']
            }
            for key in performance_by_subset.keys()
            }
        }

        # 2) RSP dictionary
        rsp_dict = {
            'seed': seed,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'model_type': model_type,
            'RSP_object': RSP,
            'distance_optimization_study': study_RSP,
            'best_trial_distance': best_trial_RSP,
            'model_optimization_study': RSP.best_model_study,
            'classifier_optimization_study': study_classifier,
            'model': model_RSP,
            'classifier': model_classifier,
            'best_distance_params': study_RSP.best_params,
            'selected_data': {
                'base_indices': RSP.selected_indices_base,
                'fusion_indices': RSP.selected_indices_new,
                'smiles_train': smis_train_selected_RSP,
                'smis_test': smis_test,
                'drop_rate': drop_rate,
            },
            'selected_features_regressor': {
                'descriptors': selected_descriptors_RSP,
                'mask': RSP.best_feature_mask,
            },
            'selected_features_classifier': {
                'descriptors': selected_descriptors_classifier
            },
            'performance': {
                key: {
                    'rmse': performance_by_subset[key]['rmse_RSP'],
                    'r2': performance_by_subset[key]['r2_RSP']
                }
                for key in performance_by_subset.keys()
            },
            'conformal': {
                'dcp': dcp,
                'lower_dcp': lower_dcp,
                'upper_dcp': upper_dcp,
                'coverage_dcp_full_test': coverage_dcp,
                'conformal_accuracy_metrics_combined': conformal_accuracy_metrics
            },
            'classification_results': classification_results,
            'distance_error_analysis': distance_error_analysis,
            'base_subset_coherence': dict_base_subset_coherence,
            'subset_indices': subset_indices
        }

        # Append results to the corresponding lists
        results_global.append(global_dict)
        results_RSP.append(rsp_dict)

    return results_global, results_RSP

