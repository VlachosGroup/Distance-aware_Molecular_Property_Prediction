# Common utilities (seed setting, etc.)
import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score,  cross_val_predict, KFold, StratifiedKFold, train_test_split
from joblib import cpu_count

def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

NUM_CORES = cpu_count()  # Define once

def select_features_rfecv(X, y, model_type='regressor', n_top_features=30, cv_folds=5, random_state=42):
    """
    Two-step feature selection:
    1. Select top N features using RandomForest importance
    2. Apply RFECV on those features
    
    Works for both regression and classification by setting `model_type='regressor'` or `model_type='classifier'`.
    """
    try:
        # Input validation and conversion
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_numpy = X.values
        else:
            X_numpy = np.array(X)
            feature_names = [f"Feature_{i}" for i in range(X_numpy.shape[1])]

        y_numpy = np.array(y)
        
        # Ensure classification labels are integer-encoded
        if model_type == 'classifier':
            y_numpy = y_numpy.astype(int)

        print(f"Input shape: {X_numpy.shape}")

        # Step 1: Initial feature selection using RandomForest importance
        if model_type == 'regressor':
            print(f"Step 1: Selecting top {n_top_features} features using RandomForestRegressor importance...")
            rf_initial = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
        elif model_type == 'classifier':
            print(f"Step 1: Selecting top {n_top_features} features using RandomForestClassifier importance...")
            rf_initial = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        else:
            raise ValueError("Invalid model_type. Choose 'regressor' or 'classifier'.")

        rf_initial.fit(X_numpy, y_numpy)
        
        # Get feature importances and select top N features
        importances = rf_initial.feature_importances_
        top_indices = np.argsort(importances)[::-1][:n_top_features]  # Top N features

        # Reduce feature set
        X_reduced = X_numpy[:, top_indices]
        reduced_feature_names = [feature_names[i] for i in top_indices]  # Keep selected feature names

        print(f"Reduced shape: {X_reduced.shape}")

        # Step 2: Apply RFECV on reduced feature set
        print(f"Step 2: Applying RFECV on reduced feature set...")
        if model_type == 'regressor':
            rf = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            scoring = 'neg_mean_squared_error'
        else:
            rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            # scoring = 'accuracy'
            scoring = 'f1_weighted'

        rfe = RFECV(
            estimator=rf,
            step=1,
            cv=cv,
            scoring=scoring,
            min_features_to_select=1,  # Prevents errors if all features are eliminated
            n_jobs=-1
        )
        rfe.fit(X_reduced, y_numpy)

        # Create final feature mask
        final_mask = np.zeros(X_numpy.shape[1], dtype=bool)
        final_selected_features = np.array(reduced_feature_names)[rfe.support_]  # Get names of selected features
        final_mask[top_indices[rfe.support_]] = True  # Apply mask to original feature set

        # Create results DataFrame
        results = pd.DataFrame({
            'Feature': feature_names,
            'Selected': final_mask,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        print(f"\nNumber of features selected: {sum(final_mask)}")
        print("\nSelected features:", list(final_selected_features))
        
        return {
            'selected_mask': final_mask,
            'cv_scores': rfe.cv_results_['mean_test_score'],
            'rankings': rfe.ranking_,
            'results_df': results,
            'n_features': sum(final_mask),
            'selected_features': list(final_selected_features)
        }
        
    except Exception as e:
        print(f"Error during feature selection: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def remove_corr_features(Xdata,corr_cutoff = 0.75):
    """
    This function will drop highly correlated features
    Output: a pd.Dataframe 
    """
    cor_matrix=Xdata.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool_))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_cutoff)]
    Xdata=Xdata.drop(columns=to_drop,axis=1)
    
    return Xdata

def weighted_distance(point, reference, weights):
    diff = point - reference
    return np.sqrt(np.sum(weights * diff**2))

def get_pairwise_distance_matrix(X_3d, weights):
    """
    X_3d: (n_samples, n_dims)
    weights: (n_dims,)
    """
    n_samples = len(X_3d)
    dist_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            dist_ij = weighted_distance(X_3d[i], X_3d[j], weights)
            dist_matrix[i, j] = dist_ij
            dist_matrix[j, i] = dist_ij
    return dist_matrix

def trust_threshold(d, min_d, max_d, min_thresh=0.4, max_thresh=0.85):
    d_norm = (d - min_d) / (max_d - min_d)
    return np.clip(min_thresh + d_norm * (max_thresh - min_thresh), min_thresh, max_thresh)

def threshold_objective(trial, d, min_d, max_d, y_train_prob, y_train_labels):
    min_thresh = trial.suggest_float('min_thresh', 0.5, 0.6) # lenient on shorter distances
    max_thresh = trial.suggest_float('max_thresh', 0.7, 0.8)

    # 1. Apply threshold on training distances
    thresholds_train = trust_threshold(d, min_d, max_d, min_thresh, max_thresh)
    
    # 2. Generate predictions
    y_train_pred = (y_train_prob >= thresholds_train).astype(int)

    # 3. Evaluate (on train set)
    return f1_score(y_train_labels, y_train_pred)


