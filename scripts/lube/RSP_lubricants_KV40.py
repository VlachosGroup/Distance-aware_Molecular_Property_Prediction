import sys
import os

try:
    # If running from a .py file
    base_path = os.path.dirname(__file__)
except NameError:
    # If running in Jupyter Notebook
    base_path = os.getcwd()

project_root = os.path.abspath(os.path.join(base_path, "..", ".."))
sys.path.append(project_root)

from rsp.ReliableSpacePredictor import *
import matplotlib
from matplotlib.ticker import MaxNLocator

df_train = pd.read_excel('../../Data/lubricants/kv40-train.xlsx')
df_test = pd.read_excel('../../Data/lubricants/kv40-test.xlsx')
X_train = pd.read_csv('../../Data/lubricants/kv40-train_descriptors.csv')
X_test = pd.read_csv('../../Data/lubricants/kv40-test_descriptors.csv')

# Find common columns
common_cols = X_test.columns.intersection(X_train.columns)

# Keep only the common columns
X_train = X_train[common_cols]

# y_vals
y_train = df_train['KV40 (cSt)'].to_numpy()
y_test = df_test['KV40 (cSt)'].to_numpy()

# smis
smis_train = df_train['Smiles'].to_numpy()
smis_test = df_test['Smiles'].to_numpy()

X_train_uncorr = remove_corr_features(X_train)
rfe_results_global = select_features_rfecv(X_train_uncorr, y_train, model_type='regressor', cv_folds=10)
selected_descriptors_global = X_train_uncorr.columns[rfe_results_global['selected_mask']].to_list()
X_test_selected_global = X_test[selected_descriptors_global]
X_train_selected_global = X_train_uncorr[selected_descriptors_global]
X_train_selected_global_numpy = X_train_selected_global.values
X_test_selected_global_numpy = X_test_selected_global.values
study_global = optuna.create_study(direction='minimize')
study_global.optimize(lambda trial: xgboost_objective(trial, X_train_selected_global_numpy, y_train, 10), n_trials=100)
best_params_global = study_global.best_trial.params
model_global = xgb.XGBRegressor(**best_params_global)
model_global.fit(X_train_selected_global, y_train)
y_test_pred_global = model_global.predict(X_test_selected_global_numpy)
test_rmse_global = np.sqrt(mean_squared_error(y_test, y_test_pred_global))

to_save_global = {
    'X_train_uncorr': X_train_uncorr,
    'rfe_results_global': rfe_results_global,
    'selected_descriptors_global': selected_descriptors_global,
    'X_test_selected_global': X_test_selected_global,
    'X_train_selected_global': X_train_selected_global,
    'X_train_selected_global_numpy': X_train_selected_global_numpy,
    'X_test_selected_global_numpy': X_test_selected_global_numpy,
    'study_global': study_global,
    'best_params_global': best_params_global,
    'model_global': model_global,
    'y_test_pred_global': y_test_pred_global,
    'test_rmse_global': test_rmse_global
}

with open('lubricants_global_model_pipeline.pkl', 'wb') as f:
    pickle.dump(to_save_global, f)

RSP = ReliableSpacePredictor(cv_folds=10, 
                             n_trials_distance = 10, 
                             n_trials_model_optimization=100,
                             threshold_range=[1.0, 2.0], 
                             model_type='xgb')
base_3D = RSP.define_space_from_target_only(smis_test, smis_train, X_train, y_train)
best_study = RSP.select_data_for_target_space(smis_test, smis_train, X_train, y_train)
distances_train_RSP = RSP.distances
train_rmse_RSP = RSP.best_mean_rmse
model_RSP = RSP.best_model

model_params_RSP = RSP.model_params
n_original_train_data = X_train.shape[0]
n_selected_train_data = RSP.X_RSP_optimized.shape[0]
drop_rate = 1 - n_selected_train_data/n_original_train_data
selected_descriptors_RSP = X_train.columns[RSP.best_feature_mask].to_list()
y_train_RSP_optimized = RSP.y_RSP_optimized
X_test_selected_RSP = X_test[selected_descriptors_RSP]

y_test_pred_RSP = model_RSP.predict(X_test_selected_RSP)
test_rmse_RSP = np.sqrt(mean_squared_error(y_test, y_test_pred_RSP))

Reff = RSP.optimal_threshold
dcp = RSP.best_dcp
weights_distance = RSP.optimal_weights

fps_test = RSP.generate_fingerprints(smis_test)
X_2d_test = RSP.pca.transform(fps_test)
y_test_scaled = RSP.scaler(y_test)
y_test_pred_RSP_scaled = RSP.scaler(y_test_pred_RSP)
y_test_pred_global_scaled = RSP.scaler(y_test_pred_global)
X_3d_test = np.column_stack([X_2d_test, y_test_scaled])
X_3d_test_pred_RSP = np.column_stack([X_2d_test, y_test_pred_RSP_scaled])
X_3d_test_pred_global = np.column_stack([X_2d_test, y_test_pred_global_scaled])     
distances_test = RSP.get_distances(X_3d_test, weights_distance)
distances_test_pred_RSP = RSP.get_distances(X_3d_test_pred_RSP, weights_distance)
distances_test_pred_global = RSP.get_distances(X_3d_test_pred_global, weights_distance)
lambda_RSP = RSP.optimal_lambda
lower_dcp, upper_dcp = dcp.predict_interval(y_test_pred_RSP, lambda_RSP, distances_test_pred_RSP)
coverage_dcp = dcp.get_coverage(y_test, lower_dcp, upper_dcp)

fcp = FixedConformalPredictor(confidence=0.95)
fcp.fit(y_train_RSP_optimized, RSP.best_cv_predictions)
lower_fcp, upper_fcp = fcp.predict_interval(y_test_pred_RSP)
coverage_fcp = fcp.get_coverage(y_test, lower_fcp, upper_fcp)

# --- SAVE ALL VARIABLES USING PICKLE ---
to_save_local = {
    'RSP': RSP,
    'base_3D': base_3D,
    'best_study': best_study,
    'distances_train_RSP': distances_train_RSP,
    'train_rmse_RSP': train_rmse_RSP,
    'model_RSP': model_RSP,
    'model_params_RSP': model_params_RSP,
    'n_original_train_data': n_original_train_data,
    'n_selected_train_data': n_selected_train_data,
    'drop_rate': drop_rate,
    'selected_descriptors_RSP': selected_descriptors_RSP,
    'y_train_RSP_optimized': y_train_RSP_optimized,
    'X_test_selected_RSP': X_test_selected_RSP,
    'y_test_pred_RSP': y_test_pred_RSP,
    'test_rmse_RSP': test_rmse_RSP,
    'Reff': Reff,
    'dcp': dcp,
    'weights_distance': weights_distance,
    'fps_test': fps_test,
    'X_2d_test': X_2d_test,
    'y_test_scaled': y_test_scaled,
    'y_test_pred_RSP_scaled': y_test_pred_RSP_scaled,
    'y_test_pred_global_scaled': y_test_pred_global_scaled,
    'X_3d_test': X_3d_test,
    'X_3d_test_pred_RSP': X_3d_test_pred_RSP,
    'X_3d_test_pred_global': X_3d_test_pred_global,
    'distances_test': distances_test,
    'distances_test_pred_RSP': distances_test_pred_RSP,
    'distances_test_pred_global': distances_test_pred_global,
    'lambda_RSP': lambda_RSP,
    'lower_dcp': lower_dcp,
    'upper_dcp': upper_dcp,
    'coverage_dcp': coverage_dcp,
    'fcp': fcp,
    'lower_fcp': lower_fcp,
    'upper_fcp': upper_fcp,
    'coverage_fcp': coverage_fcp
}

with open('lubricants_local_model_pipeline.pkl', 'wb') as f:
    pickle.dump(to_save_local, f)
