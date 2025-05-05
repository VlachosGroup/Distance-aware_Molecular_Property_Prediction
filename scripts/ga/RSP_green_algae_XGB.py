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

df = pd.read_csv('../../Data/green_algae/green_algae_ECOTOX.csv')
y = df['-LOG10(mol/L)'].to_numpy()
X = pd.read_csv('../../Data/green_algae/descriptors_green_algae_ECOTOX.csv')

results_global, results_RSP = method_validation(df, X, y, model_type='xgb', threshold_range=[4.0, 7.0], model_params=None, n_seeds=100, n_folds=10, n_trials_model_optimization = 50, n_trials_distance=10, save=True)

results = [results_global, results_RSP]

# Open a file in binary write mode
with open('RSP_results_ga_xgb_041425.pkl', 'wb') as f:
    # Dump the data to the file
    pickle.dump(results, f)