# Scientific computing
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from joblib import Parallel, delayed, cpu_count
from datetime import datetime
import pandas as pd

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

# Optuna for hyperparameter optimization
import optuna
from optuna.trial import Trial
from optuna.exceptions import TrialPruned
import warnings
warnings.simplefilter("ignore", UserWarning)  # Suppress all user warnings globally


# Logging (optional but recommended)
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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

# Constants
NUM_CORES = cpu_count()  # Use all available cores for parallel processing

class Net(nn.Module):
    def __init__(self, input_size, layer_sizes, dropout_rates, batch_norm=False, activation='relu'):
        """
        Enhanced neural network with additional features
        Args:
            input_size: Number of input features
            layer_sizes: List of integers representing number of neurons in each layer
            dropout_rates: List of dropout rates for each layer
            batch_norm: Whether to use batch normalization
            activation: Activation function to use ('relu', 'leaky_relu', or 'elu')
        """
        super(Net, self).__init__()
        
        # Create list to hold all layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, layer_sizes[0]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(layer_sizes[0]))
        layers.append(self._get_activation(activation))
        if dropout_rates[0] > 0:
            layers.append(nn.Dropout(dropout_rates[0]))
        
        # Hidden layers
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            layers.append(self._get_activation(activation))
            if dropout_rates[i+1] > 0:
                layers.append(nn.Dropout(dropout_rates[i+1]))
            
        # Output layer
        layers.append(nn.Linear(layer_sizes[-1], 1))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
        
    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'elu':
            return nn.ELU()
        else:
            return nn.ReLU()
        
    def forward(self, x):
        return self.network(x)

def process_fold_nn(train_idx, val_idx, X_np, y_np, model_params, get_model=False):
    """Process a single fold of the cross-validation"""
    # print(f"Starting fold in process {os.getpid()}")
    
    device = torch.device("cpu")
    
    # Convert numpy arrays to tensors
    X_fold_train = torch.tensor(X_np[train_idx], dtype=torch.float32).to(device)
    X_fold_val = torch.tensor(X_np[val_idx], dtype=torch.float32).to(device)
    y_fold_train = torch.tensor(y_np[train_idx], dtype=torch.float32).view(-1, 1).to(device)
    y_fold_val = torch.tensor(y_np[val_idx], dtype=torch.float32).view(-1, 1).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_fold_train, y_fold_train)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=model_params['batch_size'], 
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    # Initialize model with enhanced parameters
    model = Net(
        input_size=X_np.shape[1], 
        layer_sizes=model_params['layer_sizes'],
        dropout_rates=model_params['dropout_rates'],
        batch_norm=model_params['batch_norm'],
        activation=model_params['activation']
    ).to(device)
    
    criterion = nn.MSELoss()
    
    # Initialize optimizer based on chosen type
    if model_params['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=model_params['learning_rate'],
            weight_decay=model_params['weight_decay']
        )
    elif model_params['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=model_params['learning_rate'],
            weight_decay=model_params['weight_decay']
        )
    else:  # rmsprop
        optimizer = optim.RMSprop(
            model.parameters(), 
            lr=model_params['learning_rate'],
            weight_decay=model_params['weight_decay']
        )

    # Training loop
    model.train()
    for epoch in range(model_params['n_epochs']):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping if enabled
            if model_params['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), model_params['gradient_clip'])
                
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        val_preds = model(X_fold_val)
        rmse = np.sqrt(mean_squared_error(
            y_fold_val.cpu().numpy(), 
            val_preds.cpu().numpy()
        ))
    
    if get_model == False:
        return rmse, val_preds, val_idx  # Return RMSE, predictions, and indices
    else:
        return model

def nn_objective(trial, X_torch, y_torch, n_splits):
    # Determine number of layers
    n_layers = trial.suggest_int('n_layers', 1, 3)
    
    # Create list of layer sizes with more controlled scaling
    layer_sizes = []
    first_layer_size = trial.suggest_int('n_units_l0', 32, 256)
    layer_sizes.append(first_layer_size)
    
    for i in range(1, n_layers):
        # Each subsequent layer can be between 1/4 and 1 times the size of the previous layer
        prev_size = layer_sizes[-1]
        min_size = max(16, prev_size // 4)
        max_size = prev_size
        layer_sizes.append(trial.suggest_int(f'n_units_l{i}', min_size, max_size))
    
    # Dropout rates for each layer
    dropout_rates = [
        trial.suggest_float(f'dropout_l{i}', 0.0, 0.5)
        for i in range(n_layers)
    ]
    
    model_params = {
        'layer_sizes': layer_sizes,
        'dropout_rates': dropout_rates,
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'rmsprop']),
        'batch_norm': trial.suggest_categorical('batch_norm', [True, False]),
        'activation': trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu']),
        'gradient_clip': trial.suggest_float('gradient_clip', 0.0, 1.0),
        'n_epochs': trial.suggest_int('n_epochs', 50, 300),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    }

    # Convert tensors to numpy arrays before parallel processing
    X_np = X_torch.cpu().numpy()
    y_np = y_torch.cpu().numpy()

    # Initialize K-Fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Use joblib's Parallel
    results = Parallel(n_jobs=NUM_CORES, backend='loky')(
        delayed(process_fold_nn)(
            train_idx, val_idx, X_np, y_np, model_params
        ) for train_idx, val_idx in kf.split(X_np)
    )
    
    # Extract RMSE per fold and predictions
    cv_rmse_scores, predictions_per_fold, indices_per_fold = zip(*results)

    # Reconstruct CV predictions in the correct order
    all_predictions = np.zeros_like(y_np, dtype=float)
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

def train_final_nn(params, X_torch, y_torch, model_name, device, save = True):
    """Train the final neural network model with the best parameters
    Args:
        params: Dictionary containing all model parameters
        X_torch: Input features tensor
        y_torch: Target values tensor
        model_name: Name for saving the model
        device: Device to train on
    """
    # Extract layer sizes
    layer_sizes = [params[f'n_units_l{i}'] for i in range(params['n_layers'])]
    
    # Extract dropout rates
    dropout_rates = [params[f'dropout_l{i}'] for i in range(params['n_layers'])]
    
    # Initialize model with all parameters
    model = Net(
        input_size=X_torch.shape[1],
        layer_sizes=layer_sizes,
        dropout_rates=dropout_rates,
        batch_norm=params['batch_norm'],
        activation=params['activation']
    ).to(device)
    
    # Create dataset and loader
    dataset = TensorDataset(X_torch, y_torch)
    loader = DataLoader(
        dataset, 
        batch_size=params['batch_size'], 
        shuffle=True, 
        drop_last=True,
        num_workers=0
    )
    
    # Initialize optimizer
    if params['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
    elif params['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
    else:  # rmsprop
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
    
    criterion = nn.MSELoss()
    
    # Training loop with progress tracking
    print(f"Starting training for {params['n_epochs']} epochs...")
    for epoch in range(params['n_epochs']):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            num_batches += 1
            
            loss.backward()
            
            # Apply gradient clipping if enabled
            if params['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    params['gradient_clip']
                )
                
            optimizer.step()
        
        # Print progress every 50 epochs
        if (epoch + 1) % 50 == 0:
            avg_loss = total_loss / num_batches
            print(f'Epoch [{epoch+1}/{params["n_epochs"]}], '
                  f'Average Loss: {avg_loss:.4f}')
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_outputs = model(X_torch.to(device))
        final_loss = criterion(final_outputs, y_torch.to(device))
        final_rmse = np.sqrt(mean_squared_error(
            y_torch.cpu().numpy(),
            final_outputs.cpu().numpy()
        ))
    print(f"Training completed. Final RMSE: {final_rmse:.4f}")
    
    # Save final model with all parameters and metadata
    save_dict = {
        'state_dict': model.state_dict(),
        'params': params,
        'input_size': X_torch.shape[1],
        'final_rmse': final_rmse,
        'training_metadata': {
            'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'device': str(device),
            'pytorch_version': torch.__version__
        }
    }
    if save == True:
        torch.save(save_dict, model_name + '.pt')
        print(f"Model saved to {model_name}.pt")
    return model

def load_trained_model(model_path, device):
    """Load a saved model with all its parameters
    Args:
        model_path: Path to the saved model file
        device: Device to load the model on
    Returns:
        model: Loaded model
        params: Model parameters
        metadata: Training metadata
    """
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract parameters
        params = checkpoint['params']
        layer_sizes = [params[f'n_units_l{i}'] for i in range(params['n_layers'])]
        dropout_rates = [params[f'dropout_l{i}'] for i in range(params['n_layers'])]
        
        # Initialize model with all parameters
        model = Net(
            input_size=checkpoint['input_size'],
            layer_sizes=layer_sizes,
            dropout_rates=dropout_rates,
            batch_norm=params['batch_norm'],
            activation=params['activation']
        ).to(device)
        
        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])
        
        # Prepare metadata dictionary
        metadata = {
            'input_size': checkpoint['input_size'],
            'final_rmse': checkpoint.get('final_rmse', None),
            'training_metadata': checkpoint.get('training_metadata', {})
        }
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Architecture: {layer_sizes} units")
        print(f"Training RMSE: {metadata['final_rmse']:.4f}")
        print(f"Training date: {metadata['training_metadata'].get('datetime', 'Not recorded')}")
        
        return model, params, metadata
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def predict_with_model(model, X, device, batch_size=128):
    """Make predictions with loaded model
    Args:
        model: Trained neural network model
        X: Input features (numpy array or torch tensor)
        device: Device to run predictions on
        batch_size: Batch size for predictions
    Returns:
        Predictions as numpy array
    """
    # Convert input to tensor if needed
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    
    # Create dataloader for batched predictions
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    # Make predictions
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for (batch_X,) in loader:
            batch_X = batch_X.to(device)
            batch_preds = model(batch_X)
            predictions.append(batch_preds.cpu().numpy())
    
    # Combine all predictions
    return np.vstack(predictions)

