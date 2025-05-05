import numpy as np

class ConformalPredictor:
    """Base class for conformal prediction"""
    def __init__(self, confidence=0.95):
        self.confidence = confidence
        self.error_bound = None
        
    def get_coverage(self, y_true, lower, upper):
        lower = np.array(lower).flatten()  # Convert to 1D
        upper = np.array(upper).flatten()
        y_true = np.array(y_true).flatten()

        covered = np.sum((y_true >= lower) & (y_true <= upper))  # Number of covered values
        total = len(y_true)  # Total count
        return covered / float(total)  # Normalize to fraction

class FixedConformalPredictor(ConformalPredictor):
    """Fixed error bound conformal predictor"""
    def fit(self, y_true, y_pred):
        abs_errors = np.abs(y_true - y_pred)
        n = len(abs_errors)
        bound_index = int(np.ceil((n + 1) * self.confidence) - 1)
        self.error_bound = np.sort(abs_errors)[bound_index]
        
    def predict_interval(self, predictions):
        if self.error_bound is None:
            raise ValueError("Fit the predictor first")
        lower = predictions - self.error_bound
        upper = predictions + self.error_bound
        return lower, upper

class RelativeConformalPredictor(ConformalPredictor):
    """Relative error bound conformal predictor"""
    def fit(self, y_true, y_pred):
        relative_errors = np.abs(y_true - y_pred) / y_pred
        n = len(relative_errors)
        bound_index = int(np.ceil((n + 1) * self.confidence) - 1)
        self.error_bound = np.sort(relative_errors)[bound_index]
        
    def predict_interval(self, predictions):
        if self.error_bound is None:
            raise ValueError("Fit the predictor first")
        lower = predictions * (1 - self.error_bound)
        upper = predictions * (1 + self.error_bound)
        return lower, upper

class DistanceBasedConformalPredictor(ConformalPredictor):
    """Distance-based conformal predictor that adjusts intervals based on distance in feature space"""
    def __init__(self, confidence=0.95):
        super().__init__(confidence)
        self.lambda_ = None  # Distance scaling factor
        self.distances = None  # Store distances for interval calculation

    def fit(self, y_true, y_pred, distances):
        """Fit conformal predictor using residuals and distances"""
        abs_errors = np.abs(y_true - y_pred)
        n = len(abs_errors)
        bound_index = int(np.ceil((n + 1) * self.confidence) - 1)
        self.error_bound = np.sort(abs_errors)[bound_index]
        self.distances = distances  # Store distances for use in prediction

    def predict_interval(self, y_pred, lambda_, distances):
        """Generate prediction intervals that scale with distance"""
        if self.error_bound is None:
            raise ValueError("Fit the predictor first")
        # Compute distance-based interval width
        interval_width = self.error_bound * (np.ones(len(distances)) + lambda_*np.log(np.ones(len(distances)) + distances))

        lower = y_pred - interval_width
        upper = y_pred + interval_width
        return lower, upper
    
