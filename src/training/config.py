"""
Default configuration for model training.
"""

from typing import Dict, Any

# Data configuration
DATA_CONFIG = {
    'batch_size': 32,              # Number of samples processed at once during training
    'num_workers': 4,              # Number of parallel processes for data loading
    'input_dim': 7,                # Number of input features (e.g., temp, humidity, etc.)
    'sequence_length': 24,         # Number of time steps the model looks at (e.g., 24 hours)
    'prediction_length': 24,       # Number of time steps to predict (e.g., next 24 hours)
}

MODEL_CONFIG = {
    'hidden_dim': 256,             # Size of hidden layers in the model (LSTM/GNN/Attention)
    'num_layers': 3,               # Number of LSTM (or GNN) layers stacked
    'dropout_rate': 0.2,           # Fraction of units dropped for regularization (prevents overfitting)
    'bidirectional': True,         # Whether LSTM processes data both forward and backward in time
    'output_dim': 1,               # Output size (predicting a single bias value)
}

TRAINING_CONFIG = {
    'learning_rate': 1e-3,         # How fast the model learns (step size in gradient descent)
    'weight_decay': 1e-5,          # L2 regularization to prevent overfitting
    'physics_weight': 0.1,         # Weight for the physics-based loss term (enforces physical realism)
    'max_epochs': 100,             # Maximum number of times to go through the training data
    'patience': 10,                # Early stopping: stop if validation doesn't improve for 10 epochs
    'gradient_clip_val': 1.0,      # Maximum value for gradients (prevents exploding gradients)
    'accumulate_grad_batches': 1,  # Number of batches to accumulate gradients over before updating
}

LOGGING_CONFIG = {
    'log_dir': 'logs',             # Directory where training logs are saved
    'experiment_name': 'bias_correction', # Name for this experiment/run
    'log_every_n_steps': 50,       # How often to log training progress (in steps)
}