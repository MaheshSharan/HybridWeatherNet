"""
Default configuration for model training.
"""

from typing import Dict, Any

# Data configuration
DATA_CONFIG = {
    'batch_size': 32,
    'num_workers': 4,
    'input_dim': 7,  # Fixed to match paper
    'sequence_length': 24,
    'prediction_length': 24,
}

MODEL_CONFIG = {
    'hidden_dim': 256,
    'num_layers': 3,
    'dropout_rate': 0.2,
    'bidirectional': True,
    'output_dim': 1,
}

TRAINING_CONFIG = {
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'physics_weight': 0.1,
    'max_epochs': 100,
    'patience': 10,
    'gradient_clip_val': 1.0,
    'accumulate_grad_batches': 1,
}

LOGGING_CONFIG = {
    'log_dir': 'logs',
    'experiment_name': 'bias_correction',
    'log_every_n_steps': 50,
}

def get_config():
    return {
        'data': DATA_CONFIG,
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'logging': LOGGING_CONFIG,
    }
