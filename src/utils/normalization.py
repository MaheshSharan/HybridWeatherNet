"""
Utility functions for saving and loading normalization parameters.
"""
import os
import json
import numpy as np
from typing import Dict, Any, Tuple

def save_normalization_params(
    target_mean: float,
    target_std: float,
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    save_path: str
) -> None:
    """
    Save normalization parameters to a JSON file.
    
    Args:
        target_mean (float): Mean of the target variable
        target_std (float): Standard deviation of the target variable
        feature_means (np.ndarray): Means of the features
        feature_stds (np.ndarray): Standard deviations of the features
        save_path (str): Path to save the parameters
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    params = {
        'target_mean': float(target_mean),
        'target_std': float(target_std),
        'feature_means': feature_means.tolist() if isinstance(feature_means, np.ndarray) else feature_means,
        'feature_stds': feature_stds.tolist() if isinstance(feature_stds, np.ndarray) else feature_stds
    }
    
    # Save to file
    with open(save_path, 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"Normalization parameters saved to {save_path}")

def load_normalization_params(load_path: str) -> Dict[str, Any]:
    """
    Load normalization parameters from a JSON file.
    
    Args:
        load_path (str): Path to load the parameters from
        
    Returns:
        Dict[str, Any]: Dictionary containing the normalization parameters
    """
    try:
        with open(load_path, 'r') as f:
            params = json.load(f)
        
        # Convert lists back to numpy arrays
        params['feature_means'] = np.array(params['feature_means'])
        params['feature_stds'] = np.array(params['feature_stds'])
        
        return params
    except FileNotFoundError:
        raise FileNotFoundError(f"Normalization parameters file not found at {load_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in normalization parameters file at {load_path}")

def denormalize_target(normalized_target: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    """
    Denormalize target values.
    
    Args:
        normalized_target (np.ndarray): Normalized target values
        target_mean (float): Mean of the target variable
        target_std (float): Standard deviation of the target variable
        
    Returns:
        np.ndarray: Denormalized target values
    """
    return normalized_target * target_std + target_mean

def normalize_features(features: np.ndarray, feature_means: np.ndarray, feature_stds: np.ndarray) -> np.ndarray:
    """
    Normalize features.
    
    Args:
        features (np.ndarray): Raw feature values
        feature_means (np.ndarray): Means of the features
        feature_stds (np.ndarray): Standard deviations of the features
        
    Returns:
        np.ndarray: Normalized features
    """
    return (features - feature_means) / (feature_stds + 1e-8)
