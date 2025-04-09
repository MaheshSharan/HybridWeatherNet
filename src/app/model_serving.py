import os
import torch
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
from functools import lru_cache

from src.models import DeepBiasCorrectionModel
from src.training.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelServer:
    """
    Model server for optimized inference.
    
    This class handles model loading, optimization, and inference
    with batch processing for large datasets.
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        cache_size: int = 100
    ):
        """
        Initialize the model server.
        
        Args:
            model_path (str): Path to the model checkpoint
            device (Optional[str]): Device to run the model on
            cache_size (int): Size of the prediction cache (unused now)
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_size = cache_size
        
        # Load and optimize model
        self.model = self._load_model()
        self.model = self._optimize_model()
        
        logger.info(f"Model server initialized on device: {self.device}")
    
    def _load_model(self) -> DeepBiasCorrectionModel:
        """
        Load the model from checkpoint.
        
        Returns:
            DeepBiasCorrectionModel: Loaded model
        """
        try:
            logger.info(f"Attempting to load model from {self.model_path}")
            model = DeepBiasCorrectionModel.load_from_checkpoint(self.model_path)
            model.to(self.device)
            model.eval()
            logger.info(f"Model loaded successfully from checkpoint")
            return model
        except Exception as e:
            logger.error(f"Error loading model directly: {str(e)}")
            logger.info("Attempting to load using config...")
            try:
                config = get_config()
                model = DeepBiasCorrectionModel(
                    input_dim=config['model']['input_dim'],
                    hidden_dim=config['model']['hidden_dim'],
                    output_dim=config['model']['output_dim'],
                    num_layers=config['model']['num_layers'],
                    dropout_rate=config['model']['dropout_rate'],
                    learning_rate=config['training']['learning_rate'],
                    weight_decay=config['training']['weight_decay'],
                    physics_weight=config['training']['physics_weight'],
                    bidirectional=config['model']['bidirectional']
                )
                checkpoint = torch.load(self.model_path, map_location=self.device)
                model.load_state_dict(checkpoint['state_dict'])
                model.to(self.device)
                model.eval()
                logger.info("Model loaded successfully using config")
                return model
            except Exception as nested_e:
                logger.error(f"Error loading with config: {str(nested_e)}")
                raise Exception(f"Failed to load model: {str(e)}. Config loading failed: {str(nested_e)}")
    
    def _optimize_model(self) -> DeepBiasCorrectionModel:
        """
        Optimize the model for inference.
        
        Returns:
            DeepBiasCorrectionModel: Optimized model
        """
        try:
            self.model.eval()
            self.model.to(self.device)
            logger.info(f"Model optimized for inference on {self.device}")
            return self.model
        except Exception as e:
            logger.error(f"Error optimizing model: {str(e)}")
            raise
    
    def predict(
        self,
        input_data: np.ndarray,
        return_uncertainty: bool = False,
        batch_size: int = 32
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions with the model in batches.
        
        Args:
            input_data (np.ndarray): Input data (n_samples, n_features)
            return_uncertainty (bool): Whether to return uncertainty estimates
            batch_size (int): Number of samples per batch
            
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: Predictions, Uncertainty
        """
        try:
            logger.info(f"Predicting on input shape: {input_data.shape}")
            num_samples = input_data.shape[0]
            predictions_list = []
            uncertainty_list = [] if return_uncertainty else None
            
            for i in range(0, num_samples, batch_size):
                batch = input_data[i:i + batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1}: shape {batch.shape}")
                input_tensor = torch.from_numpy(batch).float().to(self.device)
                if len(input_tensor.shape) == 2:
                    input_tensor = input_tensor.unsqueeze(1)  # [batch, 1, features]
                
                with torch.no_grad():
                    if return_uncertainty:
                        # Expect 3 outputs: mean, attention_weights, variance
                        preds, _, uncert = self.model(input_tensor, return_uncertainty=True)
                        predictions_list.append(preds.cpu().numpy())
                        uncertainty_list.append(uncert.cpu().numpy())
                    else:
                        # Expect 3 outputs: output, attention_weights, dummy_uncertainty
                        preds, _, _ = self.model(input_tensor)
                        predictions_list.append(preds.cpu().numpy())
            
            predictions = np.concatenate(predictions_list, axis=0)
            uncertainty = np.concatenate(uncertainty_list, axis=0) if return_uncertainty else None
            logger.info(f"Prediction complete: output shape {predictions.shape}")
            return predictions, uncertainty
        
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def clear_cache(self) -> None:
        """Clear the prediction cache (no-op since caching is disabled)."""
        logger.info("Prediction cache cleared (no-op since caching is disabled)")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            'device': self.device,
            'model_path': self.model_path,
            'cache_size': self.cache_size,
            'model_type': type(self.model).__name__,
            'is_optimized': isinstance(self.model, torch.jit.ScriptModule)
        }