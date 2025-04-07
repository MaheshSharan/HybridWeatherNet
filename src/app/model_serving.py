import os
import torch
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
from functools import lru_cache

from ..models import DeepBiasCorrectionModel
from ..training.config import get_config

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
    with caching for better performance.
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
            cache_size (int): Size of the prediction cache
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
            # Get configuration
            config = get_config()
            
            # Create model with same configuration as training
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
            
            # Load checkpoint
            model.load_from_checkpoint(self.model_path)
            model.to(self.device)
            model.eval()
            
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _optimize_model(self) -> DeepBiasCorrectionModel:
        """
        Optimize the model for inference.
        
        Returns:
            DeepBiasCorrectionModel: Optimized model
        """
        try:
            # Convert model to TorchScript
            if self.device == 'cuda':
                # Use CUDA optimization
                self.model = torch.jit.script(self.model)
                self.model = self.model.cuda()
            else:
                # Use CPU optimization
                self.model = torch.jit.script(self.model)
            
            return self.model
        except Exception as e:
            logger.error(f"Error optimizing model: {str(e)}")
            raise
    
    @lru_cache(maxsize=100)
    def predict(
        self,
        input_data: np.ndarray,
        return_uncertainty: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions with the model.
        
        Args:
            input_data (np.ndarray): Input data for prediction
            return_uncertainty (bool): Whether to return uncertainty estimates
            
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: 
                - Predictions
                - Uncertainty estimates (if requested)
        """
        try:
            # Convert input to tensor
            input_tensor = torch.from_numpy(input_data).float().to(self.device)
            
            # Add batch dimension if needed
            if len(input_tensor.shape) == 2:
                input_tensor = input_tensor.unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                if return_uncertainty:
                    predictions, uncertainty = self.model(input_tensor, return_uncertainty=True)
                    predictions = predictions.cpu().numpy()
                    uncertainty = uncertainty.cpu().numpy()
                    return predictions, uncertainty
                else:
                    predictions = self.model(input_tensor)
                    predictions = predictions.cpu().numpy()
                    return predictions, None
        
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def clear_cache(self) -> None:
        """Clear the prediction cache."""
        self.predict.cache_clear()
        logger.info("Prediction cache cleared")
    
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