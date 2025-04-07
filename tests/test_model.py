import pytest
import torch
import numpy as np
from typing import Dict, Any, Tuple

from src.models import DeepBiasCorrectionModel, LSTMModule

@pytest.fixture
def model_config() -> Dict[str, Any]:
    """Create model configuration for testing."""
    return {
        'input_dim': 5,
        'hidden_dim': 64,
        'output_dim': 1,
        'num_layers': 2,
        'dropout_rate': 0.1,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'physics_weight': 0.1,
        'bidirectional': True
    }

@pytest.fixture
def sample_input() -> Tuple[torch.Tensor, torch.Tensor]:
    """Create sample input data for testing."""
    # Create sample input tensor
    batch_size = 32
    seq_length = 24
    input_dim = 5
    
    x = torch.randn(batch_size, seq_length, input_dim)
    y = torch.randn(batch_size, seq_length, 1)
    
    return x, y

@pytest.fixture
def model(model_config: Dict[str, Any]) -> DeepBiasCorrectionModel:
    """Create a model instance for testing."""
    return DeepBiasCorrectionModel(**model_config)

@pytest.fixture
def lstm_module(model_config: Dict[str, Any]) -> LSTMModule:
    """Create an LSTM module instance for testing."""
    return LSTMModule(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        dropout_rate=model_config['dropout_rate'],
        bidirectional=model_config['bidirectional']
    )

def test_model_initialization(model: DeepBiasCorrectionModel, model_config: Dict[str, Any]):
    """Test model initialization."""
    assert isinstance(model, DeepBiasCorrectionModel)
    assert model.input_dim == model_config['input_dim']
    assert model.hidden_dim == model_config['hidden_dim']
    assert model.output_dim == model_config['output_dim']
    assert model.num_layers == model_config['num_layers']
    assert model.dropout_rate == model_config['dropout_rate']
    assert model.bidirectional == model_config['bidirectional']

def test_lstm_module_initialization(lstm_module: LSTMModule, model_config: Dict[str, Any]):
    """Test LSTM module initialization."""
    assert isinstance(lstm_module, LSTMModule)
    assert lstm_module.input_dim == model_config['input_dim']
    assert lstm_module.hidden_dim == model_config['hidden_dim']
    assert lstm_module.num_layers == model_config['num_layers']
    assert lstm_module.dropout_rate == model_config['dropout_rate']
    assert lstm_module.bidirectional == model_config['bidirectional']

def test_model_forward(model: DeepBiasCorrectionModel, sample_input: Tuple[torch.Tensor, torch.Tensor]):
    """Test model forward pass."""
    x, _ = sample_input
    
    # Forward pass
    output, hidden = model(x)
    
    # Check output shape
    assert output.shape == (x.shape[0], x.shape[1], model.output_dim)
    assert isinstance(hidden, tuple)
    assert len(hidden) == 2

def test_lstm_module_forward(lstm_module: LSTMModule, sample_input: Tuple[torch.Tensor, torch.Tensor]):
    """Test LSTM module forward pass."""
    x, _ = sample_input
    
    # Forward pass
    output, hidden = lstm_module(x)
    
    # Check output shape
    expected_output_dim = lstm_module.hidden_dim * 2 if lstm_module.bidirectional else lstm_module.hidden_dim
    assert output.shape == (x.shape[0], x.shape[1], expected_output_dim)
    assert isinstance(hidden, tuple)
    assert len(hidden) == 2

def test_model_training_step(model: DeepBiasCorrectionModel, sample_input: Tuple[torch.Tensor, torch.Tensor]):
    """Test model training step."""
    x, y = sample_input
    
    # Training step
    loss = model.training_step({'input': x, 'target': y}, 0)
    
    # Check loss
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

def test_model_validation_step(model: DeepBiasCorrectionModel, sample_input: Tuple[torch.Tensor, torch.Tensor]):
    """Test model validation step."""
    x, y = sample_input
    
    # Validation step
    metrics = model.validation_step({'input': x, 'target': y}, 0)
    
    # Check metrics
    assert isinstance(metrics, dict)
    assert all(isinstance(v, torch.Tensor) for v in metrics.values())
    assert all(not torch.isnan(v) for v in metrics.values())
    assert all(not torch.isinf(v) for v in metrics.values())

def test_model_test_step(model: DeepBiasCorrectionModel, sample_input: Tuple[torch.Tensor, torch.Tensor]):
    """Test model test step."""
    x, y = sample_input
    
    # Test step
    metrics = model.test_step({'input': x, 'target': y}, 0)
    
    # Check metrics
    assert isinstance(metrics, dict)
    assert all(isinstance(v, torch.Tensor) for v in metrics.values())
    assert all(not torch.isnan(v) for v in metrics.values())
    assert all(not torch.isinf(v) for v in metrics.values())

def test_physics_loss(model: DeepBiasCorrectionModel, sample_input: Tuple[torch.Tensor, torch.Tensor]):
    """Test physics-guided loss calculation."""
    x, y = sample_input
    
    # Forward pass
    y_pred, _ = model(x)
    
    # Calculate physics loss
    physics_loss = model.calculate_physics_loss(y_pred, y)
    
    # Check physics loss
    assert isinstance(physics_loss, torch.Tensor)
    assert not torch.isnan(physics_loss)
    assert not torch.isinf(physics_loss)

def test_calibration_metrics(model: DeepBiasCorrectionModel, sample_input: Tuple[torch.Tensor, torch.Tensor]):
    """Test calibration metrics calculation."""
    x, y = sample_input
    
    # Forward pass
    y_pred, _ = model(x)
    
    # Calculate calibration metrics
    metrics = model.calculate_calibration_metrics(y_pred, y)
    
    # Check metrics
    assert isinstance(metrics, dict)
    assert all(isinstance(v, torch.Tensor) for v in metrics.values())
    assert all(not torch.isnan(v) for v in metrics.values())
    assert all(not torch.isinf(v) for v in metrics.values()) 