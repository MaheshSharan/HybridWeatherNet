import pytest
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Set up test data directory
@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    data_dir = project_root / "tests" / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir

# Set up test model directory
@pytest.fixture(scope="session")
def test_model_dir():
    """Create a temporary directory for test models."""
    model_dir = project_root / "tests" / "test_models"
    model_dir.mkdir(exist_ok=True)
    return model_dir

# Set up test cache directory
@pytest.fixture(scope="session")
def test_cache_dir():
    """Create a temporary directory for test cache."""
    cache_dir = project_root / "tests" / "test_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir

# Set up test environment variables
@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    os.environ["TESTING"] = "true"
    os.environ["MODEL_PATH"] = str(project_root / "tests" / "test_models")
    os.environ["CACHE_DIR"] = str(project_root / "tests" / "test_cache")
    os.environ["DATA_DIR"] = str(project_root / "tests" / "test_data")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    yield
    
    # Clean up environment variables
    del os.environ["TESTING"]
    del os.environ["MODEL_PATH"]
    del os.environ["CACHE_DIR"]
    del os.environ["DATA_DIR"]

# Set up test logging
@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Set up test logging configuration."""
    import logging
    
    # Configure logging for tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(project_root / "tests" / "test.log")
        ]
    )
    
    # Create logger
    logger = logging.getLogger("test")
    logger.setLevel(logging.INFO)
    
    return logger

# Set up test device
@pytest.fixture(scope="session")
def device():
    """Get the device to run tests on."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up test batch size
@pytest.fixture(scope="session")
def batch_size():
    """Get the batch size for tests."""
    return 32

# Set up test sequence length
@pytest.fixture(scope="session")
def seq_length():
    """Get the sequence length for tests."""
    return 24

# Set up test input dimension
@pytest.fixture(scope="session")
def input_dim():
    """Get the input dimension for tests."""
    return 5

# Set up test hidden dimension
@pytest.fixture(scope="session")
def hidden_dim():
    """Get the hidden dimension for tests."""
    return 64

# Set up test output dimension
@pytest.fixture(scope="session")
def output_dim():
    """Get the output dimension for tests."""
    return 1 