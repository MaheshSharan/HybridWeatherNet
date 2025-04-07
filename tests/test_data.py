import os
import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any

from src.data import DataAligner

@pytest.fixture
def sample_data() -> Dict[str, np.ndarray]:
    """Create sample data for testing."""
    # Create sample timestamps
    timestamps = pd.date_range(
        start='2023-01-01',
        end='2023-01-02',
        freq='H'
    )
    
    # Create sample temperature data
    n_samples = len(timestamps)
    temperature = np.random.normal(20, 5, n_samples)
    
    # Create sample features
    features = np.column_stack([
        temperature,  # Original temperature
        temperature + np.random.normal(0, 1, n_samples),  # Forecast temperature
        np.random.normal(0, 1, n_samples),  # Humidity
        np.random.normal(0, 1, n_samples),  # Wind speed
        np.random.normal(0, 1, n_samples)   # Pressure
    ])
    
    return {
        'timestamps': timestamps,
        'features': features
    }

@pytest.fixture
def data_aligner(tmp_path) -> DataAligner:
    """Create a DataAligner instance for testing."""
    # Create temporary data directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    return DataAligner(
        data_dir=str(data_dir),
        batch_size=32,
        num_workers=0
    )

def test_data_loading(data_aligner: DataAligner, sample_data: Dict[str, np.ndarray]):
    """Test data loading functionality."""
    # Save sample data
    df = pd.DataFrame(
        sample_data['features'],
        columns=['temperature', 'forecast', 'humidity', 'wind_speed', 'pressure'],
        index=sample_data['timestamps']
    )
    df.to_csv(os.path.join(data_aligner.data_dir, 'test_data.csv'))
    
    # Test loading data
    loaded_data = data_aligner.load_data('test_data.csv')
    assert isinstance(loaded_data, pd.DataFrame)
    assert len(loaded_data) == len(sample_data['timestamps'])
    assert all(col in loaded_data.columns for col in [
        'temperature', 'forecast', 'humidity', 'wind_speed', 'pressure'
    ])

def test_data_preprocessing(data_aligner: DataAligner, sample_data: Dict[str, np.ndarray]):
    """Test data preprocessing functionality."""
    # Save sample data
    df = pd.DataFrame(
        sample_data['features'],
        columns=['temperature', 'forecast', 'humidity', 'wind_speed', 'pressure'],
        index=sample_data['timestamps']
    )
    df.to_csv(os.path.join(data_aligner.data_dir, 'test_data.csv'))
    
    # Test preprocessing
    processed_data = data_aligner.preprocess_data('test_data.csv')
    assert isinstance(processed_data, pd.DataFrame)
    assert len(processed_data) == len(sample_data['timestamps'])
    assert all(col in processed_data.columns for col in [
        'temperature', 'forecast', 'humidity', 'wind_speed', 'pressure'
    ])
    
    # Check for missing values
    assert not processed_data.isnull().any().any()
    
    # Check for outliers
    assert not (processed_data > 1e6).any().any()
    assert not (processed_data < -1e6).any().any()

def test_data_alignment(data_aligner: DataAligner, sample_data: Dict[str, np.ndarray]):
    """Test data alignment functionality."""
    # Save sample data
    df = pd.DataFrame(
        sample_data['features'],
        columns=['temperature', 'forecast', 'humidity', 'wind_speed', 'pressure'],
        index=sample_data['timestamps']
    )
    df.to_csv(os.path.join(data_aligner.data_dir, 'test_data.csv'))
    
    # Test alignment
    aligned_data = data_aligner.align_data('test_data.csv')
    assert isinstance(aligned_data, pd.DataFrame)
    assert len(aligned_data) == len(sample_data['timestamps'])
    assert all(col in aligned_data.columns for col in [
        'temperature', 'forecast', 'humidity', 'wind_speed', 'pressure'
    ])
    
    # Check temporal alignment
    assert aligned_data.index.is_monotonic_increasing

def test_bias_calculation(data_aligner: DataAligner, sample_data: Dict[str, np.ndarray]):
    """Test bias calculation functionality."""
    # Save sample data
    df = pd.DataFrame(
        sample_data['features'],
        columns=['temperature', 'forecast', 'humidity', 'wind_speed', 'pressure'],
        index=sample_data['timestamps']
    )
    df.to_csv(os.path.join(data_aligner.data_dir, 'test_data.csv'))
    
    # Test bias calculation
    bias = data_aligner.calculate_bias('test_data.csv')
    assert isinstance(bias, float)
    assert not np.isnan(bias)
    assert not np.isinf(bias)

def test_data_validation(data_aligner: DataAligner, sample_data: Dict[str, np.ndarray]):
    """Test data validation functionality."""
    # Save sample data
    df = pd.DataFrame(
        sample_data['features'],
        columns=['temperature', 'forecast', 'humidity', 'wind_speed', 'pressure'],
        index=sample_data['timestamps']
    )
    df.to_csv(os.path.join(data_aligner.data_dir, 'test_data.csv'))
    
    # Test validation
    validation_result = data_aligner.validate_data('test_data.csv')
    assert isinstance(validation_result, bool)
    assert validation_result 