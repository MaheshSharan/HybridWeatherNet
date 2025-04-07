# Weather Forecast Bias Correction API Documentation

## Overview
The Weather Forecast Bias Correction system provides both a Python API and a Streamlit web interface for correcting biases in weather forecasts using our hybrid deep learning model. This documentation covers the Python API for programmatic access to the bias correction functionality.

## Installation
```bash
pip install -e .
```

## Python API Usage

### Basic Usage
```python
from weather_bias_correction.models import DeepBiasCorrectionModel
from weather_bias_correction.data import DataLoader

# Load the model
model = DeepBiasCorrectionModel.load_from_checkpoint('path/to/checkpoint.pth')

# Prepare data
data_loader = DataLoader(
    ncep_file='path/to/ncep_data.nc',
    gsod_file='path/to/gsod_data.csv'
)

# Make predictions
predictions, uncertainties = model.predict_with_uncertainty(data_loader.test_data)
```

### Model Class
```python
class DeepBiasCorrectionModel:
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 256,
        output_dim: int = 1,
        num_layers: int = 3,
        dropout_rate: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        """
        Initialize the bias correction model.

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (usually 1 for temperature bias)
            num_layers: Number of layers in LSTM and GNN
            dropout_rate: Dropout rate for uncertainty estimation
            learning_rate: Learning rate for optimization
            weight_decay: L2 regularization factor
        """
        pass

    def predict_with_uncertainty(
        self,
        data: torch.Tensor,
        num_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimation using Monte Carlo dropout.

        Args:
            data: Input data tensor
            num_samples: Number of Monte Carlo samples

        Returns:
            Tuple of (predictions, uncertainties)
        """
        pass
```

### Data Loading
```python
class DataLoader:
    def __init__(
        self,
        ncep_file: str,
        gsod_file: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Initialize data loader.

        Args:
            ncep_file: Path to NCEP NetCDF file
            gsod_file: Path to GSOD CSV file
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
        """
        pass

    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess data.

        Returns:
            Tuple of (features, targets)
        """
        pass
```

## Streamlit Interface

The system also provides a Streamlit web interface for interactive use. To launch it:

```bash
streamlit run src/app/app.py
```

The web interface provides:
1. Data upload functionality
2. Real-time bias correction
3. Visualization of results
4. Uncertainty estimation
5. Performance metrics

## Error Handling

The API includes comprehensive error handling:

```python
try:
    predictions, uncertainties = model.predict_with_uncertainty(data)
except ValueError as e:
    print(f"Invalid input data: {e}")
except RuntimeError as e:
    print(f"Model prediction error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

1. **Data Preprocessing**
   - Normalize input features
   - Handle missing values
   - Check data ranges

2. **Model Usage**
   - Use GPU when available
   - Batch large predictions
   - Save model checkpoints

3. **Uncertainty Handling**
   - Always use uncertainty estimates
   - Consider confidence intervals
   - Monitor outliers

## Examples

### Complete Example
```python
import torch
from weather_bias_correction.models import DeepBiasCorrectionModel
from weather_bias_correction.data import DataLoader
from weather_bias_correction.visualization import plot_predictions

# Load data
data_loader = DataLoader(
    ncep_file='data/ncep_2023.nc',
    gsod_file='data/gsod_2023.csv',
    start_date='2023-01-01',
    end_date='2023-12-31'
)

# Initialize model
model = DeepBiasCorrectionModel(
    input_dim=5,
    hidden_dim=256,
    num_layers=3
)

# Load trained weights
model.load_state_dict(torch.load('models/best_model.pth'))

# Make predictions
predictions, uncertainties = model.predict_with_uncertainty(
    data_loader.test_data,
    num_samples=100
)

# Visualize results
plot_predictions(
    actual=data_loader.test_targets,
    predicted=predictions,
    uncertainties=uncertainties
)
```

## Support

For issues and feature requests, please use the [GitHub Issues](https://github.com/MaheshSharan/weather-bias-correction-dl/issues) page.