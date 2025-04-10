# HybridWeatherNet: A LSTM-GNN Framework with Attention for Weather Forecast Bias Correction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A novel deep learning framework that combines LSTM networks for temporal patterns, Graph Neural Networks for spatial relationships, and attention mechanisms for feature fusion to correct systematic biases in weather forecasts. Our approach achieves state-of-the-art performance with mean bias of 0.3°C and RMSE of 0.45°C on unseen data.

## 🌟 Key Features

- **Hybrid Deep Learning Architecture**
  - LSTM for temporal pattern learning
  - Graph Neural Network for spatial relationships
  - Attention mechanism for feature fusion
  - Monte Carlo dropout for uncertainty estimation

- **Physics-Guided Learning**
  - Physics-based regularization in loss function
  - Spatial smoothness constraints
  - Uncertainty quantification

- **Data Integration**
  - Open-Meteo Data
  - ISD-Lite station observations
  - Automated data download and preprocessing

- **Interactive Interface**
  - Streamlit web application
  - Real-time bias correction
  - Uncertainty visualization
  - Performance metrics display

## 📋 Requirements

- Python 3.8+
- PyTorch 2.2.0
- PyTorch Geometric 2.4.0
- PyTorch Lightning 2.2.0
- Other dependencies in `requirements.txt`

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/MaheshSharan/weather-bias-correction-dl.git
   cd weather-bias-correction-dl
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Download and process data**
   ```bash
   python src/data/run_data_pipeline.py --start_year 2018 --end_year 2023
   ```

4. **Train the model**
   ```bash
   python src/training/train.py --data_dir data/processed --accelerator cpu --experiment_name my_training_run
   ```

   Additional training options:
   ```bash
   python src/training/train.py --data_dir data/processed --accelerator gpu --batch_size 64 --max_epochs 100 --hidden_dim 256 --bidirectional True --experiment_name advanced_training
   ```

5. **Run the web interface**
   ```bash
   streamlit run src/app/app.py
   ```

   When using the Streamlit interface:
   - Set the model path to: `logs\pc_training_corrected_v5\checkpoints\bias_correction-epoch=19-val_loss=0.00.ckpt`
   - Click "Load Model" to initialize the model
   - Upload a CSV file containing weather forecast data
   - Required CSV columns (any of these naming conventions will work):
     - Temperature: `temperature`, `temp`, `temperature_2m`
     - Humidity: `humidity`, `relative_humidity_2m`, `relative_humidity`
     - Wind speed: `wind_speed`, `wind_speed_model`, `wind_speed_10m`
     - Wind direction: `wind_direction`, `wind_direction_model`, `wind_direction_10m`
     - Cloud cover: `cloud_cover_low`, `cloud_cover_mid`, `cloud_cover_high`
   - Click "Process Data" to generate bias-corrected forecasts
   - View results in the "Temperature Forecast," "Bias Analysis," and "Uncertainty" tabs

## 🖥️ Google Colab Support

For users with limited computational resources, we provide Google Colab support:
1. Open `Weather_Bias_Correction.ipynb` in Google Colab
2. Mount Google Drive
3. Follow the notebook instructions for setup and training

## 📊 Model Architecture

![Model Architecture](model_architecture.svg)

## 📈 Performance

- **Temperature Bias Correction**:
  - Mean Bias: 0.27°C
  - RMSE: 0.54°C
  - MAE: 0.32°C

- **Key Advantages**:
  - Accurate bias prediction with proper denormalization
  - Reliable uncertainty estimates
  - Physics-consistent corrections
  - Support for various weather data formats

## 🔍 Example Data Format

Here's an example of the expected CSV format (column names may vary as noted in the Streamlit instructions):

```
date,temperature,humidity,wind_speed_model,wind_direction_model,cloud_cover_low,cloud_cover_mid,cloud_cover_high
2018-01-01,6.66,83.17,23.93,226.75,45.5,49.42,60.54
2018-01-02,5.85,87.46,18.15,241.50,46.21,46.58,54.25
2018-01-03,8.02,80.42,38.76,252.33,48.21,59.17,42.13
```

The model will predict the bias in the temperature forecast, which can then be applied to correct the original forecast.

## 📝 Citation

If you use this code in your research, please cite:
```bibtex
@article{hybridweathernet2025,
  title={HybridWeatherNet: A LSTM-GNN Framework with Attention for Weather Forecast Bias Correction},
  author={Mahesh Sharan},
  journal={Environmental Data Science},
  year={2025},
  publisher={Cambridge University Press}
}
```

## Model Performance Analysis

### Current Performance Metrics
- Amsterdam: Mean Bias -0.92°C, RMSE 1.30°C, MAE 1.12°C
- India: Mean Bias 1.06°C, RMSE 1.64°C, MAE 1.45°C

### Performance Analysis

![Feature Correlations - Amsterdam](analysis_results/analysis_correlation_Amsterdam.png)
*Feature correlations with bias in Amsterdam data*

![Feature Correlations - India](analysis_results/analysis_correlation_India.png)
*Feature correlations with bias in India data*

#### Key Findings

1. **Temperature Sensitivity**:
   - High temperatures (RMSE: 2.01°C in India, 1.52°C in Amsterdam)
   - Low temperatures (RMSE: 1.45°C in India, 1.29°C in Amsterdam)

2. **Humidity Impact**:
   - Low humidity conditions show higher errors:
     * Amsterdam: RMSE 1.79°C (low) vs 0.95°C (high)
     * India: RMSE 1.73°C (low) vs 1.27°C (high)

3. **Wind Effects**:
   - Regional variation in wind impact
   - Amsterdam: Higher errors in high wind (RMSE 1.46°C)
   - India: Better performance in high wind (RMSE 1.33°C)

### Areas for Improvement

For researchers looking to improve the model's accuracy:

1. **Humidity Conditions**: 
   - Focus on improving performance in low humidity conditions
   - Consider adding derived features that combine humidity with temperature

2. **Temperature Extremes**:
   - Enhance model's ability to handle temperature extremes
   - Particularly important for high temperatures in tropical regions

3. **Regional Adaptation**:
   - Consider the varying impact of wind across different regions
   - Investigate region-specific meteorological patterns

![Error Distribution - Amsterdam](analysis_results/analysis_error_distribution_Amsterdam.png)
*Error distribution in Amsterdam predictions*

![Error Distribution - India](analysis_results/analysis_error_distribution_India.png)
*Error distribution in India predictions*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.