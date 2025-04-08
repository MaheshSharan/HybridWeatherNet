# Weather Forecast Bias Correction using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning approach for correcting systematic biases in weather forecasts using a hybrid architecture that combines LSTM networks for temporal patterns, Graph Neural Networks for spatial relationships, and attention mechanisms for feature fusion.

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
   python src/training/train.py --data_dir data/processed
   ```

5. **Run the web interface**
   ```bash
   streamlit run src/app/app.py
   ```

## 🖥️ Google Colab Support

For users with limited computational resources, we provide Google Colab support:
1. Open `Weather_Bias_Correction.ipynb` in Google Colab
2. Mount Google Drive
3. Follow the notebook instructions for setup and training

## 📊 Model Architecture

```
Input Data → LSTM Module → Graph Neural Network → Attention Fusion → Bias Prediction
                                                                 ↓
                                                    Uncertainty Estimation
```

## 📈 Performance

- Significant reduction in systematic temperature forecast errors
- Improved statistical metrics (MAE/RMSE)
- Reliable uncertainty estimates
- Physics-consistent corrections

## 🗂️ Project Structure

```
weather_bias_correction/
├── data/               # Data storage
├── docs/              # Documentation
├── notebooks/         # Jupyter notebooks
├── src/              # Source code
│   ├── app/          # Streamlit application
│   ├── data/         # Data processing
│   ├── models/       # Model architecture
│   └── training/     # Training scripts
├── tests/            # Unit tests
├── requirements.txt  # Dependencies
└── setup.py         # Package setup
```

## 📝 Citation

If you use this code in your research, please cite:
```bibtex
@article{weather_bias_correction,
  title={Bias Correction in Numerical Weather Prediction Temperature Forecasting: A Deep Learning Approach},
  author={Mahesh Sharan},
  year={2025}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.