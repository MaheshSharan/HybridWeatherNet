# Weather Forecast Bias Correction System User Guide

## Introduction
The Weather Forecast Bias Correction System is a deep learning-based solution that combines LSTM networks, Graph Neural Networks, and attention mechanisms to correct systematic biases in weather forecasts. This guide will help you understand how to use the system effectively.

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Data Sources](#data-sources)
4. [Using the Web Interface](#using-the-web-interface)
5. [Model Training](#model-training)
6. [Google Colab Usage](#google-colab-usage)
7. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Git

### Installation Steps
1. Clone the repository:
```bash
git clone https://github.com/MaheshSharan/weather-bias-correction-dl.git
cd weather-bias-correction-dl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

1. **Download and Process Data**
```bash
python src/data/run_data_pipeline.py --start_year 2018 --end_year 2023
```

2. **Train the Model**
```bash
python src/training/train.py --data_dir data/processed
```

3. **Launch Web Interface**
```bash
streamlit run src/app/app.py
```

## Data Sources

### NCEP Data
- Source: NCEP/NCAR Reanalysis dataset
- Variables: Temperature, pressure, wind components, etc.
- Resolution: 0.25° x 0.25°
- Time range: 2018-2023

### GSOD Data
- Source: Global Surface Summary of the Day
- Variables: Temperature observations from weather stations
- Coverage: Global weather stations
- Time range: 2018-2023

## Using the Web Interface

The Streamlit web interface provides:

1. **Data Upload**
   - Upload CSV files with forecast data
   - Supported format: See `example_data.csv` in docs

2. **Bias Correction**
   - Real-time bias correction
   - Uncertainty estimation
   - Performance metrics

3. **Visualization**
   - Time series plots
   - Uncertainty bands
   - Error distribution plots
   - Station-wise performance metrics

## Model Training

### Configuration
Edit `src/training/config.py` to modify:
- Model architecture parameters
- Training hyperparameters
- Data preprocessing settings

### Training Process
1. **Data Preparation**
   ```bash
   python src/data/run_data_pipeline.py --start_year 2018 --end_year 2023
   ```

2. **Model Training**
   ```bash
   python src/training/train.py \
       --data_dir data/processed \
       --batch_size 32 \
       --epochs 100 \
       --learning_rate 1e-3
   ```

3. **Model Evaluation**
   ```bash
   python src/training/evaluate.py --model_path models/best_model.pth
   ```

### Training Monitoring
- TensorBoard integration for monitoring training
- MLflow for experiment tracking
- Early stopping and model checkpointing

## Google Colab Usage

1. **Setup**
   - Open `Weather_Bias_Correction.ipynb` in Google Colab
   - Mount your Google Drive
   - Follow the notebook instructions

2. **Data Management**
   - Data will be stored in your Google Drive
   - Models and checkpoints are automatically saved

3. **Resource Optimization**
   - Use smaller batch sizes if running out of memory
   - Enable GPU acceleration in Colab
   - Save checkpoints frequently

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size
   - Use data sampling
   - Enable gradient checkpointing

2. **Data Loading Issues**
   - Check data directory structure
   - Verify file permissions
   - Ensure correct date ranges

3. **Training Problems**
   - Check GPU availability
   - Verify data preprocessing
   - Monitor learning rate

### Getting Help
- Check the [Issues](https://github.com/MaheshSharan/weather-bias-correction-dl/issues) page
- Submit detailed bug reports
- Join discussions in the repository

## Additional Resources
- Research paper: See `Research paper.md`
- Example notebooks: See `notebooks/` directory
- API documentation: See `docs/api.md`