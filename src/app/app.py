import streamlit as st
import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import torch
import plotly.graph_objects as go

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.app.model_serving import ModelServer
from src.app.visualization import (
    plot_temperature_forecast,
    plot_bias_correction,
    plot_uncertainty,
    display_metrics
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize session state
if 'model_server' not in st.session_state:
    st.session_state.model_server = None

def load_model(model_path: str) -> ModelServer:
    """
    Load and initialize the model server.
    
    Args:
        model_path (str): Path to the model checkpoint
        
    Returns:
        ModelServer: Initialized model server
    """
    try:
        # Try to find normalization parameters in the same directory as the model
        model_dir = os.path.dirname(os.path.abspath(model_path))
        norm_params_path = os.path.join(os.path.dirname(model_dir), "normalization_params.json")
        
        return ModelServer(
            model_path=model_path,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            cache_size=100,
            norm_params_path=norm_params_path if os.path.exists(norm_params_path) else None
        )
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def process_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Process the uploaded data for prediction.
    
    Args:
        data (pd.DataFrame): Uploaded data
        
    Returns:
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]: 
            - Processed features
            - Timestamps
            - Actual temperatures (if available, else None)
    """
    try:
        # Debug logging
        logger.info(f"Input data columns: {data.columns.tolist()}")
        logger.info(f"Input data sample:\n{data.head()}")
        
        # Map of possible column names to standardized names
        column_mapping = {
            # Temperature (forecast)
            'temperature': 'temperature',
            'temp': 'temperature',
            'temperature_2m': 'temperature',
            
            # Temperature (actual)
            'temp_avg': 'actual_temperature',
            'temperature_obs': 'actual_temperature',
            
            # Humidity
            'humidity': 'humidity',
            'relative_humidity_2m': 'humidity',
            'relative_humidity': 'humidity',
            'rh': 'humidity',
            
            # Wind speed
            'wind_speed': 'wind_speed',
            'wind_speed_model': 'wind_speed',
            'wind_speed_10m': 'wind_speed',
            'windspeed_10m': 'wind_speed',
            
            # Wind direction
            'wind_direction': 'wind_direction',
            'wind_direction_model': 'wind_direction',
            'wind_direction_10m': 'wind_direction',
            'winddirection_10m': 'wind_direction',
            'wind_dir': 'wind_direction',
            
            # Cloud cover
            'cloud_cover_low': 'cloud_cover_low',
            'cloudcover_low': 'cloud_cover_low',
            'low_cloud_cover': 'cloud_cover_low',
            
            'cloud_cover_mid': 'cloud_cover_mid',
            'cloudcover_mid': 'cloud_cover_mid',
            'mid_cloud_cover': 'cloud_cover_mid',
            
            'cloud_cover_high': 'cloud_cover_high',
            'cloudcover_high': 'cloud_cover_high',
            'high_cloud_cover': 'cloud_cover_high'
        }
        
        # Create a new DataFrame with standardized column names
        processed_data = pd.DataFrame()
        
        # Map existing columns to standardized names
        required_features = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 
                           'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high']
        
        # Check which columns are available in the input data
        available_columns = {}
        for std_col in required_features:
            possible_cols = [col for col, mapped_col in column_mapping.items() if mapped_col == std_col]
            found = False
            for col in possible_cols:
                if col in data.columns:
                    available_columns[std_col] = col
                    found = True
                    break
            if not found:
                if std_col in ['cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high']:
                    # Cloud cover columns are optional, fill with zeros if missing
                    available_columns[std_col] = None
                else:
                    raise ValueError(f"Missing required column: {std_col}")
        
        # Debug logging
        logger.info(f"Available columns mapping: {available_columns}")
        
        # Process timestamps
        if 'date' in data.columns:
            processed_data['date'] = pd.to_datetime(data['date'])
        else:
            raise ValueError("Missing date column")
        
        # Copy data with standardized column names
        for std_col, orig_col in available_columns.items():
            if orig_col is not None:
                processed_data[std_col] = data[orig_col]
                # Debug logging for each column
                logger.info(f"Column {std_col} stats - NaN count: {data[orig_col].isna().sum()}, Range: [{data[orig_col].min()}, {data[orig_col].max()}]")
            else:
                processed_data[std_col] = 0.0  # Fill missing cloud cover with zeros
        
        # Check for actual temperature
        actual_temp = None
        for actual_col in ['temp_avg', 'temperature_obs']:
            if actual_col in data.columns:
                actual_temp = data[actual_col].values
                # Debug logging for actual temperature
                logger.info(f"Found actual temperature column: {actual_col}")
                logger.info(f"Actual temperature stats - NaN count: {np.isnan(actual_temp).sum()}, Range: [{np.nanmin(actual_temp)}, {np.nanmax(actual_temp)}]")
                break
        
        # Prepare feature array
        features = np.column_stack([
            processed_data[col] for col in required_features
        ])
        
        # Final debug logging
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Features NaN count: {np.isnan(features).sum()}")
        
        return features, processed_data['date'].values, actual_temp
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

def plot_temperature_forecast(original, corrected, timestamps):
    """
    Plot the original and corrected temperature forecasts.
    
    Args:
        original (np.ndarray): Original temperature forecast
        corrected (np.ndarray): Corrected temperature forecast
        timestamps (pd.DatetimeIndex): Timestamps for the forecasts
        
    Returns:
        plotly.graph_objects.Figure: Plot figure
    """
    # Debug logging
    logger.info(f"Plotting data shapes - Original: {original.shape}, Corrected: {corrected.shape}, Timestamps: {len(timestamps)}")
    logger.info(f"NaN check - Original: {np.isnan(original).sum()}, Corrected: {np.isnan(corrected).sum()}")
    logger.info(f"Sample values - Original: {original[:5]}, Corrected: {corrected[:5]}")
    
    # Convert timestamps to datetime if they aren't already
    if isinstance(timestamps[0], (str, np.datetime64)):
        timestamps = pd.to_datetime(timestamps)
    
    # Handle any NaN values by interpolating
    original_clean = pd.Series(original).interpolate().values
    corrected_clean = pd.Series(corrected).interpolate().values
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=original_clean,
        mode='lines',
        name='Original Forecast',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=corrected_clean,
        mode='lines',
        name='Bias-Corrected Forecast',
        line=dict(color='green', width=2)
    ))
    
    fig.update_layout(
        title='Temperature Forecast Comparison',
        xaxis_title='Time',
        yaxis_title='Temperature (°C)',
        legend=dict(x=0, y=1, traceorder='normal'),
        hovermode='x unified'
    )
    
    return fig

def plot_bias_correction(bias, timestamps):
    """
    Plot the bias correction applied to the forecast.
    
    Args:
        bias (np.ndarray): Bias correction
        timestamps (pd.DatetimeIndex): Timestamps for the forecasts
        
    Returns:
        plotly.graph_objects.Figure: Plot figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=bias,
        mode='lines',
        name='Bias Correction',
        line=dict(color='red', width=2)
    ))
    
    # Add a zero line for reference - handle empty timestamps safely
    if len(timestamps) > 0:
        fig.add_shape(
            type="line",
            x0=timestamps.iloc[0] if hasattr(timestamps, 'iloc') else timestamps[0],
            y0=0,
            x1=timestamps.iloc[-1] if hasattr(timestamps, 'iloc') else timestamps[-1],
            y1=0,
            line=dict(color="gray", width=1, dash="dash"),
        )
    
    fig.update_layout(
        title='Bias Correction Applied',
        xaxis_title='Time',
        yaxis_title='Bias (°C)',
        legend=dict(x=0, y=1, traceorder='normal'),
        hovermode='x unified'
    )
    
    return fig

def plot_uncertainty(corrected, uncertainty, timestamps):
    """
    Plot the uncertainty of the corrected forecast.
    
    Args:
        corrected (np.ndarray): Corrected temperature forecast
        uncertainty (np.ndarray): Uncertainty of the corrected forecast
        timestamps (pd.DatetimeIndex): Timestamps for the forecasts
        
    Returns:
        plotly.graph_objects.Figure: Plot figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=corrected,
        mode='lines',
        name='Corrected Forecast',
        line=dict(color='green', width=2)
    ))
    
    # Add upper and lower bounds with safety checks
    if len(timestamps) > 0 and len(corrected) > 0 and len(uncertainty) > 0:
        # Upper bound
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=corrected + uncertainty,
            mode='lines',
            name='Upper Bound',
            line=dict(color='gray', width=1, dash="dash")
        ))
        
        # Lower bound
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=corrected - uncertainty,
            mode='lines',
            name='Lower Bound',
            line=dict(color='gray', width=1, dash="dash")
        ))
    
    fig.update_layout(
        title='Uncertainty of Corrected Forecast',
        xaxis_title='Time',
        yaxis_title='Temperature (°C)',
        legend=dict(x=0, y=1, traceorder='normal'),
        hovermode='x unified'
    )
    
    return fig

def display_metrics(metrics: Dict[str, float], has_actual_temps: bool):
    """
    Display the metrics in a Streamlit table.
    
    Args:
        metrics (Dict[str, float]): Dictionary of metrics
        has_actual_temps (bool): Whether actual temperatures are available
    """
    st.subheader("Results")
    
    if has_actual_temps:
        st.write("Evaluation Metrics (comparing corrected forecast with actual temperatures):")
    else:
        st.write("Predicted Bias Statistics (no actual temperatures available for evaluation):")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Mean Bias (°C)",
            f"{metrics['mean_bias']:.4f}",
            help="Mean Bias: Average difference between original and corrected forecasts. Should be near 0 for unbiased corrections."
        )
    
    with col2:
        st.metric(
            "RMSE (°C)",
            f"{metrics['rmse']:.4f}",
            help="RMSE: Root Mean Squared Error—measures correction magnitude. Should be low (~1-2°C)."
        )
    
    with col3:
        st.metric(
            "MAE (°C)",
            f"{metrics['mae']:.4f}",
            help="MAE: Mean Absolute Error—average correction size. Should match typical bias (~1-2°C)."
        )

def main():
    """Main Streamlit app function."""
    # Set page config
    st.set_page_config(
        page_title="Weather Forecast Bias Correction",
        page_icon="🌤️",
        layout="wide"
    )
    
    # Add title and description
    st.title("Weather Forecast Bias Correction")
    st.markdown("""
    This application helps correct biases in weather forecasts using deep learning.
    Upload your forecast data to get bias-corrected predictions.
    """)
    
    # Create sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Model selection
        model_path = st.text_input(
            "Model Path",
            value="C:\\Users\\SeoYea-Ji\\weather_bias_correction\\logs\\final_fix\\checkpoints\\bias_correction-epoch=16-val_loss=0.00.ckpt",
            help="Path to the trained model checkpoint"
        )
        
        # Load model button
        if st.button("Load Model"):
            try:
                st.session_state.model_server = load_model(model_path)
                st.success("Model loaded successfully!")
                
                # Display model info
                model_info = st.session_state.model_server.get_model_info()
                st.json(model_info)
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
        
        # Data upload
        st.header("Data Upload")
        uploaded_file = st.file_uploader(
            "Upload forecast data (CSV format)",
            type=['csv'],
            help="Upload your forecast data in CSV format"
        )
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Input Data")
        if uploaded_file is not None:
            # Load and display data
            data = pd.read_csv(uploaded_file)
            st.dataframe(data)
            
            # Add process button
            if st.button("Process Data"):
                if st.session_state.model_server is None:
                    st.warning("Please load the model first.")
                else:
                    try:
                        # Process data
                        features, timestamps, actual_temps = process_data(data)
                        
                        # Make predictions
                        predictions, uncertainty = st.session_state.model_server.predict(
                            features,
                            return_uncertainty=True,
                            batch_size=32
                        )
                        
                        # Calculate corrected temperatures
                        corrected_temps = features[:, 0] - predictions.squeeze()
                        
                        # Store results in session state
                        st.session_state.results = {
                            'original': features[:, 0],  # Original forecast temperature
                            'bias': predictions.squeeze(),  # Predicted bias
                            'corrected': corrected_temps,  # Corrected temperatures
                            'uncertainty': uncertainty.squeeze() if uncertainty is not None else None,
                            'timestamps': timestamps,
                            'actual_temps': actual_temps  # May be None if not available
                        }
                        
                        # Calculate metrics
                        if actual_temps is not None:
                            # Calculate metrics against actual temperatures
                            error = corrected_temps - actual_temps
                            metrics = {
                                'mean_bias': np.mean(error),
                                'rmse': np.sqrt(np.mean(error**2)),
                                'mae': np.mean(np.abs(error))
                            }
                        else:
                            # Just show statistics of the predicted bias
                            metrics = {
                                'mean_bias': np.mean(predictions),
                                'rmse': np.sqrt(np.mean(predictions**2)),
                                'mae': np.mean(np.abs(predictions))
                            }
                        
                        st.session_state.metrics = {
                            'values': metrics,
                            'has_actual_temps': actual_temps is not None
                        }
                        
                        st.success("Data processed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")
        else:
            st.info("Please upload your forecast data to begin.")
    
    with col2:
        st.header("Results")
        if 'results' in st.session_state:
            results = st.session_state.results
            
            # Display metrics
            display_metrics(st.session_state.metrics['values'], st.session_state.metrics['has_actual_temps'])
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs([
                "Temperature Forecast",
                "Bias Analysis",
                "Uncertainty"
            ])
            
            with tab1:
                # Plot temperature forecast
                fig = plot_temperature_forecast(
                    results['original'],
                    results['corrected'],
                    results['timestamps']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Plot bias analysis
                fig = plot_bias_correction(
                    results['bias'],  # Use the predicted bias directly
                    results['timestamps']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                if results['uncertainty'] is not None:
                    # Plot uncertainty
                    fig = plot_uncertainty(
                        results['corrected'],
                        results['uncertainty'],
                        results['timestamps']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Uncertainty estimates are not available.")
        else:
            st.info("Results will appear here after processing the data.")

if __name__ == '__main__':
    main()