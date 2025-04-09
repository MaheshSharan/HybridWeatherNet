import streamlit as st
import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import torch

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
        return ModelServer(
            model_path=model_path,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            cache_size=100
        )
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def process_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process the uploaded data for prediction.
    
    Args:
        data (pd.DataFrame): Uploaded data
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Processed features and timestamps
    """
    try:
        features, timestamps = process_data(data)
        predictions, uncertainty = st.session_state.model_server.predict(
            features,
            return_uncertainty=True,
            batch_size=32
        )
        st.session_state.results = {
            'original': features[:, 0],
            'corrected': predictions.squeeze(),
            'uncertainty': uncertainty.squeeze() if uncertainty is not None else None,
            'timestamps': timestamps
        }
        st.success("Data processed successfully!")
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")

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
                        features, timestamps = process_data(data)
                        
                        # Make predictions
                        predictions, uncertainty = st.session_state.model_server.predict(
                            features,
                            return_uncertainty=True
                        )
                        
                        # Store results in session state
                        st.session_state.results = {
                            'original': features[:, 0],  # Assuming first column is temperature
                            'corrected': predictions.squeeze(),
                            'uncertainty': uncertainty.squeeze() if uncertainty is not None else None,
                            'timestamps': timestamps
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
            metrics = {
                'mean_bias': np.mean(results['original'] - results['corrected']),
                'rmse': np.sqrt(np.mean((results['original'] - results['corrected'])**2)),
                'mae': np.mean(np.abs(results['original'] - results['corrected']))
            }
            display_metrics(metrics)
            
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
                    results['original'],
                    results['corrected'],
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