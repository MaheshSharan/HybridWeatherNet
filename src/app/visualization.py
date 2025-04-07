import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Tuple

def plot_temperature_forecast(
    original: np.ndarray,
    corrected: np.ndarray,
    timestamps: List[str],
    title: str = "Temperature Forecast Comparison"
) -> go.Figure:
    """
    Create a plot comparing original and bias-corrected temperature forecasts.
    
    Args:
        original (np.ndarray): Original temperature forecasts
        corrected (np.ndarray): Bias-corrected temperature forecasts
        timestamps (List[str]): List of timestamps
        title (str): Plot title
        
    Returns:
        go.Figure: Plotly figure object
    """
    fig = go.Figure()
    
    # Add original forecast
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=original,
            name="Original Forecast",
            line=dict(color="blue", dash="dash")
        )
    )
    
    # Add corrected forecast
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=corrected,
            name="Bias-Corrected Forecast",
            line=dict(color="red")
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        hovermode="x unified"
    )
    
    return fig

def plot_bias_correction(
    original: np.ndarray,
    corrected: np.ndarray,
    timestamps: List[str],
    title: str = "Bias Correction Analysis"
) -> go.Figure:
    """
    Create a plot showing the bias correction analysis.
    
    Args:
        original (np.ndarray): Original temperature forecasts
        corrected (np.ndarray): Bias-corrected temperature forecasts
        timestamps (List[str]): List of timestamps
        title (str): Plot title
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Calculate bias
    bias = original - corrected
    
    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Temperature Forecasts", "Bias"),
        vertical_spacing=0.2
    )
    
    # Add temperature forecasts
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=original,
            name="Original Forecast",
            line=dict(color="blue", dash="dash")
        ),
        row=1,
        col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=corrected,
            name="Bias-Corrected Forecast",
            line=dict(color="red")
        ),
        row=1,
        col=1
    )
    
    # Add bias
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=bias,
            name="Bias",
            line=dict(color="green")
        ),
        row=2,
        col=1
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=800,
        showlegend=True
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Bias (°C)", row=2, col=1)
    
    # Update x-axes labels
    fig.update_xaxes(title_text="Time", row=2, col=1)
    
    return fig

def plot_uncertainty(
    mean: np.ndarray,
    std: np.ndarray,
    timestamps: List[str],
    title: str = "Forecast Uncertainty"
) -> go.Figure:
    """
    Create a plot showing the forecast uncertainty.
    
    Args:
        mean (np.ndarray): Mean temperature forecast
        std (np.ndarray): Standard deviation of temperature forecast
        timestamps (List[str]): List of timestamps
        title (str): Plot title
        
    Returns:
        go.Figure: Plotly figure object
    """
    fig = go.Figure()
    
    # Add mean forecast
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=mean,
            name="Mean Forecast",
            line=dict(color="blue")
        )
    )
    
    # Add uncertainty bands
    fig.add_trace(
        go.Scatter(
            x=timestamps + timestamps[::-1],
            y=np.concatenate([mean + 2*std, (mean - 2*std)[::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        hovermode="x unified"
    )
    
    return fig

def display_metrics(metrics: Dict[str, float]) -> None:
    """
    Display evaluation metrics in the Streamlit app.
    
    Args:
        metrics (Dict[str, float]): Dictionary of metric names and values
    """
    # Create columns for metrics
    cols = st.columns(len(metrics))
    
    # Display each metric
    for col, (metric_name, value) in zip(cols, metrics.items()):
        with col:
            st.metric(
                label=metric_name.replace('_', ' ').title(),
                value=f"{value:.4f}"
            ) 