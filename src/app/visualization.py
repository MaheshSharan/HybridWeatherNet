import plotly.graph_objects as go
import numpy as np
from typing import Optional

def plot_temperature_forecast(
    original: np.ndarray,
    corrected: np.ndarray,
    timestamps: np.ndarray
) -> go.Figure:
    """Plot original vs corrected temperature forecasts."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=original,
        name="Original Forecast",
        line=dict(color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=corrected,
        name="Corrected Forecast",
        line=dict(color="green")
    ))
    # Add difference trace
    diff = corrected - original
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=diff,
        name="Predicted Bias",
        line=dict(color="red", dash="dash")
    ))
    fig.update_layout(
        title="Temperature Forecast Comparison",
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

def plot_bias_correction(
    original: np.ndarray,
    corrected: np.ndarray,
    timestamps: np.ndarray
) -> go.Figure:
    """Plot bias correction over time."""
    bias = corrected - original  # Predicted bias
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=bias,
        name="Predicted Bias",
        line=dict(color="purple")
    ))
    # Add zero line for reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Zero Bias")
    fig.update_layout(
        title="Bias Correction Analysis",
        xaxis_title="Time",
        yaxis_title="Bias (°C)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

def plot_uncertainty(
    corrected: np.ndarray,
    uncertainty: np.ndarray,
    timestamps: np.ndarray
) -> go.Figure:
    """Plot corrected forecast with uncertainty bounds."""
    fig = go.Figure()
    std = np.sqrt(uncertainty)  # Convert variance to std dev
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=corrected,
        name="Corrected Forecast",
        line=dict(color="green")
    ))
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=corrected + 2 * std,
        name="Upper Bound (95%)",
        line=dict(color="green", dash="dash"),
        opacity=0.3
    ))
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=corrected - 2 * std,
        name="Lower Bound (95%)",
        line=dict(color="green", dash="dash"),
        opacity=0.3,
        fill="tonexty"  # Fill between bounds
    ))
    fig.update_layout(
        title="Corrected Forecast with Uncertainty",
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

def display_metrics(metrics: dict) -> None:
    """Display performance metrics in Streamlit."""
    import streamlit as st
    st.subheader("Performance Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Bias (°C)", f"{metrics['mean_bias']:.4f}")
    with col2:
        st.metric("RMSE (°C)", f"{metrics['rmse']:.4f}")
    with col3:
        st.metric("MAE (°C)", f"{metrics['mae']:.4f}")
    # Add interpretation
    st.write("""
    - **Mean Bias**: Average difference between original and corrected forecasts. Should be near 0 for unbiased corrections.
    - **RMSE**: Root Mean Squared Error—measures correction magnitude. Should be low (~1-2°C).
    - **MAE**: Mean Absolute Error—average correction size. Should match typical bias (~1-2°C).
    """)