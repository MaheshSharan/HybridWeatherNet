import pytest
from fastapi.testclient import TestClient
from src.web.app import app
from src.web.models import BiasCorrectionRequest, BiasCorrectionResponse
from src.web.cache import cache
import json

@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)

@pytest.fixture
def sample_request_data():
    """Create sample request data for testing."""
    return {
        "forecast_data": {
            "temperature": [20.0, 21.0, 22.0],
            "humidity": [60.0, 65.0, 70.0],
            "wind_speed": [5.0, 6.0, 7.0],
            "pressure": [1013.0, 1012.0, 1011.0],
            "precipitation": [0.0, 0.1, 0.2]
        },
        "location": {
            "latitude": 37.5665,
            "longitude": 126.9780
        },
        "timestamp": "2024-03-20T12:00:00Z"
    }

def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_bias_correction_endpoint(client, sample_request_data):
    """Test the bias correction endpoint."""
    response = client.post(
        "/api/v1/bias-correction",
        json=sample_request_data
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "corrected_forecast" in data
    assert "metrics" in data
    assert "timestamp" in data
    
    # Check corrected forecast structure
    corrected = data["corrected_forecast"]
    assert "temperature" in corrected
    assert "humidity" in corrected
    assert "wind_speed" in corrected
    assert "pressure" in corrected
    assert "precipitation" in corrected
    
    # Check metrics structure
    metrics = data["metrics"]
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "bias" in metrics

def test_invalid_request_data(client):
    """Test the bias correction endpoint with invalid data."""
    invalid_data = {
        "forecast_data": {
            "temperature": [20.0, 21.0, 22.0]
        }
    }
    
    response = client.post(
        "/api/v1/bias-correction",
        json=invalid_data
    )
    
    assert response.status_code == 422

def test_cache_functionality(client, sample_request_data):
    """Test the caching functionality."""
    # First request
    response1 = client.post(
        "/api/v1/bias-correction",
        json=sample_request_data
    )
    assert response1.status_code == 200
    
    # Second request with same data
    response2 = client.post(
        "/api/v1/bias-correction",
        json=sample_request_data
    )
    assert response2.status_code == 200
    
    # Check if responses are identical
    assert response1.json() == response2.json()

def test_model_loading():
    """Test if the model is loaded correctly."""
    from src.web.models import model
    assert model is not None
    assert hasattr(model, "forward")
    assert hasattr(model, "predict")

def test_error_handling(client):
    """Test error handling for various scenarios."""
    # Test with empty request
    response = client.post("/api/v1/bias-correction", json={})
    assert response.status_code == 422
    
    # Test with invalid JSON
    response = client.post(
        "/api/v1/bias-correction",
        data="invalid json",
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    
    # Test with missing required fields
    invalid_data = {
        "forecast_data": {},
        "location": {}
    }
    response = client.post(
        "/api/v1/bias-correction",
        json=invalid_data
    )
    assert response.status_code == 422

def test_response_validation(client, sample_request_data):
    """Test if the response matches the expected schema."""
    response = client.post(
        "/api/v1/bias-correction",
        json=sample_request_data
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Validate response against schema
    try:
        BiasCorrectionResponse(**data)
    except Exception as e:
        pytest.fail(f"Response validation failed: {str(e)}")

def test_request_validation(sample_request_data):
    """Test if the request data matches the expected schema."""
    try:
        BiasCorrectionRequest(**sample_request_data)
    except Exception as e:
        pytest.fail(f"Request validation failed: {str(e)}")

def test_metrics_calculation(client, sample_request_data):
    """Test if the metrics are calculated correctly."""
    response = client.post(
        "/api/v1/bias-correction",
        json=sample_request_data
    )
    
    assert response.status_code == 200
    data = response.json()
    metrics = data["metrics"]
    
    # Check if metrics are within expected ranges
    assert -100 <= metrics["mae"] <= 100
    assert -100 <= metrics["rmse"] <= 100
    assert -100 <= metrics["bias"] <= 100 