"""Comprehensive tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, reset_predictor


@pytest.fixture(autouse=True)
def reset_model():
    """Reset predictor before each test to ensure clean state."""
    reset_predictor()
    yield
    reset_predictor()


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthCheck:
    """Tests for the health check endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_health_returns_status(self, client):
        """Health endpoint should return status field."""
        response = client.get("/")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_returns_model_loaded(self, client):
        """Health endpoint should indicate if model is loaded."""
        response = client.get("/")
        data = response.json()
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)

    def test_health_returns_version(self, client):
        """Health endpoint should return version."""
        response = client.get("/")
        data = response.json()
        assert "version" in data


class TestPrediction:
    """Tests for the prediction endpoint."""

    def test_predict_returns_200(self, client):
        """Prediction endpoint should return 200 for valid input."""
        request_data = {
            "hour_of_day": 18,
            "day_of_week": 2,
            "is_weekend": False,
            "passenger_demand": 50,
            "driver_availability": 10,
            "weather_condition": "rain",
            "temperature": 12.5,
            "visibility_km": 5.0,
            "base_fare": 8.50,
        }
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200

    def test_predict_returns_expected_fields(self, client):
        """Prediction should return multiplier, price, and confidence."""
        request_data = {
            "hour_of_day": 18,
            "day_of_week": 2,
            "is_weekend": False,
            "passenger_demand": 50,
            "driver_availability": 10,
            "weather_condition": "rain",
            "temperature": 12.5,
            "visibility_km": 5.0,
            "base_fare": 8.50,
        }
        response = client.post("/predict", json=request_data)
        data = response.json()

        assert "price_multiplier" in data
        assert "estimated_price" in data
        assert "confidence" in data

    def test_predict_multiplier_range(self, client):
        """Price multiplier should be between 1.0 and 3.0."""
        request_data = {
            "hour_of_day": 12,
            "day_of_week": 0,
            "is_weekend": False,
            "passenger_demand": 30,
            "driver_availability": 20,
            "weather_condition": "clear",
            "temperature": 20.0,
            "visibility_km": 15.0,
            "base_fare": 5.0,
        }
        response = client.post("/predict", json=request_data)
        data = response.json()

        assert 1.0 <= data["price_multiplier"] <= 3.0

    def test_predict_confidence_valid(self, client):
        """Confidence should be low, medium, or high."""
        request_data = {
            "hour_of_day": 18,
            "day_of_week": 2,
            "is_weekend": False,
            "passenger_demand": 50,
            "driver_availability": 10,
            "weather_condition": "rain",
            "temperature": 12.5,
            "visibility_km": 5.0,
            "base_fare": 8.50,
        }
        response = client.post("/predict", json=request_data)
        data = response.json()

        assert data["confidence"] in ["low", "medium", "high"]

    def test_predict_estimated_price_calculation(self, client):
        """Estimated price should be base_fare * multiplier."""
        request_data = {
            "hour_of_day": 18,
            "day_of_week": 2,
            "is_weekend": False,
            "passenger_demand": 50,
            "driver_availability": 10,
            "weather_condition": "rain",
            "temperature": 12.5,
            "visibility_km": 5.0,
            "base_fare": 10.0,
        }
        response = client.post("/predict", json=request_data)
        data = response.json()

        expected_price = 10.0 * data["price_multiplier"]
        assert abs(data["estimated_price"] - expected_price) < 0.1

    def test_predict_high_demand_higher_multiplier(self, client):
        """Higher demand should produce higher multiplier."""
        base_request = {
            "hour_of_day": 12,
            "day_of_week": 0,
            "is_weekend": False,
            "driver_availability": 10,
            "weather_condition": "clear",
            "temperature": 20.0,
            "visibility_km": 10.0,
            "base_fare": 5.0,
        }

        # Low demand
        low_demand = base_request.copy()
        low_demand["passenger_demand"] = 5
        response_low = client.post("/predict", json=low_demand)

        # High demand
        high_demand = base_request.copy()
        high_demand["passenger_demand"] = 100
        response_high = client.post("/predict", json=high_demand)

        assert response_high.json()["price_multiplier"] >= response_low.json()["price_multiplier"]

    def test_predict_invalid_weather_returns_422(self, client):
        """Invalid weather condition should return 422."""
        request_data = {
            "hour_of_day": 18,
            "day_of_week": 2,
            "is_weekend": False,
            "passenger_demand": 50,
            "driver_availability": 10,
            "weather_condition": "hurricane",  # Invalid
            "temperature": 12.5,
            "visibility_km": 5.0,
            "base_fare": 8.50,
        }
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422

    def test_predict_invalid_hour_returns_422(self, client):
        """Invalid hour (25) should return 422."""
        request_data = {
            "hour_of_day": 25,  # Invalid
            "day_of_week": 2,
            "is_weekend": False,
            "passenger_demand": 50,
            "driver_availability": 10,
            "weather_condition": "rain",
            "temperature": 12.5,
            "visibility_km": 5.0,
            "base_fare": 8.50,
        }
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422

    def test_predict_negative_fare_returns_422(self, client):
        """Negative base fare should return 422."""
        request_data = {
            "hour_of_day": 18,
            "day_of_week": 2,
            "is_weekend": False,
            "passenger_demand": 50,
            "driver_availability": 10,
            "weather_condition": "rain",
            "temperature": 12.5,
            "visibility_km": 5.0,
            "base_fare": -5.0,  # Invalid
        }
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422

    def test_predict_missing_field_returns_422(self, client):
        """Missing required field should return 422."""
        request_data = {
            "hour_of_day": 18,
            "day_of_week": 2,
            # Missing is_weekend and other fields
        }
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422

    def test_predict_all_weather_conditions(self, client):
        """Should accept all valid weather conditions."""
        for weather in ["clear", "cloudy", "rain", "storm", "snow"]:
            request_data = {
                "hour_of_day": 12,
                "day_of_week": 0,
                "is_weekend": False,
                "passenger_demand": 30,
                "driver_availability": 15,
                "weather_condition": weather,
                "temperature": 20.0,
                "visibility_km": 10.0,
                "base_fare": 5.0,
            }
            response = client.post("/predict", json=request_data)
            assert response.status_code == 200, f"Failed for weather: {weather}"

    def test_predict_weekend(self, client):
        """Should handle weekend requests."""
        request_data = {
            "hour_of_day": 14,
            "day_of_week": 6,  # Sunday
            "is_weekend": True,
            "passenger_demand": 40,
            "driver_availability": 8,
            "weather_condition": "clear",
            "temperature": 25.0,
            "visibility_km": 15.0,
            "base_fare": 6.0,
        }
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200

    def test_predict_late_night(self, client):
        """Should handle late night requests."""
        request_data = {
            "hour_of_day": 3,  # 3 AM
            "day_of_week": 0,
            "is_weekend": False,
            "passenger_demand": 5,
            "driver_availability": 3,
            "weather_condition": "clear",
            "temperature": 5.0,
            "visibility_km": 10.0,
            "base_fare": 4.0,
        }
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200


class TestModelInfo:
    """Tests for the model info endpoint."""

    def test_model_info_returns_200(self, client):
        """Model info endpoint should return 200."""
        response = client.get("/model-info")
        assert response.status_code == 200

    def test_model_info_has_required_fields(self, client):
        """Model info should have model_name, model_type, features, metrics."""
        response = client.get("/model-info")
        data = response.json()

        assert "model_name" in data
        assert "model_type" in data
        assert "features" in data
        assert "metrics" in data

    def test_model_info_features_is_list(self, client):
        """Features should be a list."""
        response = client.get("/model-info")
        data = response.json()
        assert isinstance(data["features"], list)

    def test_model_info_metrics_is_dict(self, client):
        """Metrics should be a dictionary."""
        response = client.get("/model-info")
        data = response.json()
        assert isinstance(data["metrics"], dict)
