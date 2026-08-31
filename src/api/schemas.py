"""Pydantic schemas for the Dynamic Pricing API request/response models."""

from pydantic import BaseModel, Field


class RideRequest(BaseModel):
    """Input schema for ride price prediction."""

    hour_of_day: int = Field(
        ..., ge=0, le=23, description="Hour of the day (0-23)", examples=[18]
    )
    day_of_week: int = Field(
        ..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)", examples=[2]
    )
    is_weekend: bool = Field(
        ..., description="Whether it is a weekend", examples=[False]
    )
    passenger_demand: int = Field(
        ..., ge=0, description="Number of ride requests in the area", examples=[50]
    )
    driver_availability: int = Field(
        ..., ge=0, description="Number of available drivers", examples=[10]
    )
    weather_condition: str = Field(
        ...,
        description="Weather condition",
        examples=["clear"],
        pattern="^(clear|cloudy|rain|storm|snow)$",
    )
    temperature: float = Field(
        ..., description="Temperature in Celsius", examples=[12.5]
    )
    visibility_km: float = Field(
        ..., ge=0, description="Visibility in kilometers", examples=[5.0]
    )
    base_fare: float = Field(
        ..., gt=0, description="Base fare before multiplier", examples=[8.50]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Output schema for ride price prediction."""

    price_multiplier: float = Field(
        ..., description="Predicted surge multiplier (1.0-3.0)"
    )
    estimated_price: float = Field(
        ..., description="Estimated final price (base_fare * multiplier)"
    )
    confidence: str = Field(
        ..., description="Prediction confidence level", pattern="^(low|medium|high)$"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    model_loaded: bool = True
    version: str = "1.0.0"


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_name: str
    model_type: str
    features: list[str]
    metrics: dict
