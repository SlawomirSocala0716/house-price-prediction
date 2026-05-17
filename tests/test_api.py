# libraries for importing path 
import sys
from pathlib import Path

# path to the test file
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_read_root():
    """1. Test check if website work"""
    response = client.get("/")
    assert response.status_code == 200
    assert "<html" in response.text

def test_predict_validation_error_negative():
    """2. Test check if -5 bedroom cause ERROR 422 (because -5 is invalid value)"""
    invalid_data = {
        "Bedrooms": -5,
        "Bathrooms": 2.5,
        "Sqft_Living": 2100,
        "Sqft_lot": 5000,
        "Floors": 2.0,
        "Waterfront": 0,
        "view": 0,
        "Condition": 3,
        "Sqft_Basement": 800
    }
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422

def test_predict_success_standard_house():
    """3. Test check, if after input correct/realistic data we get status 200"""
    valid_data = {
        "Bedrooms": 3,
        "Bathrooms": 2.0,
        "Sqft_Living": 1800,
        "Sqft_lot": 4000,
        "Floors": 1.5,
        "Waterfront": 0,
        "view": 0,
        "Condition": 4,
        "Sqft_Basement": 0
    }
    response = client.post("/predict", json=valid_data)
    
    assert response.status_code == 200
    
    # Sprawdzane jest, czy odpowiedź JSON zawiera wszystkie pola wymagane przez frontend
    data = response.json()
    assert data["status"] == "success"
    assert "estimated_price" in data
    assert data["currency"] == "USD"
    assert "model_version" in data
    # price > 0
    assert data["estimated_price"] > 0

def test_predict_extreme_values():
    """4. Model robustness test: Checking whether the model avoids producing a negative price for an extremely large luxury residence."""
    extreme_data = {
        "Bedrooms": 14,
        "Bathrooms": 9.5,
        "Sqft_Living": 14000,
        "Sqft_lot": 900000,
        "Floors": 4.0,
        "Waterfront": 1,
        "view": 4,
        "Condition": 5,
        "Sqft_Basement": 5000
    }
    response = client.post("/predict", json=extreme_data)
    
    assert response.status_code == 200
    data = response.json()
    # here we check if model not generate negative price
    assert data["estimated_price"] > 0