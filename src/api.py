# API Service for House Price Prediction

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

# Import the prediction logic from your Step 9
from src.step9_predict import predict_price

# Initialize FastAPI application
app = FastAPI(
    title="House Price Champion API",
    description="A professional ML service for real estate valuation using a Hybrid Ensemble model.",
    version="1.0.0"
)

# Define the data schema for incoming requests
# This acts as a 'Gatekeeper' ensuring the data is valid before hitting the model.
class HouseFeatures(BaseModel):
    Bedrooms: int
    Bathrooms: float
    Sqft_Living: int
    Sqft_lot: int
    Floors: float
    Waterfront: int
    view: int
    Condition: int
    Sqft_Basement: int

    # Configuration for Swagger UI documentation examples
    model_config = {
        "json_schema_extra": {
            "example": {
                "Bedrooms": 3,
                "Bathrooms": 2.5,
                "Sqft_Living": 2100,
                "Sqft_lot": 5000,
                "Floors": 2.0,
                "Waterfront": 0,
                "view": 0,
                "Condition": 3,
                "Sqft_Basement": 800
            }
        }
    }

@app.get("/", tags=["Health Check"])
async def root():
    """Returns a simple message to verify the API is online."""
    return {
        "status": "online",
        "service": "House Price Champion API",
        "documentation": "/docs"
    }

@app.post("/predict", tags=["Machine Learning"])
async def get_prediction(house: HouseFeatures):
    """
    Receives house features, processes them via Step 9, 
    and returns the estimated market price.
    """
    try:
        # Convert Pydantic object to a standard Python dictionary
        input_data = house.model_dump()
        
        # Calculate prediction using the Champion Model
        estimated_price = predict_price(input_data)
        
        print(f"DEBUG: Prediction generated: ${estimated_price:,.2f}")
        
        return {
            "status": "success",
            "estimated_price": estimated_price,
            "currency": "USD",
            "model_version": "1.0.0-hybrid"
        }
    except Exception as e:
        # Log the error and return a 500 Internal Server Error
        print(f"ERROR: Prediction failed. Details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    # Launch the Uvicorn server (host 0.0.0.0 is required for Docker)
    uvicorn.run(app, host="0.0.0.0", port=8000)