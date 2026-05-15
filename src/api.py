# API Service for House Price Prediction
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware  # !new library for .html
import uvicorn
import os

from fastapi.responses import FileResponse # new, for creating website page for user

# Import the prediction logic from your Step 9
from src.step9_predict import predict_price

# 1. Initialize FastAPI application
app = FastAPI(
    title="House Price Champion API",
    description="A professional ML service for real estate valuation using a Hybrid Ensemble model.",
    version="1.0.0"
)

# 2. !NEW - ADD CORS MIDDLEWARE for .html 
# we establish middleware directly after FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Allows index.html frontend to communicate with the API
    allow_credentials=True,
    allow_methods=["*"],   # Enables all HTTP methods such as POST, GET, etc.
    allow_headers=["*"],
)

# 3. Define the data schema
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
    return FileResponse("src/index.html")

@app.post("/predict", tags=["Machine Learning"])
async def get_prediction(house: HouseFeatures):
    try:
        input_data = house.model_dump()
        estimated_price = predict_price(input_data)
        
        print(f"DEBUG: Prediction generated: ${estimated_price:,.2f}")
        
        return {
            "status": "success",
            "estimated_price": estimated_price,
            "currency": "USD",
            "model_version": "1.0.0-hybrid"
        }
    except Exception as e:
        print(f"ERROR: Prediction failed. Details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)