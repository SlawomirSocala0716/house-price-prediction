# API Service for House Price Prediction
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware  # Middleware for handling cross-origin HTML connections
import uvicorn
import os
from contextlib import asynccontextmanager  # Handles application startup and shutdown events
import psycopg2  # PostgreSQL database adapter

from pydantic import BaseModel, Field # this is for establishing adequate values to avoid situation e.g.: bedroom -1

# Importing response and path utilities
from fastapi.responses import FileResponse
from pathlib import Path

# Import the prediction logic from Step 9
from src.step9_predict import predict_price

BASE_DIR = Path(__file__).resolve().parent

# Database initialization logic
def init_db():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("WARNING: DATABASE_URL environment variable not found. Database logging is disabled.")
        return
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        # Create table schema if it doesn't exist yet
        cur.execute("""
            CREATE TABLE IF NOT EXISTS house_predictions (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                bedrooms INT,
                bathrooms FLOAT,
                sqft_living INT,
                sqft_lot INT,
                floors FLOAT,
                waterfront INT,
                view INT,
                condition INT,
                sqft_basement INT,
                predicted_price FLOAT
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("INFO: PostgreSQL database initialized successfully (table is ready).")
    except Exception as e:
        print(f"ERROR: Database initialization failed: {str(e)}")

# Define application lifespan context to trigger DB setup on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()  # Run the database configuration when the server starts
    yield

# 1. Initialize FastAPI application with lifespan management
app = FastAPI(
    title="House Price Champion API",
    description="A professional ML service for real estate valuation using a Hybrid Ensemble model.",
    version="1.0.0",
    lifespan=lifespan
)
# 2. ADD CORS MIDDLEWARE for HTML client communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Allows index.html frontend to communicate with the API
    allow_credentials=True,
    allow_methods=["*"],   # Enables all HTTP methods such as POST, GET, etc.
    allow_headers=["*"],
)

# 3. Define the data schema
class HouseFeatures(BaseModel):
    Bedrooms: int = Field(..., ge=0, le=15, description="Number of bedrooms (0 to 15)")
    Bathrooms: float = Field(..., ge=0.5, le=10.0, description="Number of bathrooms (0.5 to 10) (in USA 0.5 = halfbath)")
    Sqft_Living: int = Field(..., ge=150, le=15000, description="Lot size (square feet)")
    Sqft_lot: int = Field(..., ge=150, le=1000000, description="Number of floors (1 to 5)")
    Floors: float = Field(..., ge=1.0, le=5.0, description="Number of floors (1 to 5) (in USA ground is the first floor)")
    Waterfront: int = Field(..., ge=0, le=1, description="Waterfront: 0 = no, 1 = yes")
    view: int = Field(..., ge=0, le=4, description="View rating (0 to 4)")
    Condition: int = Field(..., ge=1, le=5, description="Condition rating (1 to 5)")
    Sqft_Basement: int = Field(..., ge=0, le=10000, description="Basement area (0 if none))")

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
    html_path = BASE_DIR / "index.html"  # Path pointing to the frontend template
    return FileResponse(html_path)

@app.get("/analytics", tags=["Analytics"])
async def get_analytics():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise HTTPException(status_code=500, detail="Database connection string not found.")
    
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        # using SQL we can see have many entries were conducted and will see average questions 
        cur.execute("""
            SELECT 
                COUNT(*) as total_searches,
                AVG(predicted_price) as avg_price,
                AVG(sqft_living) as avg_living,
                AVG(bedrooms) as avg_beds
            FROM house_predictions;
        """)
        
        row = cur.fetchone()
        cur.close()
        conn.close()
        
        # the data will be in JSON format 
        return {
            "status": "success",
            "metrics": {
                "total_predictions_made": row[0] if row[0] else 0,
                "average_predicted_price": round(row[1], 2) if row[1] else 0.0,
                "average_sqft_living": round(row[2], 1) if row[2] else 0.0,
                "average_bedrooms_requested": round(row[3], 1) if row[3] else 0.0
            }
        }
    except Exception as e:
        print(f"ERROR: Failed to fetch data for analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while fetching metrics.")

@app.post("/predict", tags=["Machine Learning"])
async def get_prediction(house: HouseFeatures):
    print(f"DEBUG: Input data received: {house}")
    try:
        input_data = house.model_dump()
        estimated_price = predict_price(input_data)
        
        print(f"DEBUG: Prediction generated: ${estimated_price:,.2f}")
        
        # --- POSTGRESQL DATA LOGGING LAYER ---
        db_url = os.environ.get("DATABASE_URL")
        if db_url:
            try:
                conn = psycopg2.connect(db_url)
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO house_predictions 
                    (bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, sqft_basement, predicted_price)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """, (
                    input_data.get('Bedrooms'),
                    input_data.get('Bathrooms'),
                    input_data.get('Sqft_Living'),
                    input_data.get('Sqft_lot'),
                    input_data.get('Floors'),
                    input_data.get('Waterfront'),
                    input_data.get('view'),
                    input_data.get('Condition'),
                    input_data.get('Sqft_Basement'),
                    estimated_price
                ))
                conn.commit()
                cur.close()
                conn.close()
                print("DEBUG: Prediction successfully logged to PostgreSQL!")
            except Exception as db_err:
                # Catch DB errors safely so the user still receives their prediction
                print(f"WARNING: Failed to log prediction to DB: {str(db_err)}")
        # -------------------------------------
        
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