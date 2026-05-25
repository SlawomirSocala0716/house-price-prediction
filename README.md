# 🏠 Real Estate End-to-End Prediction Platform (MLOps)

An advanced, production-ready MLOps platform designed for real-time house price estimation. The system integrates automated data preprocessing, non-linear feature engineering, a custom hybrid machine learning ensemble, a headless API serving layer, and cloud-native monitoring infrastructure.

🌐 **Live Demo:** [house-price-prediction-439u.onrender.com](https://house-price-prediction-439u.onrender.com/)

---

## 🏗️ System Architecture & Core Features

* **Hybrid Ensemble Champion (R²: 0.5650):** A custom `VotingEnsemble` merging a Linear Regression model (operating in a 2nd-degree polynomial space via OLS) with an optimized, gradient-boosted tree component (`XGBoost`). It balances global market trends with local categorical non-linearities.
* **Encapsulated Target Transformation:** Implements `TransformedTargetRegressor` using a mathematical logarithmic mapping (y* = ln(y + 1)) to stabilize residual variance and protect the inference API from handling manual inverse conversions.
* **User-Centric Frontend UX (m² to sqft):** Built for the European market using Tailwind CSS. It accepts metrics in square meters (`Living Space`, `Lot Space`, `Basement Space`) and dynamically translates them to imperial units required by the US-trained core models.
* **Production Serving Layer:** An asynchronous `FastAPI` instance using `Pydantic` for rigorous input contract validation (e.g., rejecting negative feature values like `-5 bedrooms` with an HTTP 422 code).
* **Advanced MLOps Prediction Store:** Integrates a chmurowa `PostgreSQL` database to log live inference vectors with timestamps for active operational audit:
    * *Data Drift Detection:* Identifies structural shifts in user behaviors vs. historical training distributions.
    * *Scraping & Inversion Protection:* Detects high-frequency IP requests mimicking bot attacks.
    * *Feedback Loop Blueprint:* Collects real-world validation data from users for future continuous learning (CT).
* **CI/CD & Infrastructure as Code:** Containerized via `Docker` (`python:3.12-slim`), continuously integrated using `GitHub Actions` (executing a 4-path integration test matrix via `pytest`), and deployed via GitOps to `Render` using a *Zero-Downtime* strategy.

---

III. 📁 Project Structure

* `.github/workflows/main.yml` - CI/CD pipeline automation setup (GitHub Actions).
* `data/` - Storage for raw data inputs and preprocessed datasets.
* `model_selection/` - Experimental scripts documenting cross-validation and hyperparameter search.
* `models/` - Serialized deployment-ready production model artifacts (`.joblib`).
* `notebooks/` - Structured Jupyter Notebooks covering EDA, prototyping, and final data preparation.
* `src/` - Main application source code:
  * `step1_load_data.py` to `step8_save_model.py` - Modular components handling data ingestion, cleaning, and non-linear feature engineering.
  * `step9_predict.py` - Asynchronous production inference manager handling model state restoration and runtime caching.
  * `api.py` - Core application backend (FastAPI router and PostgreSQL connector).
  * `index.html` - Lightweight frontend dashboard (Tailwind CSS) integrated with the API.
* `tests/` - Automated unit tests repository (`test_api.py`).
* `Dockerfile` - Configuration blueprint for the production container.
* `requirements_docker.txt` - Frozen production dependencies
* `run_model_selection.pu` - Hyperparameter tuning via GridSearchCV pipeline
---

IV. 🛠️ Local Deployment Guide

### a. - Option 1: Native Python Execution
1. Install the core production and inference dependencies:
   
   pip install -r requirements_docker.txt
    
2. Start the local Uvicorn development server:
    
    python src/api.py
    
3. Access the interactive API documentation (Swagger UI) at: http://127.0.0.1:8000/docs


### b. - Option 2: Isolated Containerization (Docker)
The system is fully containerized to ensure consistent behavior across different environments:

1. Build the production Docker image
docker build -t house-price-api .

2.  Run the containerized service locally
docker run -p 8000:8000 house-price-api

V. 🧪 Automated Testing
The project maintains a rigorous quality assurance loop for critical backend components. To execute the automated test runner locally, run:

    pytest

    Note: This suite triggers automatically inside the GitHub Actions virtual environment prior to every production deployment step.