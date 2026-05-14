# Dockerfile scheme

# 1. Use an official lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file and install dependencies
# We use --no-cache-dir to keep the image small and clean
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# 4. Copy ONLY the necessary files for inference
# Copy the brain (model)
COPY models/house_price_hybrid_champion.joblib models/

# Now we surgically pick only the needed scripts from src/
COPY src/api.py src/
COPY src/step2_feature_engineering.py src/
COPY src/step9_predict.py src/

# 5. Expose the port that FastAPI will run on
EXPOSE 8000

# 6. Set the startup command
# We use 'src.api:app' because api.py is inside the src folder
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]