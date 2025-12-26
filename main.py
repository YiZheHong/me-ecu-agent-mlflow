import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import pandas as pd
from pathlib import Path
from typing import List

# Import logic from your experiment and registration scripts
from experiments.run_experiment import run_experiment_grid, PARAM_GRID
from experiments.register_best_model import get_params_from_run_id, save_and_register_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ecu-agent-main")

app = FastAPI(title="ECU Agent Lifecycle Manager")

# CONSTANTS
MODEL_URI = "models:/ECUAgent/latest"
PROJECT_ROOT = Path(__file__).parent.resolve()

# Global model instance
agent_model = None

# --- Request Schemas ---
class PredictRequest(BaseModel):
    query: str
    session_id: str = "default"

class RegisterRequest(BaseModel):
    run_id: str
    model_name: str = "ECUAgent"

# --- Lifecycle Hooks ---
@app.on_event("startup")
async def startup_event():
    """
    Attempt to load the latest registered model on startup.
    """
    global agent_model
    try:
        agent_model = mlflow.pyfunc.load_model(MODEL_URI)
        logger.info(f"✓ Model loaded successfully from {MODEL_URI}")
    except Exception as e:
        logger.warning(f"⚠ Initial load failed: {e}. System waiting for /train and /register.")

# --- 1. Training Endpoint ---
@app.post("/train")
async def train_model():
    """
    Trigger the grid search experiment.
    Returns: List of Run IDs, Pass Rates, and Execution Times.
    """
    logger.info("Starting grid search optimization...")
    try:
        # Execute the grid search logic from run_experiment.py
        results_df = run_experiment_grid(
            param_grid=PARAM_GRID,
            test_csv_path="test-questions.csv",
            project_root=PROJECT_ROOT
        )
        
        # Simplify results for the API response
        summary = results_df[['run_id', 'pass_rate', 'total_eval_time']].to_dict(orient='records')
        return {
            "status": "success",
            "experiments_completed": len(summary),
            "results": summary
        }
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 2. Registration Endpoint ---
@app.post("/register")
async def register_specific_run(request: RegisterRequest):
    """
    Register a specific Run ID to the MLflow Model Registry.
    """
    logger.info(f"Registering Run ID: {request.run_id} as '{request.model_name}'")
    try:
        # Fetch parameters associated with the provided Run ID
        best_run_params = get_params_from_run_id(request.run_id)
        
        # Recreate and register the model
        new_run_id = save_and_register_model(
            best_run=best_run_params,
            project_root=PROJECT_ROOT,
            model_name=request.model_name
        )
        
        return {
            "status": "success", 
            "registered_version_run_id": new_run_id,
            "message": f"Model {request.model_name} is now updated to the latest version."
        }
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 3. Serving Endpoints ---
@app.post("/predict")
async def predict(request: PredictRequest):
    """
    Inference endpoint. Wraps query in a DataFrame to match MLflow expected schema.
    """
    global agent_model
    if agent_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train and register a model first.")

    try:
        # Wrap input string in a DataFrame to satisfy MLflow input signature
        input_df = pd.DataFrame([{"query": request.query}])
        prediction = agent_model.predict(input_df)
        return {"result": prediction}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/reload")
async def reload_model():
    """
    Manually refresh the model instance from the MLflow Registry.
    """
    global agent_model
    try:
        agent_model = mlflow.pyfunc.load_model(MODEL_URI)
        return {"status": "success", "message": "Model reloaded to latest version."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ready" if agent_model else "waiting_for_model"}