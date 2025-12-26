"""
FastAPI Service for ECU Agent Model

This service loads a registered MLflow model and provides REST API endpoints.
"""

import os
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration from environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-train:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "ECUAgent")
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")

# Global model instance
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model
    
    logger.info("="*80)
    logger.info("LOADING MODEL FROM MLFLOW")
    logger.info("="*80)
    logger.info(f"Tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"Model Name: {MODEL_NAME}")
    logger.info(f"Model Version: {MODEL_VERSION}")
    
    try:
        # Try multiple loading methods
        model_loaded = False
        
        # Method 1: Try loading from MLflow tracking server (with retry)
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
            logger.info(f"Attempting to load from tracking server: {model_uri}")
            model = mlflow.pyfunc.load_model(model_uri)
            model_loaded = True
            logger.info("✓ Model loaded from tracking server successfully!")
        except Exception as e:
            logger.warning(f"Could not load from tracking server: {e}")
            logger.info("Trying alternative method...")
        
        # Method 2: Load from local mlruns directory (fallback)
        if not model_loaded:
            try:
                import os
                import glob
                
                # Find the latest model in mlruns
                mlruns_path = "/app/mlruns"
                
                # Look for registered models
                models_path = os.path.join(mlruns_path, "models", MODEL_NAME)
                if os.path.exists(models_path):
                    # Find latest version
                    versions = sorted([d for d in os.listdir(models_path) if d.startswith("version-")])
                    if versions:
                        latest_version = versions[-1] if MODEL_VERSION == "latest" else f"version-{MODEL_VERSION}"
                        version_path = os.path.join(models_path, latest_version)
                        
                        # Find the model artifact
                        model_files = glob.glob(os.path.join(version_path, "artifacts", "model"))
                        if model_files:
                            model_uri = f"file://{model_files[0]}"
                            logger.info(f"Attempting to load from file system: {model_uri}")
                            model = mlflow.pyfunc.load_model(model_uri)
                            model_loaded = True
                            logger.info("✓ Model loaded from file system successfully!")
                
                if not model_loaded:
                    # Try to find any model artifact in mlruns
                    logger.info("Searching for model artifacts in mlruns...")
                    artifact_paths = glob.glob(f"{mlruns_path}/**/model", recursive=True)
                    if artifact_paths:
                        # Use the most recent one
                        latest_artifact = max(artifact_paths, key=os.path.getmtime)
                        model_uri = f"file://{latest_artifact}"
                        logger.info(f"Found model artifact: {model_uri}")
                        model = mlflow.pyfunc.load_model(model_uri)
                        model_loaded = True
                        logger.info("✓ Model loaded from artifact path successfully!")
                        
            except Exception as e:
                logger.warning(f"Could not load from file system: {e}")
        
        if not model_loaded:
            raise Exception("Could not load model from any source")
            
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Service will start but /query endpoint will fail")
        logger.error("Make sure you have registered a model in MLflow")
        import traceback
        logger.error(traceback.format_exc())
    
    yield
    
    # Cleanup
    logger.info("Shutting down service...")


# Create FastAPI app
app = FastAPI(
    title="ECU Agent API",
    description="API service for ECU documentation query agent",
    version="1.0.0",
    lifespan=lifespan
)


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the operating temperature range for ECU-700?"
            }
        }


class QueryResponse(BaseModel):
    answer: str
    sources: Optional[list] = None
    metadata: Optional[Dict[str, Any]] = None


# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_loaded": model is not None
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "tracking_uri": MLFLOW_TRACKING_URI
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Query the ECU Agent model.
    
    Args:
        request: QueryRequest with query text
    
    Returns:
        QueryResponse with answer and metadata
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        logger.info(f"Received query: {request.query}")
        
        # Call model
        result = model.predict({"query": request.query})
        
        # Extract response
        # The exact format depends on your ECUAgentWrapper.predict() implementation
        if isinstance(result, dict):
            answer = result.get("answer", str(result))
            sources = result.get("sources", [])
            metadata = result.get("metadata", {})
        else:
            answer = str(result)
            sources = []
            metadata = {}
        
        logger.info(f"Response generated (length: {len(answer)} chars)")
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/model-info")
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get model metadata from MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        
        # Get latest version info
        model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        
        if not model_versions:
            return {"error": "No model versions found"}
        
        latest = model_versions[0]
        
        return {
            "model_name": MODEL_NAME,
            "version": latest.version,
            "status": latest.status,
            "description": latest.description,
            "run_id": latest.run_id,
            "source": latest.source
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)