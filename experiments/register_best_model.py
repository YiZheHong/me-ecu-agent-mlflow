"""
Save and Register Best Model to MLflow Model Registry

This script takes the best run from your experiments and:
1. Recreates the agent with the best parameters
2. Saves it as an MLflow model with proper signature
3. Registers it to the Model Registry for deployment

Usage:
    # Automatic mode: Find best from CSV
    python register_best_model.py --csv experiment_summary_20251225_232515.csv
    
    # Manual mode: Specify run ID
    python register_best_model.py --run-id 50d0c40b328143b4b3b611492db445b4
    
    # Custom model name
    python register_best_model.py --csv experiment_summary.csv --model-name "ECUAgent-Production"
"""

import argparse
import logging
import json
import time
import mlflow
import mlflow.pyfunc
import pandas as pd
import tempfile
from pathlib import Path
from typing import Dict

from me_ecu_agent.ingest import ingest, IngestConfig
from me_ecu_agent.query import QueryFactory
from me_ecu_agent.agent.graph import build_graph
from me_ecu_agent_mlflow.mlflow_model import ECUAgentWrapper
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# ============================================================
# Configuration
# ============================================================
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "ECU-Agent-Optimization"
# Create experiments directory structure if needed
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)
(EXPERIMENTS_DIR / "mlartifacts").mkdir(exist_ok=True)
(EXPERIMENTS_DIR / "rag").mkdir(exist_ok=True)
(EXPERIMENTS_DIR / "meta").mkdir(exist_ok=True)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================
# Helper Functions
# ============================================================

def get_best_run_from_csv(csv_path: str) -> Dict:
    """Find the best run from experiment summary CSV."""
    logger.info(f"Reading experiment results from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Filter out failed runs
    df = df[df['run_id'] != 'FAILED']
    
    if len(df) == 0:
        raise ValueError("No successful runs found in CSV")
    
    # Sort by pass_rate (descending), then by avg_time_per_query (ascending)
    df = df.sort_values(['pass_rate', 'avg_time_per_query'], ascending=[False, True])
    
    best_run = df.iloc[0]
    
    logger.info("="*80)
    logger.info("BEST RUN IDENTIFIED")
    logger.info("="*80)
    logger.info(f"Run ID: {best_run['run_id']}")
    logger.info(f"Run Name: {best_run['run_name']}")
    logger.info(f"Pass Rate: {best_run['pass_rate']:.2%}")
    logger.info(f"Avg Time: {best_run['avg_time_per_query']:.2f}s")
    logger.info(f"\nParameters:")
    logger.info(f"  chunk_size: {best_run['chunk_size']}")
    logger.info(f"  chunk_overlap: {best_run['chunk_overlap']}")
    logger.info(f"  default_top_k: {best_run['default_top_k']}")
    logger.info(f"  generic_top_k: {best_run['generic_top_k']}")
    logger.info("="*80)
    
    return {
        'run_id': best_run['run_id'],
        'run_name': best_run['run_name'],
        'chunk_size': int(best_run['chunk_size']),
        'chunk_overlap': int(best_run['chunk_overlap']),
        'default_top_k': int(best_run['default_top_k']),
        'generic_top_k': int(best_run['generic_top_k']),
        'pass_rate': float(best_run['pass_rate']),
    }


def get_params_from_run_id(run_id: str) -> Dict:
    """Get parameters from MLflow run."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    
    run = client.get_run(run_id)
    params = run.data.params
    metrics = run.data.metrics
    
    logger.info("="*80)
    logger.info("PARAMETERS FROM RUN")
    logger.info("="*80)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Run Name: {run.info.run_name}")
    logger.info(f"\nParameters:")
    logger.info(f"  chunk_size: {params.get('chunk_size')}")
    logger.info(f"  chunk_overlap: {params.get('chunk_overlap')}")
    logger.info(f"  default_top_k: {params.get('default_top_k')}")
    logger.info(f"  generic_top_k: {params.get('generic_top_k')}")
    logger.info(f"\nMetrics:")
    logger.info(f"  pass_rate: {metrics.get('pass_rate', 0):.2%}")
    logger.info("="*80)
    
    return {
        'run_id': run_id,
        'run_name': run.info.run_name,
        'chunk_size': int(params.get('chunk_size')),
        'chunk_overlap': int(params.get('chunk_overlap')),
        'default_top_k': int(params.get('default_top_k')),
        'generic_top_k': int(params.get('generic_top_k')),
        'pass_rate': metrics.get('pass_rate', 0),
    }


def get_agent_config(project_root: Path, data_dir: Path, params: Dict) -> dict:
    """Create agent configuration from parameters."""
    agent_config = {
        # Paths
        "project_root": str(project_root),
        "vector_dir": str(project_root / "experiments" / "rag"),
        "meta_dir": str(project_root / "experiments" / "meta"),
        "data_dir": str(data_dir),
        
        # Variable parameters from best run
        "chunk_size": params['chunk_size'],
        "chunk_overlap": params['chunk_overlap'],
        "default_top_k": params['default_top_k'],
        "generic_top_k": params['generic_top_k'],
        
        # Fixed parameters
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "default_threshold_score": 1.5,
        "retrieval_buffer_k": 40,
        "compare_top_k": 3,
        "single_model_top_k": 3,
        "spec_top_k": 2,
        "max_chars_per_model": 1000,
        "max_chars_per_model_specs": 1000,
    }
    
    return agent_config


def save_and_register_model(
    best_run: Dict,
    project_root: Path,
    model_name: str = "ECUAgent"
) -> str:
    """
    Recreate agent with best parameters, save and register to MLflow.
    
    Args:
        best_run: Dictionary with best run info and parameters
        project_root: Project root directory
        model_name: Name to register in Model Registry
    
    Returns:
        New run ID where the model is saved
    """
    logger.info("="*80)
    logger.info("RECREATING AGENT WITH BEST PARAMETERS")
    logger.info("="*80)
    
    # Setup
    data_dir = project_root / "data"
    agent_config = get_agent_config(project_root, data_dir, best_run)
    
    # Step 1: Rebuild vector store
    logger.info("Step 1: Rebuilding vector store...")
    logger.info(f"  chunk_size: {best_run['chunk_size']}")
    logger.info(f"  chunk_overlap: {best_run['chunk_overlap']}")
    
    start_time = time.time()
    vectorstore = ingest(
        rebuild=True,
        config=IngestConfig.from_dict(agent_config),
    )
    build_time = time.time() - start_time
    logger.info(f"  ✓ Vector store rebuilt in {build_time:.2f}s")
    
    # Step 2: Initialize retriever
    logger.info("Step 2: Initializing retriever...")
    retriever = QueryFactory.from_dict(agent_config)
    logger.info("  ✓ Retriever ready")
    
    # Step 3: Build LangGraph
    logger.info("Step 3: Building LangGraph workflow...")
    app = build_graph(retriever, agent_config)
    logger.info("  ✓ Graph ready")
    
    # Step 4: Create wrapper
    logger.info("Step 4: Creating MLflow wrapper...")
    wrapper = ECUAgentWrapper()
    wrapper.app = app
    # Convert all Path objects to strings to avoid WindowsPath issues in Linux containers
    wrapper.agent_config = {k: str(v) if isinstance(v, Path) else v 
                           for k, v in agent_config.items()}
    wrapper.retriever = retriever
    logger.info("  ✓ Wrapper ready")
    
    # Step 5: Prepare artifacts
    logger.info("Step 5: Preparing artifacts...")
    temp_dir = tempfile.gettempdir()
    config_path = Path(temp_dir) / f"agent_config_{int(time.time())}.json"
    
    with open(config_path, 'w') as f:
        json_config = {
            k: str(v) if isinstance(v, Path) else v 
            for k, v in agent_config.items()
            if k not in ['vector_dir', 'meta_dir', 'project_root', 'data_dir']
        }
        json.dump(json_config, f, indent=2)
    
    artifacts = {
        "config": str(config_path),
        "vector_store": agent_config["vector_dir"],
        "meta_store": agent_config["meta_dir"],
    }
    logger.info("  ✓ Artifacts ready")
    
    # Step 6: Create input example for signature inference
    logger.info("Step 6: Creating input example...")
    input_example = {"query": "What is the operating temperature range for ECU-700?"}
    logger.info("  ✓ Input example ready")
    
    # Step 7: Save to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Enable LangChain autolog for tracing
    mlflow.langchain.autolog()
    
    run_name = f"PRODUCTION_{best_run['run_name']}"
    
    with mlflow.start_run(run_name=run_name):
        logger.info("Step 7: Logging model to MLflow...")
        
        # Log parameters
        mlflow.log_params({
            "chunk_size": best_run['chunk_size'],
            "chunk_overlap": best_run['chunk_overlap'],
            "default_top_k": best_run['default_top_k'],
            "generic_top_k": best_run['generic_top_k'],
        })
        mlflow.log_param("source_run_id", best_run['run_id'])
        mlflow.log_param("source_run_name", best_run['run_name'])
        
        # Log metrics from original run
        mlflow.log_metric("source_pass_rate", best_run['pass_rate'])
        mlflow.log_metric("vector_store_build_time", build_time)
        
        # Log model with input example for signature inference
        mlflow.pyfunc.log_model(
            name="model",
            python_model=wrapper,
            artifacts=artifacts,
            input_example=input_example,  # Important for signature inference
            conda_env={
                'channels': ['defaults', 'conda-forge'],
                'dependencies': [
                    'python=3.10',
                    'pip',
                    {'pip': [
                        'mlflow',
                        'langchain',
                        'langchain-openai',
                        'langgraph',
                        'faiss-cpu',
                        'sentence-transformers',
                    ]}
                ]
            }
        )
        
        model_uri = mlflow.get_artifact_uri("model")
        new_run_id = mlflow.active_run().info.run_id
        
        logger.info("  ✓ Model logged to MLflow")
        logger.info(f"  Model URI: {model_uri}")
        logger.info(f"  New Run ID: {new_run_id}")
    
    # Step 8: Register model to Model Registry
    logger.info("Step 8: Registering model to Model Registry...")
    try:
        model_version = mlflow.register_model(
            model_uri=f"runs:/{new_run_id}/model",
            name=model_name
        )
        logger.info(f"  ✓ Model registered as '{model_name}' version {model_version.version}")
        
        # Add description to model version
        client = mlflow.tracking.MlflowClient()
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=f"Best model from experiment run {best_run['run_id']} with {best_run['pass_rate']:.1%} pass rate"
        )
        
    except Exception as e:
        logger.warning(f"  Could not register model: {e}")
        logger.info(f"  You can manually register using: mlflow.register_model('runs:/{new_run_id}/model', '{model_name}')")
    
    logger.info("="*80)
    logger.info("✓ MODEL SAVED AND REGISTERED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"New Run ID: {new_run_id}")
    logger.info(f"Model Name: {model_name}")
    logger.info(f"Source Run: {best_run['run_id']}")
    logger.info(f"Pass Rate: {best_run['pass_rate']:.1%}")
    logger.info(f"\n📦 To load this model:")
    logger.info(f"  # By run ID:")
    logger.info(f"  model = mlflow.pyfunc.load_model('runs:/{new_run_id}/model')")
    logger.info(f"  \n  # By registered name (recommended):")
    logger.info(f"  model = mlflow.pyfunc.load_model('models:/{model_name}/latest')")
    logger.info(f"  \n  # Use the model:")
    logger.info(f"  result = model.predict({{'query': 'your question'}})")
    logger.info("="*80)
    
    return new_run_id


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Save and register best model to MLflow Model Registry'
    )
    parser.add_argument(
        '--csv',
        type=str,
        help='Path to experiment summary CSV (will auto-select best run)'
    )
    parser.add_argument(
        '--run-id',
        type=str,
        help='Specific run ID to save as model'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='ECUAgent',
        help='Name to register the model in Model Registry (default: ECUAgent)'
    )
    
    args = parser.parse_args()
    
    if not args.csv and not args.run_id:
        parser.error("Must specify either --csv or --run-id")
    
    # Get project root
    project_root = PROJECT_ROOT
    
    # Get best run parameters
    if args.csv:
        csv_path = project_root / args.csv
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            return
        best_run = get_best_run_from_csv(str(csv_path))
    else:
        best_run = get_params_from_run_id(args.run_id)
    
    # Confirm before proceeding
    print("\n" + "="*80)
    print("CONFIRMATION")
    print("="*80)
    print(f"Will create and register model with:")
    print(f"  Model Name: {args.model_name}")
    print(f"  Source Run: {best_run['run_id']}")
    print(f"  Pass Rate: {best_run['pass_rate']:.1%}")
    print("="*80)
    response = input("\nProceed? (yes/no): ")
    
    if response.lower() != 'yes':
        logger.info("Cancelled by user")
        return
    
    # Save and register model
    new_run_id = save_and_register_model(
        best_run=best_run,
        project_root=project_root,
        model_name=args.model_name
    )
    
    logger.info(f"\n✅ Done! Model registered and ready for deployment.")


if __name__ == "__main__":
    main()