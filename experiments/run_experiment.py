"""
MLflow Experiment Runner for ECU Agent Parameter Optimization

This script runs grid search experiments using MLflow to track different
parameter combinations for the ECU Agent. Each experiment:
1. Creates a new agent with specific parameters
2. Rebuilds the vector store with those parameters
3. Logs the model to MLflow
4. Evaluates using .eval() method
5. Records metrics and parameters

Usage:
    python mlflow_experiment_runner.py
"""

import logging
import json
import time
import csv
import mlflow
import mlflow.pyfunc
import pandas as pd
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from itertools import product
from datetime import datetime

from me_ecu_agent.ingest import ingest, IngestConfig
from me_ecu_agent.query import QueryFactory
from me_ecu_agent.agent.graph import build_graph
from me_ecu_agent_mlflow.mlflow_model import ECUAgentWrapper

# Enable LangChain/LangGraph auto-tracing
mlflow.langchain.autolog()


# ============================================================
# CONFIGURATION
# ============================================================
LOG_LEVEL = "INFO"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")
EXPERIMENT_NAME = "ECU-Agent-Optimization"
TEST_QUESTIONS_CSV = "test-questions.csv"
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Parameter grid for experiments
PARAM_GRID = {
    "chunk_size": [1500],
    "chunk_overlap": [100],
    "default_top_k": [5],
    "generic_top_k": [5],
}

# Fixed parameters (not varied in experiments)
FIXED_PARAMS = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "default_threshold_score": 1.5,
    "retrieval_buffer_k": 40,
    "compare_top_k": 3,
    "single_model_top_k": 3,
    "spec_top_k": 2,
    "max_chars_per_model": 1000,
    "max_chars_per_model_specs": 1000,
}

# ============================================================
# Logger Setup
# ============================================================
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================
# Helper Functions
# ============================================================

def load_test_questions(csv_path: Path) -> List[Dict[str, str]]:
    """
    Load test questions from CSV file.
    
    Expected CSV format:
    Question, Expected_Answer, Evaluation_Criteria
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        List of test cases with query, expected_answer, evaluation_criteria
    """
    logger.info(f"Loading test questions from: {csv_path}")
    
    if not csv_path.exists():
        logger.error(f"Test CSV not found: {csv_path}")
        raise FileNotFoundError(f"Test CSV not found: {csv_path}")
    
    test_cases = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Check if required columns exist
        if 'Question' not in reader.fieldnames:
            logger.error("CSV missing 'Question' column")
            raise ValueError("CSV must have 'Question' column")
        
        for row in reader:
            test_cases.append({
                'query': row.get('Question', '').strip(),
                'expected_answer': row.get('Expected_Answer', '').strip(),
                'evaluation_criteria': row.get('Evaluation_Criteria', '').strip()
            })
    
    logger.info(f"Loaded {len(test_cases)} test questions")
    return test_cases


def get_agent_config(project_root: Path, data_dir: Path, params: Dict[str, Any]) -> dict:
    """
    Create agent configuration from parameters.
    
    Args:
        project_root: Root directory of project
        data_dir: Data directory
        params: Parameter dictionary
    
    Returns:
        Complete agent configuration
    """
    agent_config = {
        # Paths
        "project_root": str(project_root),
        "vector_dir": str(project_root / "experiments" / "rag"),
        "meta_dir": str(project_root / "experiments" / "meta"),
        "data_dir": str(data_dir),
        
        # Merge variable and fixed parameters
        **params,
        **FIXED_PARAMS,
    }
    
    return agent_config


def initialize_agent_and_artifacts(agent_config: dict) -> tuple:
    """
    Initialize agent with given config and prepare MLflow artifacts.
    
    This function:
    1. Rebuilds vector store with new parameters
    2. Initializes retriever
    3. Builds LangGraph
    4. Prepares artifacts for MLflow logging
    
    Args:
        agent_config: Agent configuration
    
    Returns:
        (app, artifacts_dict) tuple
    """
    logger.info("="*80)
    logger.info("INITIALIZING AGENT FOR EXPERIMENT")
    logger.info("="*80)
    
    # Step 1: Rebuild vector store (always rebuild for new parameters)
    logger.info("Step 1: Rebuilding vector store with new parameters...")
    logger.info(f"  chunk_size: {agent_config['chunk_size']}")
    logger.info(f"  chunk_overlap: {agent_config['chunk_overlap']}")
    
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
    
    # Step 3: Build graph
    logger.info("Step 3: Building LangGraph workflow...")
    app = build_graph(retriever, agent_config)
    logger.info("  ✓ Graph ready")
    
    # Step 4: Prepare artifacts
    logger.info("Step 4: Preparing MLflow artifacts...")
    
    # Save config to temp file (cross-platform)
    temp_dir = tempfile.gettempdir()
    config_path = Path(temp_dir) / f"agent_config_{int(time.time())}.json"
    
    with open(config_path, 'w') as f:
        # Convert Path objects to strings for JSON serialization
        json_config = {k: str(v) if isinstance(v, Path) else v 
                      for k, v in agent_config.items()}
        json.dump(json_config, f, indent=2)
    
    artifacts = {
        "config": str(config_path),
        "vector_store": agent_config["vector_dir"],
        "meta_store": agent_config["meta_dir"],
    }
    
    logger.info("  ✓ Artifacts ready")
    logger.info("="*80)
    
    return app, artifacts, build_time


def create_mlflow_wrapper(app, agent_config: dict) -> ECUAgentWrapper:
    """
    Create ECUAgentWrapper instance with initialized components.
    
    Args:
        app: Compiled LangGraph application
        agent_config: Agent configuration
    
    Returns:
        Initialized ECUAgentWrapper
    """
    wrapper = ECUAgentWrapper()
    wrapper.app = app
    wrapper.agent_config = agent_config
    
    # Also need to set retriever for consistency
    wrapper.retriever = QueryFactory.from_dict(agent_config)
    
    return wrapper


def calculate_metrics(eval_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate evaluation metrics from eval results.
    
    Args:
        eval_results: List of evaluation results from .eval()
    
    Returns:
        Dictionary of metrics
    """
    total = len(eval_results)
    if total == 0:
        return {
            "pass_rate": 0.0,
            "total_queries": 0,
            "passed": 0,
            "failed": 0,
        }
    
    passed = sum(1 for r in eval_results if r.get('passed', False))
    failed = total - passed
    
    # Calculate average response time if available
    # (This would require modification to ECUAgentWrapper.eval to track time)
    
    metrics = {
        "pass_rate": passed / total,
        "total_queries": total,
        "passed": passed,
        "failed": failed,
    }
    
    return metrics


def run_single_experiment(
    params: Dict[str, Any],
    test_cases: List[Dict[str, str]],
    project_root: Path,
    data_dir: Path,
) -> Dict[str, Any]:
    """
    Run a single experiment with given parameters.
    
    Args:
        params: Parameter dictionary for this experiment
        test_cases: List of test cases to evaluate
        project_root: Project root directory
        data_dir: Data directory
    
    Returns:
        Dictionary with experiment results
    """
    run_name = "_".join([f"{k}={v}" for k, v in params.items()])
    
    logger.info("="*80)
    logger.info(f"STARTING EXPERIMENT: {run_name}")
    logger.info("="*80)
    
    with mlflow.start_run(run_name=run_name):
        # 1. Log parameters
        logger.info("Logging parameters to MLflow...")
        mlflow.log_params(params)
        mlflow.log_params({f"fixed_{k}": v for k, v in FIXED_PARAMS.items()})
        
        # 2. Create agent config
        agent_config = get_agent_config(project_root, data_dir, params)
        
        # 3. Initialize agent and prepare artifacts
        app, artifacts, build_time = initialize_agent_and_artifacts(agent_config)
        mlflow.log_metric("vector_store_build_time", build_time)
        
        # 4. Create wrapper
        wrapper = create_mlflow_wrapper(app, agent_config)
        
        # 5. Run evaluation using .eval() directly on wrapper
        logger.info(f"Running evaluation on {len(test_cases)} test cases...")
        eval_start = time.time()
        
        eval_results = wrapper.eval(
            eval_input=test_cases
        )
        
        eval_time = time.time() - eval_start
        logger.info(f"Evaluation completed in {eval_time:.2f}s")
        
        # 6. Calculate and log metrics
        metrics = calculate_metrics(eval_results)
        metrics['total_eval_time'] = eval_time
        metrics['avg_time_per_query'] = eval_time / len(test_cases) if test_cases else 0
        
        logger.info("Logging metrics to MLflow...")
        mlflow.log_metrics(metrics)
        
        # 9. Log detailed results as artifact
        results_df = pd.DataFrame(eval_results)
        temp_dir = tempfile.gettempdir()
        results_path = os.path.join(temp_dir, f"eval_results_{int(time.time())}.csv")
        results_df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path, "evaluation_results")
        
        # Get run ID
        run_id = mlflow.active_run().info.run_id
        
        logger.info("="*80)
        logger.info(f"EXPERIMENT COMPLETED: {run_name}")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Pass Rate: {metrics['pass_rate']:.2%}")
        logger.info("="*80)
        
        return {
            "run_id": run_id,
            "run_name": run_name,
            **params,
            **metrics,
        }


def run_experiment_grid(
    param_grid: Dict[str, List[Any]],
    test_csv_path: str,
    project_root: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run grid search experiments over parameter combinations.
    
    Args:
        param_grid: Dictionary of parameter names to lists of values
        test_csv_path: Path to test questions CSV
        project_root: Project root directory (defaults to current dir)
    
    Returns:
        DataFrame with all experiment results
    """
    # Setup
    if project_root is None:
        project_root = PROJECT_ROOT
    
    data_dir = project_root / "data"
    test_csv = data_dir / test_csv_path
    
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    logger.info("="*80)
    logger.info("STARTING GRID SEARCH EXPERIMENTS")
    logger.info("="*80)
    logger.info(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"Experiment Name: {EXPERIMENT_NAME}")
    logger.info(f"Project Root: {project_root}")
    logger.info(f"Test CSV: {test_csv}")
    logger.info(f"Parameter Grid: {param_grid}")
    logger.info("="*80)
    
    # Load test cases
    test_cases = load_test_questions(test_csv)
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    logger.info(f"Total experiments to run: {len(param_combinations)}")
    
    # Run experiments
    all_results = []
    
    for i, param_tuple in enumerate(param_combinations, 1):
        param_dict = dict(zip(param_names, param_tuple))
        
        logger.info(f"\n{'='*80}")
        logger.info(f"EXPERIMENT {i}/{len(param_combinations)}")
        logger.info(f"{'='*80}\n")
        
        try:
            result = run_single_experiment(
                params=param_dict,
                test_cases=test_cases,
                project_root=project_root,
                data_dir=data_dir,
            )
            all_results.append(result)
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            all_results.append({
                "run_id": "FAILED",
                "run_name": "_".join([f"{k}={v}" for k, v in param_dict.items()]),
                **param_dict,
                "pass_rate": 0.0,
                "error": str(e),
            })
    
    # Create summary DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save summary
    summary_path = project_root / "experiments" / "summary" / f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(summary_path, index=False)
    
    logger.info("="*80)
    logger.info("ALL EXPERIMENTS COMPLETED")
    logger.info("="*80)
    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"\nTop 5 configurations by pass rate:")
    logger.info(results_df.nlargest(5, 'pass_rate')[['run_name', 'pass_rate']])
    logger.info("="*80)
    
    return results_df


# ============================================================
# Main
# ============================================================

def main():
    """Main function to run grid search experiments."""
    results = run_experiment_grid(
        param_grid=PARAM_GRID,
        test_csv_path=TEST_QUESTIONS_CSV,
    )
    
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print(results.to_string())
    print("="*80)


if __name__ == "__main__":
    main()