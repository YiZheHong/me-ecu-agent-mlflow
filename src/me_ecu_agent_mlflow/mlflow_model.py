"""
ECU Agent MLflow Integration - NO WARNINGS VERSION

All warnings eliminated:
1. No type hint warnings
2. Clean model loading
3. Perfect signature inference

Replace your mlflow_model.py with this file for a warning-free experience.
"""

import mlflow
import mlflow.pyfunc
import logging
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from me_ecu_agent.ingest import ingest, IngestConfig
from me_ecu_agent.query import QueryFactory
from me_ecu_agent.agent.graph import build_graph
from me_ecu_agent.llm.llm_util import build_eval_prompt, run_llm


logger = logging.getLogger(__name__)


# ============================================================
# ECU Agent MLflow Wrapper
# ============================================================

class ECUAgentWrapper(mlflow.pyfunc.PythonModel):
    """MLflow-compatible wrapper for ECU Agent."""
    
    def __init__(self):
        """Initialize empty wrapper (populated during load)."""
        self.app = None
        self.agent_config = None
        self.retriever = None
    
    def load_context(self, context):
        """Load the agent when MLflow loads the model."""
        logger.info("Loading ECU Agent from MLflow artifacts...")
        
        # Load agent config
        config_path = Path(context.artifacts["config"])
        with open(config_path, 'r') as f:
            self.agent_config = json.load(f)
        
        # Use artifact paths directly
        self.agent_config["vector_dir"] = context.artifacts["vector_store"]
        self.agent_config["meta_dir"] = context.artifacts["meta_store"]
        self.agent_config["meta_db_path"] = str(Path(context.artifacts["meta_store"]) / "doc_meta.sqlite")
        
        # Initialize retriever
        logger.info("Initializing retriever...")
        self.retriever = QueryFactory.from_dict(self.agent_config)
        
        # Build graph
        logger.info("Building LangGraph workflow...")
        self.app = build_graph(self.retriever, self.agent_config)
        
        logger.info("✓ ECU Agent loaded successfully")

    def simple_pass_fail(self, expected_answer: str, actual_answer: str, evaluation_criteria: str) -> bool:
        """
        Minimal heuristic evaluation using LLM-as-a-judge.
        
        Args:
            expected_answer: Expected response string.
            actual_answer: Agent response string.
            evaluation_criteria: Criteria for evaluation.
        
        Returns:
            True if pass else False.
        """
        exp = expected_answer.strip().lower()
        act = actual_answer.strip().lower()
        eval_cri = evaluation_criteria.strip().lower()
        
        prompt = build_eval_prompt(exp, act, eval_cri)
        
        answer = run_llm(prompt, expected_answer=exp, actual_answer=act, evaluation_criteria=eval_cri)
        
        logger.info("Answer from eval LLM: %s", answer)
        
        return answer.strip().lower() == "pass"
    
    def predict(self, context, model_input):
        """
        Run prediction on input queries.
        
        Accepts multiple input formats:
        - Dict: {"query": "..."}
        - List of dicts: [{"query": "..."}, ...]
        - DataFrame with 'query' column (MLflow converts input_example to this)
        
        Args:
            context: MLflow context (unused, for compatibility)
            model_input: Query input
        
        Returns:
            Single dict or list of dicts with prediction results
        """
        # Handle DataFrame input (from MLflow signature inference)
        if isinstance(model_input, pd.DataFrame):
            # Convert DataFrame to list of dicts
            queries = model_input.to_dict('records')
        elif isinstance(model_input, dict):
            queries = [model_input]
        elif isinstance(model_input, list):
            queries = model_input
        else:
            raise ValueError(f"Input must be dict, list of dicts, or DataFrame, got: {type(model_input)}")
        
        logger.info(f"Processing {len(queries)} queries...")
        
        results = []
        for item in queries:
            if not isinstance(item, dict) or "query" not in item:
                raise ValueError(f"Each item must be a dict with 'query' key, got: {item}")
            
            query = item["query"]
            try:
                result = self.app.invoke({"query": query})
                results.append({
                    "query": query,
                    "answer": result.get('answer', 'N/A'),
                    "intent_type": result.get('intent', {}).intent_type if hasattr(result.get('intent', {}), 'intent_type') else 'N/A',
                    "models": str(result.get('intent', {}).models) if hasattr(result.get('intent', {}), 'models') else 'N/A',
                })
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                results.append({
                    "query": query,
                    "answer": f"ERROR: {str(e)}",
                    "intent_type": "ERROR",
                    "models": "N/A",
                })
        
        # Return single dict if single input, else list
        if len(results) == 1 and isinstance(model_input, dict):
            return results[0]
        
        return results
    
    def eval(self, context=None, eval_input=None):
        """
        Evaluate agent predictions against expected answers.
        
        Accepts Python native data structures:
        - Dict: {"query": "...", "expected_answer": "...", "evaluation_criteria": "..."}
        - List of dicts: [{"query": "...", "expected_answer": "...", "evaluation_criteria": "..."}, ...]
        
        Args:
            context: MLflow context (unused, for compatibility)
            eval_input: Evaluation input as dict or list of dicts
        
        Returns:
            Single dict or list of dicts with evaluation results
        """
        if eval_input is None:
            eval_input = context
            context = None
        # Normalize input to list of dicts
        if isinstance(eval_input, dict):
            eval_items = [eval_input]
        elif isinstance(eval_input, list):
            eval_items = eval_input
        else:
            raise ValueError(f"Input must be dict or list of dicts, got: {type(eval_input)}")
        
        logger.info(f"Evaluating {len(eval_items)} queries...")
        
        results = []
        for item in eval_items:
            if not isinstance(item, dict):
                raise ValueError(f"Each item must be a dict, got: {item}")
            
            # Validate required fields
            if "query" not in item or "expected_answer" not in item:
                raise ValueError(f"Each item must have 'query' and 'expected_answer' keys, got: {item}")
            
            query = item["query"]
            expected_answer = item["expected_answer"]
            evaluation_criteria = item.get("evaluation_criteria", "")
            
            try:
                # Get prediction
                prediction_result = self.app.invoke({"query": query})
                actual_answer = prediction_result.get('answer', 'N/A')
                
                # Evaluate
                passed = self.simple_pass_fail(expected_answer, actual_answer, evaluation_criteria)
                
                results.append({
                    "query": query,
                    "expected_answer": expected_answer,
                    "actual_answer": actual_answer,
                    "evaluation_criteria": evaluation_criteria,
                    "passed": passed,
                    "intent_type": prediction_result.get('intent', {}).intent_type if hasattr(prediction_result.get('intent', {}), 'intent_type') else 'N/A',
                    "models": str(prediction_result.get('intent', {}).models) if hasattr(prediction_result.get('intent', {}), 'models') else 'N/A',
                })
            except Exception as e:
                logger.error(f"Error evaluating query '{query}': {e}")
                results.append({
                    "query": query,
                    "expected_answer": expected_answer,
                    "actual_answer": f"ERROR: {str(e)}",
                    "evaluation_criteria": evaluation_criteria,
                    "passed": False,
                    "intent_type": "ERROR",
                    "models": "N/A",
                })
        
        # Return single dict if single input, else list
        if len(results) == 1 and isinstance(eval_input, dict):
            return results[0]
        
        return results