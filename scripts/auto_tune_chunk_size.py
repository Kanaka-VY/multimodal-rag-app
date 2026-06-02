#!/usr/bin/env python3
"""
Automated Hyperparameter Optimization (HPO) for Chunk Size Tuning
Uses Optuna to find optimal chunk size and overlap parameters for RAG system.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from src.core.ingest import IngestPipeline
from src.core.retriever import MultimodalRetriever
from src.core.llm_handler import LLMHandler
from src.database.vector_store import VectorStore
from src.utils.config import get_model_config
from src.utils.monitoring import evaluate_faithfulness


def objective(trial: Any, test_queries: list[dict]) -> float:
    """
    Optuna objective function to minimize (negative F1 score).
    """
    # Suggest hyperparameters
    chunk_size = trial.suggest_categorical("chunk_size", [256, 512, 1024, 2048])
    chunk_overlap = trial.suggest_int("chunk_overlap", 32, 256, step=32)
    top_k = trial.suggest_int("top_k", 3, 10)
    score_threshold = trial.suggest_float("score_threshold", 0.5, 0.9, step=0.05)
    
    # Update configuration
    cfg = get_model_config()
    cfg["chunking"]["text_chunk_size"] = chunk_size
    cfg["chunking"]["text_chunk_overlap"] = chunk_overlap
    cfg["retrieval"]["top_k"] = top_k
    cfg["retrieval"]["score_threshold"] = score_threshold
    
    # Re-initialize components with new config
    retriever = MultimodalRetriever()
    llm_handler = LLMHandler()
    
    # Evaluate on test queries
    total_faithfulness = 0.0
    successful_evaluations = 0
    
    for query_data in test_queries:
        query = query_data["question"]
        expected_answer = query_data["answer"]
        
        try:
            # Retrieve contexts
            contexts = retriever.retrieve(query, top_k=top_k, modality=None)
            
            if not contexts:
                continue
            
            # Generate answer
            answer = llm_handler.generate(query, contexts)
            
            # Evaluate faithfulness
            faithfulness = evaluate_faithfulness(query, answer, contexts)
            
            if faithfulness is not None:
                total_faithfulness += faithfulness
                successful_evaluations += 1
                
        except Exception:
            continue
    
    # Calculate average faithfulness
    avg_faithfulness = total_faithfulness / successful_evaluations if successful_evaluations > 0 else 0.0
    
    # Return negative value for minimization
    return -avg_faithfulness


def run_auto_tuning(
    test_queries: list[dict],
    n_trials: int = 50,
    study_name: str = "rag_chunk_optimization"
) -> dict[str, Any]:
    """
    Run automated hyperparameter optimization using Optuna.
    """
    if not OPTUNA_AVAILABLE:
        print("❌ Optuna is not installed. Install with: pip install optuna")
        return {"status": "error", "message": "Optuna not available"}
    
    print("=" * 80)
    print("🔧 AUTOMATED HYPERPARAMETER OPTIMIZATION (HPO)")
    print("=" * 80)
    print(f"Test queries: {len(test_queries)}")
    print(f"Trials: {n_trials}")
    print(f"Study name: {study_name}")
    print()
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Optimize
    print("Starting optimization...")
    study.optimize(
        lambda trial: objective(trial, test_queries),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Get best parameters
    best_params = study.best_params
    best_value = -study.best_value
    
    print("\n" + "=" * 80)
    print("✅ OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Best Faithfulness Score: {best_value:.4f}")
    print(f"Best Parameters:")
    for param, value in best_params.items():
        print(f"  - {param}: {value}")
    
    # Save results
    results = {
        "status": "success",
        "best_faithfulness": best_value,
        "best_params": best_params,
        "n_trials": n_trials,
        "study_name": study_name
    }
    
    results_dir = ROOT / "results" / "optimization"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / f"{study_name}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_dir / f'{study_name}.json'}")
    
    return results


def update_config_with_best_params(best_params: dict[str, Any]) -> None:
    """
    Update model_config.yaml with optimal parameters.
    """
    import yaml
    
    config_path = ROOT / "configs" / "model_config.yaml"
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Update chunking parameters
    cfg["chunking"]["text_chunk_size"] = best_params.get("chunk_size", 512)
    cfg["chunking"]["text_chunk_overlap"] = best_params.get("chunk_overlap", 64)
    
    # Update retrieval parameters
    cfg["retrieval"]["top_k"] = best_params.get("top_k", 5)
    cfg["retrieval"]["score_threshold"] = best_params.get("score_threshold", 0.7)
    
    # Write back
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    
    print(f"✅ Updated {config_path} with optimal parameters")


def load_test_queries(test_data_path: Path) -> list[dict]:
    """
    Load test queries from JSON file.
    """
    if not test_data_path.exists():
        print(f"⚠️  Test data file not found: {test_data_path}")
        print("Creating sample test queries...")
        
        # Create sample test queries
        sample_queries = [
            {
                "question": "What is the main topic of the documents?",
                "answer": "The documents cover various topics related to enterprise operations."
            },
            {
                "question": "Summarize the key findings.",
                "answer": "The key findings include performance metrics and optimization strategies."
            }
        ]
        
        test_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(test_data_path, "w") as f:
            json.dump(sample_queries, f, indent=2)
        
        return sample_queries
    
    with open(test_data_path, "r") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated Hyperparameter Optimization for RAG Chunk Size",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/auto_tune_chunk_size.py
  python scripts/auto_tune_chunk_size.py --trials 100
  python scripts/auto_tune_chunk_size.py --test-data data/test_queries.json --apply
        """
    )
    
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of optimization trials (default: 50)"
    )
    
    parser.add_argument(
        "--test-data",
        type=Path,
        default=ROOT / "data" / "test_queries.json",
        help="Path to test queries JSON file"
    )
    
    parser.add_argument(
        "--study-name",
        type=str,
        default="rag_chunk_optimization",
        help="Optuna study name"
    )
    
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Automatically update model_config.yaml with best parameters"
    )
    
    args = parser.parse_args()
    
    # Load test queries
    test_queries = load_test_queries(args.test_data)
    
    if not test_queries:
        print("❌ No test queries available. Cannot run optimization.")
        sys.exit(1)
    
    # Run optimization
    results = run_auto_tuning(
        test_queries=test_queries,
        n_trials=args.trials,
        study_name=args.study_name
    )
    
    # Apply best parameters if requested
    if args.apply and results.get("status") == "success":
        print("\n" + "=" * 80)
        print("🔄 APPLYING OPTIMAL PARAMETERS")
        print("=" * 80)
        update_config_with_best_params(results["best_params"])
        print("✅ Configuration updated successfully")
        print("⚠️  Please restart your application to apply the new configuration")


if __name__ == "__main__":
    main()
