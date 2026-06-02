#!/usr/bin/env python3
"""
Check Regression Test Thresholds
Validates regression test results against minimum thresholds.
"""

import argparse
import json
import sys
from pathlib import Path


def check_thresholds(
    results_path: Path,
    min_faithfulness: float,
    min_context_precision: float,
    min_answer_relevance: float,
    min_context_recall: float,
    min_mrr: float
) -> bool:
    """
    Check if regression test results meet minimum thresholds.
    
    Returns:
        True if all thresholds met, False otherwise
    """
    with open(results_path, "r") as f:
        results = json.load(f)
    
    metrics = results.get("metrics", {})
    
    faithfulness = metrics.get("faithfulness", 0.0)
    context_precision = metrics.get("context_precision", 0.0)
    answer_relevance = metrics.get("answer_relevance", 0.0)
    context_recall = metrics.get("context_recall", 0.0)
    mrr = metrics.get("mrr", 0.0)
    
    all_passed = True
    
    print("=" * 80)
    print("🔍 THRESHOLD VALIDATION")
    print("=" * 80)
    
    print(f"Faithfulness: {faithfulness:.3f} >= {min_faithfulness} {'✅' if faithfulness >= min_faithfulness else '❌'}")
    if faithfulness < min_faithfulness:
        all_passed = False
    
    print(f"Context Precision: {context_precision:.3f} >= {min_context_precision} {'✅' if context_precision >= min_context_precision else '❌'}")
    if context_precision < min_context_precision:
        all_passed = False
    
    print(f"Answer Relevance: {answer_relevance:.3f} >= {min_answer_relevance} {'✅' if answer_relevance >= min_answer_relevance else '❌'}")
    if answer_relevance < min_answer_relevance:
        all_passed = False
    
    print(f"Context Recall: {context_recall:.3f} >= {min_context_recall} {'✅' if context_recall >= min_context_recall else '❌'}")
    if context_recall < min_context_recall:
        all_passed = False
    
    print(f"MRR: {mrr:.3f} >= {min_mrr} {'✅' if mrr >= min_mrr else '❌'}")
    if mrr < min_mrr:
        all_passed = False
    
    print("=" * 80)
    
    if all_passed:
        print("✅ All thresholds met. Deployment approved.")
    else:
        print("❌ Some thresholds not met. Deployment blocked.")
    
    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check regression test results against thresholds"
    )
    
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to regression test results JSON"
    )
    
    parser.add_argument(
        "--min-faithfulness",
        type=float,
        default=0.80,
        help="Minimum faithfulness threshold"
    )
    
    parser.add_argument(
        "--min-context-precision",
        type=float,
        default=0.75,
        help="Minimum context precision threshold"
    )
    
    parser.add_argument(
        "--min-answer-relevance",
        type=float,
        default=0.70,
        help="Minimum answer relevance threshold"
    )
    
    parser.add_argument(
        "--min-context-recall",
        type=float,
        default=0.70,
        help="Minimum context recall threshold"
    )
    
    parser.add_argument(
        "--min-mrr",
        type=float,
        default=0.75,
        help="Minimum MRR threshold"
    )
    
    args = parser.parse_args()
    
    all_passed = check_thresholds(
        args.results,
        args.min_faithfulness,
        args.min_context_precision,
        args.min_answer_relevance,
        args.min_context_recall,
        args.min_mrr
    )
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
