"""Configuration for RAGAS Evaluation.

Defines thresholds and evaluator judge settings for:
- Faithfulness
- Answer Relevance
- Context Precision
- Context Recall
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation parameters and threshold boundaries."""

    # Fallback default session ID if none is supplied in request
    default_session_id: str = "8350581e-f557-494c-b918-6dbed5356eba"

    # Evaluation LLM Judge settings
    judge_temperature: float = 0.0

    # Metric acceptance thresholds (0.0 to 1.0)
    min_faithfulness: float = 0.85
    min_answer_relevancy: float = 0.80
    min_context_precision: float = 0.75
    min_context_recall: float = 0.75


def get_eval_config() -> EvaluationConfig:
    return EvaluationConfig()
