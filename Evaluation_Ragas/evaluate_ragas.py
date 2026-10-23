"""Core RAGAS Evaluator Engine.

Evaluates RAG pipeline outputs using the 4 core RAGAS metrics:
1. Faithfulness: Factual consistency of generated response against retrieved context.
2. Answer Relevance (answer_relevancy): Pertinence of the generated response to the query.
3. Context Precision (context_precision): Signal-to-noise ratio and rank quality of retrieval.
4. Context Recall (context_recall): Extent to which retrieved context covers the ground truth.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from app.embeddings import get_embeddings
from app.llm import get_llm
from Evaluation_Ragas.config import EvaluationConfig, get_eval_config

logger = logging.getLogger("ragas_evaluator")


@dataclass
class MetricThresholdResult:
    name: str
    score: float
    threshold: float
    passed: bool
    description: str


@dataclass
class EvaluationReportData:
    timestamp: str
    total_samples: int
    mean_faithfulness: float
    mean_answer_relevancy: float
    mean_context_precision: float
    mean_context_recall: float
    overall_score: float
    threshold_results: list[dict[str, Any]]
    samples_table: list[dict[str, Any]]


class RagasEvaluator:
    """Manages LLM judge, embeddings, and metric execution for Ragas."""

    def __init__(self, config: EvaluationConfig | None = None):
        self.config = config or get_eval_config()

        # Initialize Evaluator LLM (temperature=0 for deterministic evaluation)
        raw_llm = get_llm(temperature=self.config.judge_temperature)
        self.evaluator_llm = LangchainLLMWrapper(raw_llm)

        # Initialize Evaluator Embeddings
        raw_embeddings = get_embeddings()
        self.evaluator_embeddings = LangchainEmbeddingsWrapper(raw_embeddings)

        # Configure metrics with LLM judge and embeddings
        self.faithfulness_metric = Faithfulness(llm=self.evaluator_llm)
        self.answer_relevancy_metric = AnswerRelevancy(
            llm=self.evaluator_llm, embeddings=self.evaluator_embeddings
        )
        self.context_precision_metric = ContextPrecision(llm=self.evaluator_llm)
        self.context_recall_metric = ContextRecall(llm=self.evaluator_llm)

        self.metrics = [
            self.faithfulness_metric,
            self.answer_relevancy_metric,
            self.context_precision_metric,
            self.context_recall_metric,
        ]

    def evaluate_dataset(
        self,
        dataset: EvaluationDataset,
        show_progress: bool = True,
    ) -> EvaluationReportData:
        """Run evaluation across all 4 metrics on the given EvaluationDataset."""
        logger.info(f"Starting RAGAS evaluation on {len(dataset)} samples...")

        # Run RAGAS evaluate
        results = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.evaluator_llm,
            embeddings=self.evaluator_embeddings,
            show_progress=show_progress,
        )

        df = results.to_pandas()

        def clean_mean(col_name: str) -> float:
            if col_name not in df or df[col_name].empty:
                return 0.0
            val = df[col_name].mean()
            if pd.isna(val) or val is None:
                return 0.0
            return float(val)

        m_faith = clean_mean("faithfulness")
        m_rel = clean_mean("answer_relevancy")
        m_prec = clean_mean("context_precision")
        m_rec = clean_mean("context_recall")

        overall = (m_faith + m_rel + m_prec + m_rec) / 4.0

        thresholds = [
            MetricThresholdResult(
                name="Faithfulness",
                score=round(m_faith, 4),
                threshold=self.config.min_faithfulness,
                passed=m_faith >= self.config.min_faithfulness,
                description="Checks if generated claims are directly supported by retrieved context.",
            ),
            MetricThresholdResult(
                name="Answer Relevance",
                score=round(m_rel, 4),
                threshold=self.config.min_answer_relevancy,
                passed=m_rel >= self.config.min_answer_relevancy,
                description="Checks if response directly answers the user prompt without irrelevant deviation.",
            ),
            MetricThresholdResult(
                name="Context Precision",
                score=round(m_prec, 4),
                threshold=self.config.min_context_precision,
                passed=m_prec >= self.config.min_context_precision,
                description="Measures retrieval precision & ranking (ground-truth chunks ranked top).",
            ),
            MetricThresholdResult(
                name="Context Recall",
                score=round(m_rec, 4),
                threshold=self.config.min_context_recall,
                passed=m_rec >= self.config.min_context_recall,
                description="Measures retrieval recall (all information needed to answer was retrieved).",
            ),
        ]

        def safe_float(val: Any) -> float:
            if val is None or pd.isna(val):
                return 0.0
            try:
                return round(float(val), 4)
            except Exception:
                return 0.0

        # Convert samples DataFrame to serializable dict
        samples_list = []
        for idx, row in df.iterrows():
            samples_list.append(
                {
                    "index": int(idx) + 1,
                    "user_input": str(row.get("user_input", "")),
                    "response": str(row.get("response", "")),
                    "reference": str(row.get("reference", "")),
                    "retrieved_contexts_count": len(row.get("retrieved_contexts", [])),
                    "faithfulness": safe_float(row.get("faithfulness")),
                    "answer_relevancy": safe_float(row.get("answer_relevancy")),
                    "context_precision": safe_float(row.get("context_precision")),
                    "context_recall": safe_float(row.get("context_recall")),
                }
            )

        return EvaluationReportData(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_samples=len(df),
            mean_faithfulness=round(m_faith, 4),
            mean_answer_relevancy=round(m_rel, 4),
            mean_context_precision=round(m_prec, 4),
            mean_context_recall=round(m_rec, 4),
            overall_score=round(overall, 4),
            threshold_results=[asdict(t) for t in thresholds],
            samples_table=samples_list,
        )


