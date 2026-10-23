"""Dataset converter for Ragas Evaluation.

Converts standard Python dictionary records into Ragas EvaluationDataset.
"""
from __future__ import annotations

from typing import Any
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample


def build_ragas_dataset(samples: list[dict[str, Any]]) -> EvaluationDataset:
    """Convert sample dictionaries into a Ragas EvaluationDataset.

    Each dict must provide:
    - user_input (or question)
    - response (or answer)
    - retrieved_contexts (or contexts: list[str])
    - reference (or ground_truth)
    """
    ragas_samples: list[SingleTurnSample] = []
    for s in samples:
        user_input = s.get("user_input") or s.get("question", "")
        response = s.get("response") or s.get("answer", "")
        contexts = s.get("retrieved_contexts") or s.get("contexts", [])
        if isinstance(contexts, str):
            contexts = [contexts]
        reference = s.get("reference") or s.get("ground_truth", "")

        ragas_samples.append(
            SingleTurnSample(
                user_input=user_input,
                response=response,
                retrieved_contexts=contexts,
                reference=reference,
            )
        )
    return EvaluationDataset(samples=ragas_samples)
