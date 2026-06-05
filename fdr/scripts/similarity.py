from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class IsotonicConfidenceCalibrator:
    x_thresholds: np.ndarray
    y_thresholds: np.ndarray
    best_probability_threshold: float
    source_path: Path


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(vec_a, dtype=np.float32).reshape(-1)
    b = np.asarray(vec_b, dtype=np.float32).reshape(-1)

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        raise ValueError("Cannot compute cosine similarity with zero-norm vector.")

    return float(np.dot(a, b) / denom)


def best_cosine_similarity(
    query_embedding: np.ndarray,
    candidate_embeddings: Sequence[np.ndarray],
) -> tuple[float, int]:
    """
    Return highest cosine similarity and its face index.

    Used when one image contains multiple faces.
    """
    if not candidate_embeddings:
        raise ValueError("candidate_embeddings is empty.")

    scores = [cosine_similarity(query_embedding, emb) for emb in candidate_embeddings]
    best_idx = int(np.argmax(scores))
    return float(scores[best_idx]), best_idx


def load_isotonic_calibrator(json_path: Path) -> IsotonicConfidenceCalibrator:
    """Load the selected isotonic calibration model from calibration_results.json."""
    if not json_path.exists():
        raise FileNotFoundError(f"Calibration results not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    model = payload["models"]["isotonic_regression"]
    x_thresholds = np.asarray(model["x_thresholds"], dtype=np.float32).reshape(-1)
    y_thresholds = np.asarray(model["y_thresholds"], dtype=np.float32).reshape(-1)
    if x_thresholds.size == 0 or x_thresholds.size != y_thresholds.size:
        raise ValueError("Invalid isotonic calibration thresholds.")

    return IsotonicConfidenceCalibrator(
        x_thresholds=x_thresholds,
        y_thresholds=y_thresholds,
        best_probability_threshold=float(model["best_probability_threshold_by_train_f1"]),
        source_path=json_path,
    )


def cosine_to_confidence(cosine_score: float, calibrator: IsotonicConfidenceCalibrator) -> float:
    """Map raw cosine similarity to calibrated P(same person | cosine)."""
    confidence = np.interp(
        float(cosine_score),
        calibrator.x_thresholds,
        calibrator.y_thresholds,
        left=float(calibrator.y_thresholds[0]),
        right=float(calibrator.y_thresholds[-1]),
    )
    return float(np.clip(confidence, 0.0, 1.0))
