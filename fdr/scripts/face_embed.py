from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """L2 normalize a 1D embedding vector."""
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def extract_face_embedding(face: Any) -> np.ndarray:
    """
    Extract normalized embedding from one detected face.

    InsightFace FaceAnalysis usually returns face.embedding directly.
    """
    embedding = getattr(face, "embedding", None)
    if embedding is None:
        raise ValueError("Face object does not contain embedding.")

    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    return l2_normalize(embedding)


def extract_face_embeddings(faces: Iterable[Any]) -> list[np.ndarray]:
    """Extract embeddings for a list of faces."""
    embeddings: list[np.ndarray] = []
    for idx, face in enumerate(faces):
        try:
            embeddings.append(extract_face_embedding(face))
        except ValueError as exc:
            raise ValueError(f"Face index {idx} has no embedding.") from exc
    return embeddings

