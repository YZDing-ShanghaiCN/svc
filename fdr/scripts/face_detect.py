from __future__ import annotations

from typing import Any, Iterable

import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ArcFace standard 5-point landmarks for 112x112 alignment
_ARC_FACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def create_face_analyzer(
    det_size: tuple[int, int] = (640, 640),
    det_thresh: float = 0.5,
    ctx_id: int = -1,
    providers: list[str] | None = None,
    root: str | None = None,
) -> FaceAnalysis:
    """
    Create and initialize InsightFace FaceAnalysis.

    Default settings use CPU (ctx_id=-1).
    """
    if providers is None:
        providers = ["CPUExecutionProvider"]

    kwargs = {"name": "buffalo_l", "providers": providers}
    if root is not None:
        kwargs["root"] = root

    analyzer = FaceAnalysis(**kwargs)
    analyzer.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)
    return analyzer


def detect_faces(image_bgr: np.ndarray, analyzer: FaceAnalysis) -> list[Any]:
    """Detect faces and return them sorted by detection score (high to low)."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty.")

    faces = analyzer.get(image_bgr)
    faces.sort(key=lambda f: float(getattr(f, "det_score", 0.0)), reverse=True)
    return faces


def face_area(face: Any) -> float:
    """Compute bbox area of one face."""
    bbox = np.asarray(face.bbox, dtype=np.float32)
    w = max(0.0, float(bbox[2] - bbox[0]))
    h = max(0.0, float(bbox[3] - bbox[1]))
    return w * h


def select_largest_face(faces: Iterable[Any]) -> Any:
    """
    Select the largest face by bbox area.

    Useful when query image contains multiple faces.
    """
    faces = list(faces)
    if not faces:
        raise ValueError("No faces were provided.")
    return max(faces, key=face_area)


def crop_face(image_bgr: np.ndarray, face: Any, margin_ratio: float = 0.2) -> np.ndarray:
    """Crop one face region from the original image."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = np.asarray(face.bbox, dtype=np.float32)

    bw = x2 - x1
    bh = y2 - y1
    mx = bw * margin_ratio
    my = bh * margin_ratio

    x1i = max(0, int(np.floor(x1 - mx)))
    y1i = max(0, int(np.floor(y1 - my)))
    x2i = min(w, int(np.ceil(x2 + mx)))
    y2i = min(h, int(np.ceil(y2 + my)))

    return image_bgr[y1i:y2i, x1i:x2i].copy()


def align_face(
    image_bgr: np.ndarray,
    face: Any,
    output_size: tuple[int, int] = (112, 112),
) -> np.ndarray:
    """
    Align face image with 5-point landmarks.

    If landmarks are unavailable, fallback to cropped face.
    """
    kps = getattr(face, "kps", None)
    if kps is None:
        return crop_face(image_bgr, face)

    kps = np.asarray(kps, dtype=np.float32)
    if kps.shape != (5, 2):
        return crop_face(image_bgr, face)

    out_w, out_h = output_size
    ref = _ARC_FACE_TEMPLATE.copy()
    ref[:, 0] = ref[:, 0] * (out_w / 112.0)
    ref[:, 1] = ref[:, 1] * (out_h / 112.0)

    matrix, _ = cv2.estimateAffinePartial2D(kps, ref, method=cv2.LMEDS)
    if matrix is None:
        return crop_face(image_bgr, face)

    aligned = cv2.warpAffine(
        image_bgr,
        matrix,
        dsize=(out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return aligned

