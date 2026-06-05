from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def create_run_dirs(outputs_root: Path) -> tuple[Path, Path, str]:
    """
    Create timestamp run directory under outputs/, for example:
    outputs/20260530_102030/
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = outputs_root / run_id
    vis_dir = run_dir / "bbox_confidence"
    vis_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, vis_dir, run_id


def list_image_files(pictures_dir: Path) -> list[Path]:
    """Recursively collect jpg/jpeg/png image files under pictures/."""
    if not pictures_dir.exists():
        return []

    files = [
        p
        for p in pictures_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return sorted(files)


def load_image(image_path: Path) -> np.ndarray:
    """Load an image as BGR ndarray using OpenCV."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image


def save_image(image: np.ndarray, output_path: Path) -> None:
    """Save image to disk, ensuring parent folder exists."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise RuntimeError(f"Failed to save image: {output_path}")


def bbox_to_int_list(bbox: Iterable[float]) -> list[int]:
    """Convert bbox array-like [x1, y1, x2, y2] to int list."""
    arr = np.asarray(list(bbox), dtype=np.float32).reshape(-1)
    return [int(round(v)) for v in arr[:4]]


def draw_face_annotations(
    image_bgr: np.ndarray,
    faces: list[Any],
    best_face_idx: int,
    best_score: float,
    score_label: str = "sim",
    score_as_percent: bool = False,
) -> np.ndarray:
    """
    Draw all detected bboxes and highlight the best matched face.
    """
    canvas = image_bgr.copy()

    for idx, face in enumerate(faces):
        x1, y1, x2, y2 = bbox_to_int_list(face.bbox)
        if idx == best_face_idx:
            color = (0, 200, 0)
            if score_as_percent:
                text = f"{score_label}={best_score * 100.0:.2f}%"
            else:
                text = f"{score_label}={best_score:.4f}"
            thickness = 2
        else:
            color = (0, 200, 255)
            text = "other_face"
            thickness = 1

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        text_y = y1 - 10 if y1 - 10 > 15 else y1 + 20
        cv2.putText(
            canvas,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return canvas


def build_visualization_filename(image_path: Path, pictures_dir: Path) -> str:
    """
    Build a stable output filename for visualization.
    Example: pictures/a/b/c.jpg -> a__b__c_bbox_sim.jpg
    """
    rel_stem = image_path.relative_to(pictures_dir).with_suffix("")
    safe_stem = "__".join(rel_stem.parts)
    return f"{safe_stem}_bbox_sim.jpg"


def print_ranked_results(results: list[dict[str, Any]]) -> None:
    """Print sorted comparison results in descending order."""
    if not results:
        print("[INFO] No valid comparison result.")
        return

    if "confidence" in results[0]:
        print("\n===== Confidence Ranking (High -> Low) =====")
        for rank, item in enumerate(results, start=1):
            confidence_percent = float(item["confidence"]) * 100.0
            cosine = float(item.get("cosine_similarity", item.get("similarity", 0.0)))
            print(
                f"{rank:02d}. {item['image_name']:<30} "
                f"confidence={confidence_percent:.2f}% "
                f"cosine={cosine:.4f} "
                f"faces={item['num_faces']}"
            )
        return

    print("\n===== Similarity Ranking (High -> Low) =====")
    for rank, item in enumerate(results, start=1):
        sim_percent = float(item["similarity"]) * 100.0
        print(
            f"{rank:02d}. {item['image_name']:<30} "
            f"similarity={sim_percent:.2f}% "
            f"faces={item['num_faces']}"
        )


def save_run_config(run_dir: Path, config: dict[str, Any]) -> Path:
    """Save run metadata to JSON config file."""
    config_path = run_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return config_path
