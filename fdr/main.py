from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.camera import capture_photo
from scripts.face_detect import create_face_analyzer, detect_faces, select_largest_face
from scripts.face_embed import extract_face_embedding, extract_face_embeddings
from scripts.similarity import (
    best_cosine_similarity,
    cosine_to_confidence,
    load_isotonic_calibrator,
)
from scripts.utils import (
    bbox_to_int_list,
    build_visualization_filename,
    create_run_dirs,
    draw_face_annotations,
    list_image_files,
    load_image,
    print_ranked_results,
    save_image,
    save_run_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face detection and calibrated confidence runner.")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=("compare", "capture"),
        help=(
            "compare: use the first image under pictures/ as baseline; "
            "capture: capture a query image from camera."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve project-relative paths from main.py location.
    project_root = Path(__file__).resolve().parent
    pictures_dir = project_root / "pictures/xin"
    outputs_dir = project_root / "outputs"
    calibration_path = project_root / "calibration_results.json"

    pictures_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    run_dir, vis_dir, run_id = create_run_dirs(outputs_dir)
    print(f"[INFO] Run directory: {run_dir}")

    calibrator = load_isotonic_calibrator(calibration_path)
    print(f"[INFO] Confidence model: isotonic_regression ({calibrator.source_path})")

    # Collect candidate images once. compare mode uses the first one as query baseline.
    image_paths = list_image_files(pictures_dir)
    if args.mode == "compare":
        if not image_paths:
            raise RuntimeError("No images found under pictures/ for compare mode.")
        query_path = image_paths[0]
        print(f"[INFO] Compare mode. Baseline image: {query_path}")
    else:
        query_path = outputs_dir / "query.jpg"
        print("[INFO] Opening camera. Press SPACE to capture, or q to cancel.")
        capture_photo(query_path)
        print(f"[INFO] Query image saved: {query_path}")
        if not image_paths:
            print("[WARNING] No images found under pictures/.")

    # Initialize InsightFace with pretrained models.
    analyzer = create_face_analyzer(
        det_size=(640, 640), 
        det_thresh=0.5, 
        ctx_id=-1,
        root=str(project_root)
    )

    # Step 1: detect query face and extract embedding.
    query_image = load_image(query_path)
    query_faces = detect_faces(query_image, analyzer)
    if not query_faces:
        raise RuntimeError(f"No face detected in query image: {query_path}")

    query_face = select_largest_face(query_faces)
    query_embedding = extract_face_embedding(query_face)
    query_idx = next(i for i, face in enumerate(query_faces) if face is query_face)
    query_confidence = cosine_to_confidence(1.0, calibrator)
    query_annotated = draw_face_annotations(
        query_image,
        query_faces,
        query_idx,
        query_confidence,
        score_label="conf",
        score_as_percent=True,
    )
    save_image(query_annotated, vis_dir / "query_bbox.jpg")

    # Step 2: iterate over all images in pictures/ and compare.
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    query_path_resolved = query_path.resolve()

    for image_path in image_paths:
        image_name = image_path.name

        # In compare mode, force baseline image to 100%.
        if args.mode == "compare" and image_path.resolve() == query_path_resolved:
            best_cosine = 1.0
            best_confidence = cosine_to_confidence(best_cosine, calibrator)
            best_idx = query_idx
            best_face = query_face

            annotated = draw_face_annotations(
                query_image,
                query_faces,
                best_idx,
                best_confidence,
                score_label="conf",
                score_as_percent=True,
            )
            vis_name = build_visualization_filename(image_path, pictures_dir)
            vis_path = vis_dir / vis_name
            save_image(annotated, vis_path)

            results.append(
                {
                    "image_name": image_name,
                    "image_path": str(image_path.relative_to(project_root)),
                    "cosine_similarity": float(best_cosine),
                    "similarity": float(best_cosine),
                    "similarity_percent": float(best_cosine * 100.0),
                    "confidence": float(best_confidence),
                    "confidence_percent": float(best_confidence * 100.0),
                    "confidence_model": "isotonic_regression",
                    "confidence_threshold": float(calibrator.best_probability_threshold),
                    "num_faces": len(query_faces),
                    "best_face_index": int(best_idx),
                    "best_face_bbox": bbox_to_int_list(best_face.bbox),
                    "visualization_path": str(vis_path.relative_to(project_root)),
                }
            )
            continue

        try:
            image = load_image(image_path)
        except ValueError as exc:
            print(f"[WARNING] {exc}")
            skipped.append({"image_name": image_name, "reason": "load_failed"})
            continue

        faces = detect_faces(image, analyzer)
        if not faces:
            print(f"[WARNING] No face detected in {image_name}, skipped.")
            skipped.append({"image_name": image_name, "reason": "no_face_detected"})
            continue

        try:
            embeddings = extract_face_embeddings(faces)
        except ValueError as exc:
            print(f"[WARNING] {image_name} embedding extraction failed: {exc}")
            skipped.append({"image_name": image_name, "reason": "embedding_failed"})
            continue
        best_cosine, best_idx = best_cosine_similarity(query_embedding, embeddings)
        best_confidence = cosine_to_confidence(best_cosine, calibrator)
        best_face = faces[best_idx]

        annotated = draw_face_annotations(
            image,
            faces,
            best_idx,
            best_confidence,
            score_label="conf",
            score_as_percent=True,
        )
        vis_name = build_visualization_filename(image_path, pictures_dir)
        vis_path = vis_dir / vis_name
        save_image(annotated, vis_path)

        results.append(
            {
                "image_name": image_name,
                "image_path": str(image_path.relative_to(project_root)),
                "cosine_similarity": float(best_cosine),
                "similarity": float(best_cosine),
                "similarity_percent": float(best_cosine * 100.0),
                "confidence": float(best_confidence),
                "confidence_percent": float(best_confidence * 100.0),
                "confidence_model": "isotonic_regression",
                "confidence_threshold": float(calibrator.best_probability_threshold),
                "num_faces": len(faces),
                "best_face_index": int(best_idx),
                "best_face_bbox": bbox_to_int_list(best_face.bbox),
                "visualization_path": str(vis_path.relative_to(project_root)),
            }
        )

    # Step 3: sort by calibrated confidence from high to low and print.
    results.sort(key=lambda x: (x["confidence"], x["cosine_similarity"]), reverse=True)
    print_ranked_results(results)

    run_config = {
        "run_id": run_id,
        "run_time_local": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "mode": args.mode,
        "query_image": str(query_path.relative_to(project_root)),
        "query_face_bbox": bbox_to_int_list(query_face.bbox),
        "pictures_dir": str(pictures_dir.relative_to(project_root)),
        "outputs_dir": str(outputs_dir.relative_to(project_root)),
        "visualization_dir": str(vis_dir.relative_to(project_root)),
        "confidence_model": "isotonic_regression",
        "calibration_results_json": str(calibration_path.relative_to(project_root)),
        "confidence_threshold": float(calibrator.best_probability_threshold),
        "total_images_found": len(image_paths),
        "valid_images_compared": len(results),
        "skipped_images": skipped,
        "results_sorted": results,
    }
    config_path = save_run_config(run_dir, run_config)
    print(f"\n[INFO] Config saved: {config_path}")


if __name__ == "__main__":
    main()
