from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

try:
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover - import error handled in main
    FaceAnalysis = None  # type: ignore[assignment]


IDENTITY_PATTERN = re.compile(r"(.+)_\d+\.(jpg|jpeg|png|bmp)$", re.IGNORECASE)
DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODEL_FILES = (
    "1k3d68.onnx",
    "2d106det.onnx",
    "det_10g.onnx",
    "genderage.onnx",
    "w600k_r50.onnx",
)


@dataclass(frozen=True)
class TrainRecord:
    filename: str
    fold_id: int
    identity: str
    image_path: Path


def parse_identity(filename: str) -> Optional[str]:
    match = IDENTITY_PATTERN.match(filename)
    if not match:
        return None
    return match.group(1)


def load_train_list(train_txt: Path, image_root: Path) -> tuple[list[TrainRecord], int, int]:
    if not train_txt.exists():
        raise FileNotFoundError(f"train_txt not found: {train_txt}")
    if not image_root.exists():
        raise FileNotFoundError(f"image_root not found: {image_root}")

    with train_txt.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    total_lines = len(lines)
    missing_count = 0
    malformed_count = 0
    valid_records: list[TrainRecord] = []
    seen_files: set[str] = set()

    for raw_line in tqdm(lines, desc="Loading train list"):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 2:
            malformed_count += 1
            continue

        filename = parts[0]
        fold_text = parts[1]

        try:
            fold_id = int(fold_text)
        except ValueError:
            malformed_count += 1
            continue

        identity = parse_identity(filename)
        if identity is None:
            malformed_count += 1
            continue

        image_path = image_root / filename
        if not image_path.exists():
            missing_count += 1
            continue

        # Keep first appearance if train.txt contains duplicated rows.
        if filename in seen_files:
            continue
        seen_files.add(filename)

        valid_records.append(
            TrainRecord(
                filename=filename,
                fold_id=fold_id,
                identity=identity,
                image_path=image_path,
            )
        )

    if malformed_count > 0:
        print(f"[WARNING] Skipped malformed lines: {malformed_count}")

    return valid_records, total_lines, missing_count


def build_positive_pairs(
    records: list[TrainRecord],
    max_pos_pairs_per_id: int,
    rng: random.Random,
) -> list[tuple[str, str, str, str, int, str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.fold_id != 0:
            grouped[record.identity].append(record.filename)

    positive_pairs: list[tuple[str, str, str, str, int, str]] = []
    for identity, filenames in tqdm(grouped.items(), desc="Building positive pairs"):
        if len(filenames) < 2:
            continue

        all_pairs = list(combinations(sorted(filenames), 2))
        if max_pos_pairs_per_id > 0 and len(all_pairs) > max_pos_pairs_per_id:
            sampled_pairs = rng.sample(all_pairs, k=max_pos_pairs_per_id)
        else:
            sampled_pairs = all_pairs

        for img1, img2 in sampled_pairs:
            positive_pairs.append((img1, img2, identity, identity, 1, "positive"))

    return positive_pairs


def build_negative_pairs(
    records: list[TrainRecord],
    target_count: int,
    rng: random.Random,
) -> list[tuple[str, str, str, str, int, str]]:
    if target_count <= 0 or len(records) < 2:
        return []

    identity_counts = Counter(record.identity for record in records)
    n = len(records)
    all_pairs_count = n * (n - 1) // 2
    same_id_pairs_count = sum(c * (c - 1) // 2 for c in identity_counts.values())
    max_available_negative = all_pairs_count - same_id_pairs_count

    negative_pairs: list[tuple[str, str, str, str, int, str]] = []
    used_pair_keys: set[tuple[str, str]] = set()
    max_attempts = max(target_count * 30, 10000)

    for _ in tqdm(range(max_attempts), desc="Sampling negative pairs"):
        if len(negative_pairs) >= target_count:
            break

        rec1, rec2 = rng.sample(records, 2)
        if rec1.identity == rec2.identity:
            continue

        if rec1.filename <= rec2.filename:
            img1, img2 = rec1.filename, rec2.filename
            id1, id2 = rec1.identity, rec2.identity
        else:
            img1, img2 = rec2.filename, rec1.filename
            id1, id2 = rec2.identity, rec1.identity

        key = (img1, img2)
        if key in used_pair_keys:
            continue

        used_pair_keys.add(key)
        negative_pairs.append((img1, img2, id1, id2, 0, "random_negative"))

    if target_count > max_available_negative:
        print(
            "[WARNING] Target negative pair count exceeds maximum possible "
            f"({target_count} > {max_available_negative})."
        )

    if len(negative_pairs) < target_count:
        print(
            "[WARNING] Could not sample enough negative pairs after retries: "
            f"{len(negative_pairs)} / {target_count}."
        )

    return negative_pairs


def _resolve_model_root(model_root: Path) -> tuple[Path, Path]:
    root = model_root.resolve()

    candidates = [
        (root, root / "models" / "buffalo_l"),
        (root.parent, root / "buffalo_l"),
        (root.parent.parent, root),
    ]
    for insightface_root, model_dir in candidates:
        if model_dir.name != "buffalo_l" or not model_dir.exists():
            continue

        missing_files = [
            filename for filename in REQUIRED_MODEL_FILES if not (model_dir / filename).exists()
        ]
        if missing_files:
            raise FileNotFoundError(
                f"Local InsightFace model is incomplete: {model_dir}. "
                f"Missing files: {', '.join(missing_files)}"
            )
        return insightface_root, model_dir

    raise FileNotFoundError(
        "Local InsightFace model not found. Expected one of these layouts: "
        f"{root / 'models' / 'buffalo_l'} or {root / 'buffalo_l'}"
    )


def init_face_model(det_size: int, providers: list[str], model_root: Path) -> FaceAnalysis:
    if FaceAnalysis is None:
        raise ImportError(
            "insightface is not available. Please install insightface in your environment."
        )

    insightface_root, model_dir = _resolve_model_root(model_root)
    print(f"[INFO] Loading InsightFace model from: {model_dir}")

    model = FaceAnalysis(name="buffalo_l", root=str(insightface_root), providers=providers)
    try:
        model.prepare(ctx_id=0, det_size=(det_size, det_size))
    except Exception as exc:
        print(f"[WARNING] Failed to prepare model with ctx_id=0. Fallback to CPU. ({exc})")
        model.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return model


def extract_embedding(model: FaceAnalysis, image_path: Path) -> Optional[np.ndarray]:
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return None

        faces = model.get(image)
        if not faces:
            return None

        def bbox_area(face: object) -> float:
            bbox = np.asarray(getattr(face, "bbox", [0, 0, 0, 0]), dtype=np.float32)
            w = max(0.0, float(bbox[2] - bbox[0]))
            h = max(0.0, float(bbox[3] - bbox[1]))
            return w * h

        best_face = max(faces, key=bbox_area)
        embedding = getattr(best_face, "embedding", None)
        if embedding is None:
            return None

        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(emb))
        if norm == 0.0:
            return None
        return emb / norm
    except Exception as exc:
        print(f"[WARNING] Failed to extract embedding for {image_path}: {exc}")
        return None


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    # emb1 and emb2 are already L2-normalized in extract_embedding.
    return float(np.dot(emb1, emb2))


def _parse_providers(providers_text: str) -> list[str]:
    providers = [item.strip() for item in providers_text.split(",") if item.strip()]
    if not providers:
        return ["CPUExecutionProvider"]
    return providers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LFW-style face pairs and cosine similarity scores."
    )
    parser.add_argument("--train_txt", type=Path, default=Path("./fdr/train.txt"), help="Path to train.txt")
    parser.add_argument("--image_root", type=Path, default=Path("./fdr/aligned_images"), help="Image root directory")
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("./fdr/train_pairs_with_score.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--neg_ratio",
        type=float,
        default=2.0,
        help="Negative sample ratio relative to positive pairs",
    )
    parser.add_argument(
        "--max_pos_pairs_per_id",
        type=int,
        default=2,
        help="Maximum number of positive pairs per identity",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--det_size",
        type=int,
        default=640,
        help="InsightFace detection input size",
    )
    parser.add_argument(
        "--providers",
        type=str,
        default="CUDAExecutionProvider,CPUExecutionProvider",
        help="InsightFace providers, comma separated",
    )
    parser.add_argument(
        "--model_root",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
        help=(
            "InsightFace model root. Default is the fdr project directory, "
            "which uses fdr/models/buffalo_l."
        ),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    rng = random.Random(args.seed)

    try:
        records, total_lines, missing_count = load_train_list(args.train_txt, args.image_root)
    except Exception as exc:
        print(f"[ERROR] Failed to load train list: {exc}")
        sys.exit(1)

    identity_count = len({record.identity for record in records})

    positive_pairs = build_positive_pairs(
        records=records,
        max_pos_pairs_per_id=args.max_pos_pairs_per_id,
        rng=rng,
    )
    raw_positive_count = len(positive_pairs)
    target_negative_count = int(raw_positive_count * args.neg_ratio)

    negative_pairs = build_negative_pairs(records=records, target_count=target_negative_count, rng=rng)

    providers = _parse_providers(args.providers)
    try:
        face_model = init_face_model(
            det_size=args.det_size,
            providers=providers,
            model_root=args.model_root,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to initialize InsightFace model: {exc}")
        sys.exit(1)

    filename_to_record = {record.filename: record for record in records}
    embedding_cache: dict[str, Optional[np.ndarray]] = {}

    def get_embedding(filename: str) -> Optional[np.ndarray]:
        if filename in embedding_cache:
            return embedding_cache[filename]

        record = filename_to_record.get(filename)
        if record is None:
            embedding_cache[filename] = None
            return None

        emb = extract_embedding(face_model, record.image_path)
        embedding_cache[filename] = emb
        return emb

    output_rows: list[dict[str, object]] = []
    positive_success_count = 0
    negative_success_count = 0

    for img1, img2, id1, id2, label, pair_type in tqdm(
        positive_pairs, desc="Scoring positive pairs"
    ):
        emb1 = get_embedding(img1)
        emb2 = get_embedding(img2)
        if emb1 is None or emb2 is None:
            continue

        score = cosine_similarity(emb1, emb2)
        output_rows.append(
            {
                "img1": img1,
                "img2": img2,
                "label": label,
                "cosine": f"{score:.6f}",
                "id1": id1,
                "id2": id2,
                "pair_type": pair_type,
            }
        )
        positive_success_count += 1

    for img1, img2, id1, id2, label, pair_type in tqdm(
        negative_pairs, desc="Scoring negative pairs"
    ):
        emb1 = get_embedding(img1)
        emb2 = get_embedding(img2)
        if emb1 is None or emb2 is None:
            continue

        score = cosine_similarity(emb1, emb2)
        output_rows.append(
            {
                "img1": img1,
                "img2": img2,
                "label": label,
                "cosine": f"{score:.6f}",
                "id1": id1,
                "id2": id2,
                "pair_type": pair_type,
            }
        )
        negative_success_count += 1

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["img1", "img2", "label", "cosine", "id1", "id2", "pair_type"],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"1. train.txt total lines: {total_lines}")
    print(f"2. valid images: {len(records)}")
    print(f"3. missing images: {missing_count}")
    print(f"4. identity count: {identity_count}")
    print(f"5. raw positive pair count: {raw_positive_count}")
    print(f"6. positive pairs with cosine: {positive_success_count}")
    print(f"7. target negative pair count: {target_negative_count}")
    print(f"8. negative pairs with cosine: {negative_success_count}")
    print(f"9. output csv: {args.output_csv}")


if __name__ == "__main__":
    main()
