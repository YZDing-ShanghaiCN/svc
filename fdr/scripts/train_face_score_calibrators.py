from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def _to_python(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_python(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_python(v) for v in value]
    if isinstance(value, tuple):
        return [_to_python(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_python(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def load_pair_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"label", "cosine"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    clean_df = df.copy()
    clean_df["label"] = pd.to_numeric(clean_df["label"], errors="coerce")
    clean_df["cosine"] = pd.to_numeric(clean_df["cosine"], errors="coerce")
    clean_df = clean_df.dropna(subset=["label", "cosine"])
    clean_df["label"] = clean_df["label"].astype(int)
    clean_df["cosine"] = clean_df["cosine"].astype(float)
    clean_df = clean_df[clean_df["label"].isin([0, 1])].reset_index(drop=True)

    if clean_df.empty:
        raise ValueError(f"No valid rows after cleaning: {csv_path}")

    return clean_df


def split_data_if_needed(
    train_csv: Path,
    test_csv: Optional[Path],
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    train_df = load_pair_csv(train_csv)

    if test_csv is not None:
        test_df = load_pair_csv(test_csv)
        return train_df, test_df, False

    unique_labels = train_df["label"].nunique()
    if unique_labels < 2:
        raise ValueError(
            "train_csv has fewer than 2 classes. Cannot do stratified train/test split."
        )

    train_df_split, test_df_split = train_test_split(
        train_df,
        test_size=test_size,
        random_state=seed,
        stratify=train_df["label"],
    )
    return train_df_split.reset_index(drop=True), test_df_split.reset_index(drop=True), True


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> LogisticRegression:
    if np.unique(y_train).size < 2:
        raise ValueError("LogisticRegression requires at least two classes in y_train.")
    model = LogisticRegression(solver="lbfgs", random_state=seed)
    model.fit(X_train, y_train)
    return model


def train_isotonic_regression(cosine_train: np.ndarray, y_train: np.ndarray) -> IsotonicRegression:
    if np.unique(y_train).size < 2:
        raise ValueError("IsotonicRegression requires at least two classes in y_train.")
    model = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        increasing=True,
        out_of_bounds="clip",
    )
    model.fit(cosine_train, y_train)
    return model


def predict_logistic(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def predict_isotonic(model: IsotonicRegression, cosine_scores: np.ndarray) -> np.ndarray:
    return model.predict(cosine_scores)


def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int) -> tuple[float, list[dict[str, Any]]]:
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    probs = np.clip(probs, 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins[1:-1], right=False)

    total = max(1, probs.shape[0])
    ece = 0.0
    reliability_bins: list[dict[str, Any]] = []

    for idx in range(n_bins):
        mask = bin_ids == idx
        count = int(np.sum(mask))
        bin_left = float(bins[idx])
        bin_right = float(bins[idx + 1])

        if count == 0:
            reliability_bins.append(
                {
                    "bin_left": bin_left,
                    "bin_right": bin_right,
                    "count": 0,
                    "mean_confidence": None,
                    "positive_rate": None,
                }
            )
            continue

        mean_confidence = float(np.mean(probs[mask]))
        positive_rate = float(np.mean(y_true[mask]))
        ece += (count / total) * abs(positive_rate - mean_confidence)

        reliability_bins.append(
            {
                "bin_left": bin_left,
                "bin_right": bin_right,
                "count": count,
                "mean_confidence": mean_confidence,
                "positive_rate": positive_rate,
            }
        )

    return float(ece), reliability_bins


def _safe_roc_auc(y_true: np.ndarray, probs: np.ndarray) -> Optional[float]:
    if np.unique(y_true).size < 2:
        return None
    try:
        return float(roc_auc_score(y_true, probs))
    except Exception:
        return None


def _classification_metrics_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    y_pred = (scores >= threshold).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def compute_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    probs = np.clip(probs, 0.0, 1.0)
    probs_for_log = np.clip(probs, 1e-7, 1.0 - 1e-7)
    y_pred = (probs >= threshold).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    try:
        ll = float(log_loss(y_true, probs_for_log, labels=[0, 1]))
    except Exception:
        ll = None

    ece, reliability_bins = compute_ece(y_true, probs, n_bins)

    return {
        "brier_score": float(brier_score_loss(y_true, probs)),
        "roc_auc": _safe_roc_auc(y_true, probs),
        "log_loss": ll,
        "accuracy_at_0_5": float(accuracy_score(y_true, y_pred)),
        "precision_at_0_5": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix_at_0_5": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "ece": float(ece),
        "reliability_bins": reliability_bins,
    }


def find_best_probability_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)

    thresholds = np.arange(0.01, 1.0, 0.01)
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = (probs >= threshold).astype(np.int32)
        score = float(f1_score(y_true, y_pred, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def find_best_cosine_threshold(
    y_true: np.ndarray,
    cosine_scores: np.ndarray,
    num_points: int = 200,
) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    cosine_scores = np.asarray(cosine_scores, dtype=np.float64).reshape(-1)

    c_min = float(np.min(cosine_scores))
    c_max = float(np.max(cosine_scores))

    if c_min == c_max:
        thresholds = np.array([c_min], dtype=np.float64)
    else:
        thresholds = np.linspace(c_min, c_max, num=num_points)

    best_threshold = float(thresholds[0])
    best_f1 = -1.0
    for threshold in thresholds:
        y_pred = (cosine_scores >= threshold).astype(np.int32)
        score = float(f1_score(y_true, y_pred, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def _build_data_summary(
    train_csv: Path,
    test_csv: Optional[Path],
    used_internal_split: bool,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    train_cos = train_df["cosine"].to_numpy(dtype=np.float64)
    test_cos = test_df["cosine"].to_numpy(dtype=np.float64)
    train_labels = train_df["label"].to_numpy(dtype=np.int32)
    test_labels = test_df["label"].to_numpy(dtype=np.int32)

    return {
        "train_csv": str(train_csv),
        "test_csv": str(test_csv) if test_csv is not None else None,
        "used_internal_split": used_internal_split,
        "train_size": int(train_df.shape[0]),
        "test_size": int(test_df.shape[0]),
        "train_positive": int(np.sum(train_labels == 1)),
        "train_negative": int(np.sum(train_labels == 0)),
        "test_positive": int(np.sum(test_labels == 1)),
        "test_negative": int(np.sum(test_labels == 0)),
        "cosine_train_min": float(np.min(train_cos)),
        "cosine_train_max": float(np.max(train_cos)),
        "cosine_train_mean": float(np.mean(train_cos)),
        "cosine_test_min": float(np.min(test_cos)),
        "cosine_test_max": float(np.max(test_cos)),
        "cosine_test_mean": float(np.mean(test_cos)),
    }


def _compare_models_for_recommendation(
    logistic_section: dict[str, Any],
    isotonic_section: dict[str, Any],
) -> dict[str, Any]:
    notes: list[str] = [
        "Logistic Regression output is smoother and is usually more stable on smaller datasets.",
        "Isotonic Regression is more flexible; if its test Brier/ECE is lower, it can be preferred.",
    ]

    lr_test = logistic_section["test_metrics"]
    ir_test = isotonic_section["test_metrics"]

    better_brier = (
        "logistic_regression"
        if lr_test["brier_score"] <= ir_test["brier_score"]
        else "isotonic_regression"
    )
    better_ece = (
        "logistic_regression" if lr_test["ece"] <= ir_test["ece"] else "isotonic_regression"
    )

    lr_best_f1 = logistic_section["test_metrics_at_best_probability_threshold"]["f1"]
    ir_best_f1 = isotonic_section["test_metrics_at_best_probability_threshold"]["f1"]
    better_f1 = "logistic_regression" if lr_best_f1 >= ir_best_f1 else "isotonic_regression"

    lr_train_brier = logistic_section["train_metrics"]["brier_score"]
    ir_train_brier = isotonic_section["train_metrics"]["brier_score"]
    ir_overfit = ir_train_brier < lr_train_brier and ir_test["brier_score"] > lr_test["brier_score"]
    if ir_overfit:
        notes.append(
            "Isotonic looks better on train but worse on test, which may indicate overfitting."
        )

    return {
        "better_by_test_brier_score": better_brier,
        "better_by_test_ece": better_ece,
        "better_by_test_f1_at_best_threshold": better_f1,
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train face cosine score calibrators (Logistic + Isotonic)."
    )
    parser.add_argument("--train_csv", type=Path, default="./fdr/train_pairs_with_score.csv", help="Training CSV path")
    parser.add_argument("--test_csv", type=Path, default=None, help="Optional test CSV path")
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path(".fdr/calibration_results.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test split ratio when test_csv is not provided",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_bins", type=int, default=10, help="Number of bins for ECE")
    args = parser.parse_args()

    train_df, test_df, used_internal_split = split_data_if_needed(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        test_size=args.test_size,
        seed=args.seed,
    )

    X_train = train_df["cosine"].to_numpy(dtype=np.float64).reshape(-1, 1)
    y_train = train_df["label"].to_numpy(dtype=np.int32)
    X_test = test_df["cosine"].to_numpy(dtype=np.float64).reshape(-1, 1)
    y_test = test_df["label"].to_numpy(dtype=np.int32)
    cosine_train = train_df["cosine"].to_numpy(dtype=np.float64)
    cosine_test = test_df["cosine"].to_numpy(dtype=np.float64)

    logistic_model = train_logistic_regression(X_train, y_train, seed=args.seed)
    isotonic_model = train_isotonic_regression(cosine_train, y_train)

    logistic_train_probs = predict_logistic(logistic_model, X_train)
    logistic_test_probs = predict_logistic(logistic_model, X_test)
    isotonic_train_probs = predict_isotonic(isotonic_model, cosine_train)
    isotonic_test_probs = predict_isotonic(isotonic_model, cosine_test)

    logistic_train_metrics = compute_metrics(y_train, logistic_train_probs, n_bins=args.n_bins)
    logistic_test_metrics = compute_metrics(y_test, logistic_test_probs, n_bins=args.n_bins)
    isotonic_train_metrics = compute_metrics(y_train, isotonic_train_probs, n_bins=args.n_bins)
    isotonic_test_metrics = compute_metrics(y_test, isotonic_test_probs, n_bins=args.n_bins)

    lr_best_prob_thresh, _ = find_best_probability_threshold(y_train, logistic_train_probs)
    ir_best_prob_thresh, _ = find_best_probability_threshold(y_train, isotonic_train_probs)
    lr_test_at_best_thresh = _classification_metrics_at_threshold(
        y_test, logistic_test_probs, lr_best_prob_thresh
    )
    ir_test_at_best_thresh = _classification_metrics_at_threshold(
        y_test, isotonic_test_probs, ir_best_prob_thresh
    )

    best_cosine_threshold, _ = find_best_cosine_threshold(y_train, cosine_train, num_points=200)
    raw_train_at_best_cosine = _classification_metrics_at_threshold(
        y_train, cosine_train, best_cosine_threshold
    )
    raw_test_at_best_cosine = _classification_metrics_at_threshold(
        y_test, cosine_test, best_cosine_threshold
    )

    logistic_coef = float(logistic_model.coef_[0][0])
    logistic_intercept = float(logistic_model.intercept_[0])

    logistic_section = {
        "coef": logistic_coef,
        "intercept": logistic_intercept,
        "formula": (
            f"P(same|s)=1/(1+exp(-({logistic_coef:.12f}*s+{logistic_intercept:.12f})))"
        ),
        "train_metrics": logistic_train_metrics,
        "test_metrics": logistic_test_metrics,
        "best_probability_threshold_by_train_f1": float(lr_best_prob_thresh),
        "test_metrics_at_best_probability_threshold": lr_test_at_best_thresh,
    }
    isotonic_section = {
        "x_thresholds": [float(v) for v in isotonic_model.X_thresholds_],
        "y_thresholds": [float(v) for v in isotonic_model.y_thresholds_],
        "train_metrics": isotonic_train_metrics,
        "test_metrics": isotonic_test_metrics,
        "best_probability_threshold_by_train_f1": float(ir_best_prob_thresh),
        "test_metrics_at_best_probability_threshold": ir_test_at_best_thresh,
    }

    result = {
        "data": _build_data_summary(
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            used_internal_split=used_internal_split,
            train_df=train_df,
            test_df=test_df,
        ),
        "models": {
            "logistic_regression": logistic_section,
            "isotonic_regression": isotonic_section,
        },
        "raw_cosine_baseline": {
            "best_cosine_threshold_by_train_f1": float(best_cosine_threshold),
            "train_metrics_at_best_cosine_threshold": raw_train_at_best_cosine,
            "test_metrics_at_best_cosine_threshold": raw_test_at_best_cosine,
        },
        "recommendation": _compare_models_for_recommendation(logistic_section, isotonic_section),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(_to_python(result), f, ensure_ascii=False, indent=2)

    print(f"train size: {result['data']['train_size']}")
    print(f"test size: {result['data']['test_size']}")
    print(
        "logistic test brier/ece/f1: "
        f"{logistic_test_metrics['brier_score']:.6f} / "
        f"{logistic_test_metrics['ece']:.6f} / "
        f"{logistic_test_metrics['f1_at_0_5']:.6f}"
    )
    print(
        "isotonic test brier/ece/f1: "
        f"{isotonic_test_metrics['brier_score']:.6f} / "
        f"{isotonic_test_metrics['ece']:.6f} / "
        f"{isotonic_test_metrics['f1_at_0_5']:.6f}"
    )
    print(f"raw cosine best threshold: {best_cosine_threshold:.6f}")
    print(f"output json path: {args.output_json}")


if __name__ == "__main__":
    main()
