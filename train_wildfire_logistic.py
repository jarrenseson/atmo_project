"""Train a wildfire logistic regressor from one Parquet file or monthly parts.

The split is by month: all training months precede all test months. Numeric
imputation, scaling, and land-cover encoding are fitted only on training rows.
The default drops rows with any missing feature. Large training sets keep all
fire cells and sample negative cells with inverse-sampling weights.

Example::

    python train_wildfire_logistic.py --dataset wildfire_dataset.parquet \
        --train-end 2024-08 --test-start 2024-09 --output wildfire_logistic_2024
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

LOCAL_DEPS = Path(__file__).resolve().parent / ".deps"
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import wildfire_dataset_io as dataset_io


NUMERIC = [
    "tmax_c", "tmin_c", "precip_mm", "wind_m_s", "ndvi_prior",
    "ndvi_age_days", "elevation_m", "slope_deg", "aspect_sin",
    "aspect_cos", "curvature_per_m", "road_distance_m", "stream_distance_m",
]
CATEGORICAL = ["land_cover_previous_year"]
LABEL = "wildfire_recorded"


def parse_month(text: str) -> dt.date:
    try:
        return dt.date.fromisoformat(text + "-01")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use YYYY-MM") from exc


def partitions(dataset: Path) -> list[tuple[dt.date, Path]]:
    return dataset_io.months(dataset)


def features(exclude_proximity: bool) -> tuple[list[str], list[str]]:
    numeric = [name for name in NUMERIC if not (exclude_proximity and name in
               ("road_distance_m", "stream_distance_m"))]
    return numeric, CATEGORICAL


def load_month(path: Path, period: dt.date, numeric: list[str], categorical: list[str],
               missing_strategy: str) -> tuple[pd.DataFrame, int]:
    names = ["cell_id", "month", LABEL] + numeric + categorical
    frame = dataset_io.read_month(path, period, columns=names)
    if frame["cell_id"].duplicated().any() or not frame["month"].eq(period.strftime("%Y-%m")).all():
        raise ValueError(f"Invalid cell/month key in {path}")
    if not frame[LABEL].isin((0, 1)).all():
        raise ValueError(f"Target must be 0 or 1 in {path}")
    for name in numeric + categorical:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame.loc[frame[categorical[0]] == 0, categorical[0]] = np.nan
    angle = 2 * math.pi * (period.month - 1) / 12
    frame["month_sin"] = np.float32(math.sin(angle))
    frame["month_cos"] = np.float32(math.cos(angle))
    dropped = 0
    if missing_strategy == "drop":
        before = len(frame)
        frame = frame.dropna(subset=numeric + categorical)
        dropped = before - len(frame)
    return frame, dropped


def make_model(numeric: list[str], categorical: list[str], impute: bool,
               max_iter: int, regularization: float) -> Pipeline:
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=impute)),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transform = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric + ["month_sin", "month_cos"]),
        ("land_cover", categorical_pipeline, categorical),
    ], sparse_threshold=0)
    return Pipeline([
        ("prepare", transform),
        ("regressor", LogisticRegression(solver="lbfgs", C=regularization,
                                         max_iter=max_iter)),
    ])


def run(args) -> dict:
    dataset = args.dataset.resolve()
    found = partitions(dataset)
    if not found:
        raise FileNotFoundError(f"No complete wildfire months in {dataset}")
    if args.train_end >= args.test_start:
        raise ValueError("--train-end must be earlier than --test-start")
    train_parts = [(month, path) for month, path in found if month <= args.train_end]
    test_parts = [(month, path) for month, path in found if month >= args.test_start]
    if len(train_parts) < 3 or not test_parts:
        raise ValueError("Need at least three training months and one later test month")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output is not empty: {args.output}; choose a new folder")
    numeric, categorical = features(args.exclude_proximity)
    train_rows = train_positives = train_negatives = train_dropped = 0
    for month, path in train_parts:
        frame, dropped = load_month(path, month, numeric, categorical,
                                    args.missing_strategy)
        train_rows += len(frame)
        train_positives += int(frame[LABEL].sum())
        train_negatives += len(frame) - int(frame[LABEL].sum())
        train_dropped += dropped
    if train_positives < 2 or train_negatives < 2:
        raise ValueError(f"Training data need both classes; found {train_positives} fire cells and {train_negatives} other cells")
    if args.missing_strategy == "impute":
        all_empty = []
        for name in numeric + categorical:
            nonmissing = 0
            for month, path in train_parts:
                frame = dataset_io.read_month(path, month, columns=[name])
                nonmissing += int(frame[name].notna().sum())
            if nonmissing == 0:
                all_empty.append(name)
        if all_empty:
            raise ValueError(f"Entire training features are missing: {all_empty}. Finish those source files or use --exclude-proximity.")
    fraction = min(1.0, args.max_negative_rows / train_negatives)
    selected = []
    weights = []
    for month, path in train_parts:
        frame, _ = load_month(path, month, numeric, categorical,
                              args.missing_strategy)
        rng = np.random.default_rng(args.seed + month.year * 12 + month.month)
        label = frame[LABEL].to_numpy()
        keep = (label == 1) | ((label == 0) & (rng.random(len(frame)) < fraction))
        kept = frame.loc[keep]
        selected.append(kept)
        weights.append(np.where(kept[LABEL].to_numpy() == 1, 1.0, 1.0 / fraction))
        print(f"Train {month:%Y-%m}: kept {len(kept):,}/{len(frame):,} rows", flush=True)
    train = pd.concat(selected, ignore_index=True)
    sample_weight = np.concatenate(weights)
    sample_weight *= len(sample_weight) / sample_weight.sum()
    model = make_model(numeric, categorical, args.missing_strategy == "impute",
                       args.max_iter, args.C)
    predictors = numeric + categorical + ["month_sin", "month_cos"]
    model.fit(train[predictors], train[LABEL].to_numpy(),
              regressor__sample_weight=sample_weight)
    del train, selected, weights
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output / "model.joblib")
    names = model.named_steps["prepare"].get_feature_names_out()
    coefficients = pd.DataFrame({"feature": names,
                                 "coefficient": model.named_steps["regressor"].coef_[0]})
    coefficients.to_csv(args.output / "coefficients.csv", index=False)
    y_parts = []
    probability_parts = []
    test_dropped = 0
    month_results = []
    for month, path in test_parts:
        frame, dropped = load_month(path, month, numeric, categorical,
                                    args.missing_strategy)
        test_dropped += dropped
        if frame.empty:
            continue
        probability = model.predict_proba(frame[predictors])[:, 1]
        y = frame[LABEL].to_numpy().astype(np.uint8)
        y_parts.append(y)
        probability_parts.append(probability)
        prediction = pd.DataFrame({
            "cell_id": frame.cell_id.to_numpy(), "month": month.strftime("%Y-%m"),
            "wildfire_recorded": y, "fire_probability": probability.astype(np.float32),
        })
        target = args.output / "predictions" / f"year={month.year}" / f"month_num={month.month:02d}"
        target.mkdir(parents=True, exist_ok=True)
        prediction.to_parquet(target / "data.parquet", index=False, compression="zstd")
        month_results.append({"month": month.strftime("%Y-%m"), "rows": len(frame),
                              "fires": int(y.sum()), "dropped": dropped})
        print(f"Test {month:%Y-%m}: {len(frame):,} rows, {int(y.sum()):,} positive", flush=True)
    if not y_parts:
        raise ValueError("No complete rows in the test months")
    y = np.concatenate(y_parts)
    probability = np.concatenate(probability_parts)
    if len(np.unique(y)) < 2:
        raise ValueError("Test period needs both classes for evaluation")
    binary = probability >= args.threshold
    matrix = confusion_matrix(y, binary, labels=[0, 1])
    metrics = {
        "dataset": str(dataset), "train_months": [str(x[0])[:7] for x in train_parts],
        "test_months": [str(x[0])[:7] for x in test_parts],
        "missing_strategy": args.missing_strategy,
        "exclude_proximity": args.exclude_proximity,
        "features": predictors, "train_rows_after_filter": train_rows,
        "train_positive_cells": train_positives,
        "train_negative_cells": train_negatives,
        "train_rows_dropped_for_missing": train_dropped,
        "training_rows_sampled": len(sample_weight),
        "negative_sampling_fraction": fraction,
        "test_rows_after_filter": len(y), "test_positive_cells": int(y.sum()),
        "test_rows_dropped_for_missing": test_dropped,
        "test_prevalence": float(y.mean()),
        "threshold": args.threshold,
        "precision": float(precision_score(y, binary, zero_division=0)),
        "recall": float(recall_score(y, binary, zero_division=0)),
        "f1": float(f1_score(y, binary, zero_division=0)),
        "average_precision": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "confusion_matrix_tn_fp_fn_tp": [int(matrix[0, 0]), int(matrix[0, 1]),
                                       int(matrix[1, 0]), int(matrix[1, 1])],
        "per_test_month": month_results,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps({key: metrics[key] for key in
                      ("test_rows_after_filter", "test_positive_cells", "precision",
                       "recall", "f1", "average_precision", "roc_auc")}, indent=2), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("wildfire_dataset.parquet"))
    parser.add_argument("--output", type=Path, default=Path("wildfire_logistic"))
    parser.add_argument("--train-end", type=parse_month, required=True,
                        help="Last training month, YYYY-MM")
    parser.add_argument("--test-start", type=parse_month, required=True,
                        help="First later test month, YYYY-MM")
    parser.add_argument("--missing-strategy", choices=("drop", "impute"), default="drop")
    parser.add_argument("--exclude-proximity", action="store_true",
                        help="Pilot only: omit road and stream distances")
    parser.add_argument("--max-negative-rows", type=int, default=250_000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--C", type=float, default=1.0, help="Inverse L2 regularization strength")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=449)
    args = parser.parse_args()
    if args.max_negative_rows < 1 or args.max_iter < 1 or args.C <= 0:
        parser.error("--max-negative-rows, --max-iter, and --C must be positive")
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")
    try:
        run(args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
