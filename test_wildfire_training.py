"""Small integration check for the monthly logistic training workflow."""

from __future__ import annotations

import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import train_wildfire_logistic as training
from build_wildfire_dataset import combine_partitions


class LogisticTrainingTest(unittest.TestCase):
    def test_time_split_and_missing_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assert root.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())
            dataset = root / "dataset"
            for month in range(5, 9):
                rng = np.random.default_rng(10 + month)
                count = 200
                data = {name: rng.normal(size=count) for name in training.NUMERIC
                        if name not in ("road_distance_m", "stream_distance_m")}
                frame = pd.DataFrame(data)
                frame["ndvi_prior"] = rng.uniform(0, 1, count)
                frame.loc[:9, "ndvi_prior"] = np.nan
                frame["land_cover_previous_year"] = rng.choice([41, 42, 71], count)
                frame["cell_id"] = np.arange(count, dtype=np.int64)
                frame["month"] = f"2024-{month:02d}"
                frame["wildfire_recorded"] = (np.arange(count) % 7 == 0).astype(np.uint8)
                target = dataset / "year=2024" / f"month_num={month:02d}"
                target.mkdir(parents=True)
                frame.to_parquet(target / "data.parquet", index=False)
            pd.DataFrame({"cell_id": np.arange(count, dtype=np.int64)}).to_parquet(
                dataset / "_cells.parquet", index=False)
            (dataset / "_dataset_metadata.json").write_text(json.dumps({
                "months_present": [f"2024-{month:02d}" for month in range(5, 9)],
                "grid_crs": "EPSG:5070", "cell_width_m": 3218.688,
                "proximity_skipped": True,
            }), encoding="utf-8")
            single_file = root / "dataset.parquet"
            combine_partitions(dataset, single_file)
            self.assertEqual(len(training.partitions(single_file)), 4)
            args = SimpleNamespace(
                dataset=dataset, output=root / "model_drop",
                train_end=dt.date(2024, 7, 1), test_start=dt.date(2024, 8, 1),
                missing_strategy="drop", exclude_proximity=True,
                max_negative_rows=150, threshold=0.5, C=1.0, max_iter=200,
                seed=449,
            )
            result = training.run(args)
            self.assertEqual(result["train_rows_after_filter"], 570)
            self.assertEqual(result["test_rows_after_filter"], 190)
            self.assertEqual(result["train_rows_dropped_for_missing"], 30)
            self.assertTrue((args.output / "model.joblib").is_file())
            self.assertTrue((args.output / "coefficients.csv").is_file())
            self.assertEqual(len(pd.read_parquet(args.output / "predictions")), 190)
            self.assertGreater(result["test_positive_cells"], 0)
            self.assertLess(result["test_positive_cells"], 190)

            imputed_args = SimpleNamespace(**vars(args))
            imputed_args.output = root / "model_impute"
            imputed_args.missing_strategy = "impute"
            imputed = training.run(imputed_args)
            self.assertEqual(imputed["train_rows_after_filter"], 600)
            self.assertEqual(imputed["test_rows_after_filter"], 200)
            self.assertEqual(imputed["train_rows_dropped_for_missing"], 0)

            single_args = SimpleNamespace(**vars(args))
            single_args.dataset = single_file
            single_args.output = root / "model_single_file"
            single = training.run(single_args)
            self.assertEqual(single["train_rows_after_filter"], result["train_rows_after_filter"])
            self.assertEqual(single["test_rows_after_filter"], result["test_rows_after_filter"])
            self.assertEqual(single["test_positive_cells"], result["test_positive_cells"])

            single_imputed_args = SimpleNamespace(**vars(imputed_args))
            single_imputed_args.dataset = single_file
            single_imputed_args.output = root / "model_single_file_impute"
            single_imputed = training.run(single_imputed_args)
            self.assertEqual(single_imputed["train_rows_after_filter"],
                             imputed["train_rows_after_filter"])


if __name__ == "__main__":
    unittest.main()
