from pathlib import Path

import numpy as np
import pandas as pd

from heat_mnch.round10_5_flagship_execution import (
    GESTATIONAL_MART_NAME,
    NEONATAL_MART_NAME,
    ROUND10_5_PRETERM_ENGINE_MEMO_PATH,
    STILLBIRTH_MART_NAME,
    _irls_poisson,
    _non_missing_exposure_rows_many,
    _parquet_dataset_shape,
    _preterm_engine_validation,
    _primary_exposure_column_for_mart,
    _state_month_partition_path,
)


def test_state_month_partition_path_encodes_year_state_and_month() -> None:
    root = Path("/tmp/heat_mnch")
    path = _state_month_partition_path(root, GESTATIONAL_MART_NAME, 2024, 3, "35")

    assert str(path).endswith(
        "artifacts/analysis/shared_marts/"
        "flagship_gestational_weekly_person_period_mart/"
        "YEAR=2024/"
        "flagship_gestational_weekly_person_period_mart_2024_UF35_M03.parquet"
    )


def test_primary_exposure_column_mapping_is_frozen() -> None:
    assert _primary_exposure_column_for_mart(GESTATIONAL_MART_NAME) == "tmax_weekly_mean"
    assert _primary_exposure_column_for_mart(STILLBIRTH_MART_NAME) == "tmax_weekly_mean"
    assert _primary_exposure_column_for_mart(NEONATAL_MART_NAME) == "tmax_exposure_value"


def test_irls_poisson_uses_builder_supplied_column_names() -> None:
    batch = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0], "offset": [0.0, 0.0, 0.0]})

    def batch_factory():
        yield batch

    def builder(frame: pd.DataFrame):
        X = np.column_stack([np.ones(len(frame)), frame["x"].to_numpy()])
        y = frame["y"].to_numpy()
        offset = frame["offset"].to_numpy()
        return X, y, offset, ["intercept", "x"]

    fit = _irls_poisson(batch_factory=batch_factory, design_builder=builder, max_iter=5)

    assert fit["column_names"] == ["intercept", "x"]
    assert fit["model_rows"] == 3


def test_preterm_engine_validation_memo_records_block(tmp_path: Path) -> None:
    memo_path = _preterm_engine_validation(tmp_path)
    text = memo_path.read_text(encoding="utf-8")

    assert memo_path == tmp_path / ROUND10_5_PRETERM_ENGINE_MEMO_PATH
    assert "blocked_primary_engine_not_operational_under_round5_freeze" in text
    assert "No approved alternative mixed-effects `cloglog` engine is currently" in text


def test_parquet_dataset_helpers_roll_up_multiple_files(tmp_path: Path) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pd.DataFrame({"tmax_weekly_mean": [1.0, 2.0], "other_value": [10, 20]}).to_parquet(first, index=False)
    pd.DataFrame({"tmax_weekly_mean": [3.0], "other_value": [30]}).to_parquet(second, index=False)

    row_count, column_count, columns = _parquet_dataset_shape([first, second])
    non_missing = _non_missing_exposure_rows_many([first, second], "tmax_weekly_mean")

    assert row_count == 3
    assert column_count == 2
    assert columns == ["tmax_weekly_mean", "other_value"]
    assert non_missing == 3
