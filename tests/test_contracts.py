from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from urban_mobility_dq.contracts import validate_file_schema

CONTRACT_PATH = Path(__file__).resolve().parents[1] / "contracts" / "taxi_v1.yaml"


def write_valid_parquet(path: Path) -> Path:
    """Write one row matching the contract, plus one extra source column."""
    frame = pl.DataFrame(
        {
            "tpep_pickup_datetime": [datetime(2024, 1, 1, 12, 0)],
            "tpep_dropoff_datetime": [datetime(2024, 1, 1, 12, 15)],
            "PULocationID": [161],
            "DOLocationID": [236],
            "trip_distance": [2.5],
            "total_amount": [18.5],
            "extra_source_column": ["keep but report"],
        },
        schema={
            "tpep_pickup_datetime": pl.Datetime("ns", time_zone=None),
            "tpep_dropoff_datetime": pl.Datetime("ns", time_zone=None),
            "PULocationID": pl.Int32,
            "DOLocationID": pl.Int32,
            "trip_distance": pl.Float64,
            "total_amount": pl.Float64,
            "extra_source_column": pl.String,
        },
    )
    frame.write_parquet(path)
    return path


def test_valid_schema_returns_extra_columns(tmp_path: Path) -> None:
    parquet_path = write_valid_parquet(tmp_path / "valid.parquet")

    extra_columns = validate_file_schema(parquet_path, CONTRACT_PATH)

    assert extra_columns == ["extra_source_column"]


def test_missing_required_column_fails(tmp_path: Path) -> None:
    parquet_path = write_valid_parquet(tmp_path / "missing-column.parquet")
    frame = pl.read_parquet(parquet_path).drop("trip_distance")
    frame.write_parquet(parquet_path)

    with pytest.raises(ValueError, match="missing_required_columns"):
        validate_file_schema(parquet_path, CONTRACT_PATH)


def test_incompatible_column_type_fails(tmp_path: Path) -> None:
    parquet_path = write_valid_parquet(tmp_path / "wrong-type.parquet")
    frame = pl.read_parquet(parquet_path).with_columns(
        pl.col("PULocationID").cast(pl.Int64)
    )
    frame.write_parquet(parquet_path)

    with pytest.raises(ValueError, match="incompatible_column_types"):
        validate_file_schema(parquet_path, CONTRACT_PATH)


def test_unknown_type_label_fails(tmp_path: Path) -> None:
    parquet_path = write_valid_parquet(tmp_path / "unknown-type.parquet")
    bad_contract = tmp_path / "unknown-type.yaml"
    contract_text = CONTRACT_PATH.read_text(encoding="utf-8")
    bad_contract.write_text(
        contract_text.replace("int32", "unknown_integer"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown type labels"):
        validate_file_schema(parquet_path, bad_contract)


def test_empty_required_columns_fails(tmp_path: Path) -> None:
    parquet_path = write_valid_parquet(tmp_path / "empty-contract.parquet")
    empty_contract = tmp_path / "empty-contract.yaml"
    empty_contract.write_text(
        "required_columns: []\nexpected_types: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least one"):
        validate_file_schema(parquet_path, empty_contract)
