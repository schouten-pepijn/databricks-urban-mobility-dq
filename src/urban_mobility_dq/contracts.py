from collections.abc import Mapping
from pathlib import Path

import polars as pl
import yaml

_POLARS_TYPES: Mapping[str, pl.DataType] = {
    # These short labels keep Polars-specific type syntax out of the YAML file.
    "datetime_ns_naive": pl.Datetime("ns", time_zone=None),
    "int32": pl.Int32(),
    "float64": pl.Float64(),
}


def validate_file_schema(
    parquet_path: Path,
    contract_path: Path,
) -> list[str]:
    """Validate required Parquet columns and types against a YAML contract."""

    # safe_load reads plain YAML data without constructing arbitrary Python objects.
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))

    if not isinstance(contract, dict):
        raise TypeError("Contract must be a YAML mapping.")

    required_columns = contract.get("required_columns")
    if not required_columns:
        raise ValueError("Contract must require at least one column.")

    expected_types = contract.get("expected_types")

    if not isinstance(required_columns, list) or not all(
        isinstance(name, str) for name in required_columns
    ):
        raise TypeError("'required_columns' must be a list of strings.")

    if len(required_columns) != len(set(required_columns)):
        raise ValueError("Duplicate required columns in contract.")

    if not isinstance(expected_types, dict) or not all(
        isinstance(name, str) and isinstance(type_name, str)
        for name, type_name in expected_types.items()
    ):
        raise TypeError("'expected_types' must map strings to strings.")

    # Each required field must have one type rule, so no field escapes type checking.
    if set(expected_types) != set(required_columns):
        raise ValueError("Every required column must have exactly one expected type.")

    if unknown_types := set(expected_types.values()) - _POLARS_TYPES.keys():
        raise ValueError(f"Unknown type labels: {sorted(unknown_types)}")

    # Inspect Parquet metadata first; this does not collect the taxi rows.
    schema = pl.scan_parquet(parquet_path).collect_schema()
    schema_names = set(schema.names())

    # A missing required field makes the whole file incompatible with this contract.
    if missing_columns := set(required_columns) - schema_names:
        raise ValueError(f"missing_required_columns={sorted(missing_columns)}")

    incompatible_types = {
        name: {
            "expected": str(_POLARS_TYPES[expected_types[name]]),
            "actual": str(schema[name]),
        }
        for name in required_columns
        if schema[name] != _POLARS_TYPES[expected_types[name]]
    }
    if incompatible_types:
        raise ValueError(f"incompatible_column_types={incompatible_types}")

    # Extra source fields are returned for reporting; they do not fail this version.
    return sorted(schema_names - set(required_columns))
