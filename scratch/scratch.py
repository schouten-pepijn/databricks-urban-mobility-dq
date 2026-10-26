import argparse
import json
import os
from datetime import UTC, datetime
from hashlib import sha256
from itertools import pairwise
from pathlib import Path

import polars as pl

from urban_mobility_dq.contracts import validate_file_schema

EXPECTED_HOURLY_ROWS = 768


def sha256_file(path: Path) -> str:
    """Return the uppercase SHA-256 digest of a file, reading it in chunks."""
    digest = sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def source_record(
    manifest: dict[str, object],
    source_name: str,
) -> tuple[str, int, str]:
    """Read and validate the file name, byte count, and digest from a manifest."""
    record = manifest.get(source_name)
    if not isinstance(record, dict):
        raise TypeError(f"Manifest '{source_name}' record must be an object.")

    file_name = record.get("file_name")
    byte_count = record.get("bytes")
    expected_sha256 = record.get("sha256")

    if not isinstance(file_name, str):
        raise TypeError(f"Manifest '{source_name}.file_name' must be a string.")
    if not isinstance(byte_count, int) or isinstance(byte_count, bool):
        raise TypeError(f"Manifest '{source_name}.bytes' must be an integer.")
    if not isinstance(expected_sha256, str):
        raise TypeError(f"Manifest '{source_name}.sha256' must be a string.")

    return file_name, byte_count, expected_sha256.upper()


def verify_source_file(
    path: Path,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    """Fail if a local source file differs from its manifest entry."""
    if not path.is_file():
        raise FileNotFoundError(f"Source file does not exist: {path}")

    actual_bytes = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
        raise ValueError(f"Source file does not match the manifest: {path.name}")

    print(
        {
            "source_file": path.name,
            "bytes": actual_bytes,
            "sha256": actual_sha256,
            "matches_manifest": True,
        }
    )


def validate_weather_json(path: Path) -> None:
    """Validate the saved hourly weather response and its UTC-hour sequence."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Weather response must be a JSON object.")
    if payload.get("error", False) is not False:
        raise ValueError(f"Open-Meteo returned an error: {payload.get('reason')}")

    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise TypeError("Weather response 'hourly' value must be an object.")

    times = hourly.get("time")
    temperatures = hourly.get("temperature_2m")
    precipitation = hourly.get("precipitation")
    if not all(
        isinstance(values, list) for values in (times, temperatures, precipitation)
    ):
        raise ValueError("Weather hourly values must be arrays.")

    array_lengths = (len(times), len(temperatures), len(precipitation))
    if len(set(array_lengths)) != 1:
        raise ValueError(f"Weather arrays have different lengths: {array_lengths}")
    if array_lengths[0] != EXPECTED_HOURLY_ROWS:
        raise ValueError(f"Expected 768 weather hours, found {array_lengths[0]}.")

    if any(not isinstance(value, int) or isinstance(value, bool) for value in times):
        raise ValueError("Weather times must be Unix epoch integers.")

    epochs = times
    if len(set(epochs)) != len(epochs):
        raise ValueError("Weather contains duplicate timestamps.")
    if any(value % 3600 != 0 for value in epochs):
        raise ValueError("Weather timestamps must align to whole UTC hours.")
    if any(later - earlier != 3600 for earlier, later in pairwise(epochs)):
        raise ValueError("Weather timestamps are not contiguous hourly values.")

    expected_first = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp())
    expected_last = int(datetime(2024, 2, 1, 23, tzinfo=UTC).timestamp())
    if epochs[0] != expected_first or epochs[-1] != expected_last:
        raise ValueError(
            "Weather timestamp range does not match the requested UTC period."
        )

    units = payload.get("hourly_units")
    expected_units = {
        "time": "unixtime",
        "temperature_2m": "°C",
        "precipitation": "mm",
    }
    if not isinstance(units, dict) or any(
        units.get(name) != expected for name, expected in expected_units.items()
    ):
        raise ValueError(f"Weather units differ from the expected units: {units}")

    print(
        {
            "weather_hours": array_lengths[0],
            "first_utc": datetime.fromtimestamp(epochs[0], UTC).isoformat(),
            "last_utc": datetime.fromtimestamp(epochs[-1], UTC).isoformat(),
            "units": units,
        }
    )


def create_taxi_fixtures(source_path: Path, fixture_dir: Path) -> None:
    """Write a first-100-row fixture and a local January 1 pickup slice."""
    first_100_path = fixture_dir / "yellow_tripdata_2024-01_first_100.parquet"
    january_1_path = fixture_dir / "yellow_tripdata_2024-01_pickup_2024-01-01.parquet"

    existing = [path.name for path in (first_100_path, january_1_path) if path.exists()]
    if existing:
        # Keep previously created outputs safe; inspect them before replacing anything.
        raise FileExistsError(
            f"Derived fixture already exists; inspect before replacing: {existing}"
        )

    fixture_dir.mkdir(parents=True, exist_ok=True)
    trips = pl.scan_parquet(source_path)
    trips.head(100).sink_parquet(first_100_path)

    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 2)
    january_1_trips = trips.filter(
        (pl.col("tpep_pickup_datetime") >= start)
        & (pl.col("tpep_pickup_datetime") < end)
    )
    january_1_trips.sink_parquet(january_1_path)

    first_100_rows = pl.scan_parquet(first_100_path).select(pl.len()).collect().item()
    day_summary = (
        pl.scan_parquet(january_1_path)
        .select(
            pl.len().alias("rows"),
            pl.col("tpep_pickup_datetime").min().alias("first_pickup_local"),
            pl.col("tpep_pickup_datetime").max().alias("last_pickup_local"),
        )
        .collect()
        .row(0, named=True)
    )

    if first_100_rows != 100:
        raise ValueError(f"Expected 100 rows in the fixture, found {first_100_rows}.")
    if day_summary["rows"] == 0:
        raise ValueError("The January 1 source slice is empty.")
    first_pickup = day_summary["first_pickup_local"]
    last_pickup = day_summary["last_pickup_local"]
    if first_pickup is None or last_pickup is None:
        raise ValueError("The January 1 fixture has no pickup timestamps.")
    if first_pickup < start or last_pickup >= end:
        raise ValueError("The January 1 fixture contains pickups outside that day.")

    print(
        {
            "fixture": first_100_path.name,
            "rows": first_100_rows,
            "bytes": first_100_path.stat().st_size,
        }
    )
    print(
        {
            "fixture": january_1_path.name,
            "rows": day_summary["rows"],
            "bytes": january_1_path.stat().st_size,
            "first_pickup_local": day_summary["first_pickup_local"],
            "last_pickup_local": day_summary["last_pickup_local"],
        }
    )


def main(manifest_path: Path | None = None) -> None:
    """Validate frozen source inputs and create local source-derived fixtures."""
    if manifest_path is None:
        manifest_path = Path(__file__).resolve().parents[1] / "source_manifest.json"

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_data, dict):
        raise TypeError("Source manifest must be a JSON object.")
    if manifest_data.get("source_month") != "2024-01":
        raise ValueError("This scratch script expects source_month 2024-01.")

    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise RuntimeError("LOCALAPPDATA is not set.")

    source_dir = Path(local_app_data) / "urban-mobility-dq" / "source" / "2024-01"
    taxi_name, taxi_bytes, taxi_sha256 = source_record(manifest_data, "taxi")
    weather_name, weather_bytes, weather_sha256 = source_record(
        manifest_data, "weather"
    )

    taxi_path = source_dir / taxi_name
    weather_path = source_dir / weather_name
    # Reject changed or incomplete source files before inspecting or transforming data.
    verify_source_file(taxi_path, taxi_bytes, taxi_sha256)
    verify_source_file(weather_path, weather_bytes, weather_sha256)

    # The contract path is based on the repository, not the shell's current directory.
    contract_path = Path(__file__).resolve().parents[1] / "contracts" / "taxi_v1.yaml"
    # File-level schema errors stop here; new non-required columns are reported.
    extra_columns = validate_file_schema(taxi_path, contract_path)
    print(
        {
            "taxi_schema": "valid",
            "taxi_extra_columns": extra_columns,
        }
    )
    validate_weather_json(weather_path)
    create_taxi_fixtures(taxi_path, source_dir / "fixtures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to a source manifest; defaults to source_manifest.json",
    )
    arguments = parser.parse_args()
    main(arguments.manifest)
