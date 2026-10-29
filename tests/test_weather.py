import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from urban_mobility_dq.weather import join_taxi_weather, load_weather


def test_load_weather_accepts_degree_celsius_and_converts_epoch_to_utc(
    tmp_path: Path,
) -> None:
    payload = {
        "hourly": {
            "time": [1704067200, 1704070800],
            "temperature_2m": [3.0, 3.5],
            "precipitation": [0.0, 0.2],
        },
        "hourly_units": {
            "time": "unixtime",
            "temperature_2m": "°C",
            "precipitation": "mm",
        },
    }
    weather_path = tmp_path / "weather.json"
    weather_path.write_text(json.dumps(payload), encoding="utf-8")

    weather = load_weather(weather_path)

    assert weather.schema == {
        "weather_hour_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
        "temperature_2m": pl.Float64,
        "precipitation": pl.Float64,
    }
    assert weather.get_column("weather_hour_utc").cast(pl.Int64).to_list() == [
        1704067200000000,
        1704070800000000,
    ]
    assert weather.get_column("temperature_2m").to_list() == [3.0, 3.5]
    assert weather.get_column("precipitation").to_list() == [0.0, 0.2]


def test_load_weather_rejects_different_array_lengths(tmp_path: Path) -> None:
    payload = {
        "hourly": {
            "time": [1704067200, 1704070800],
            "temperature_2m": [3.0],
            "precipitation": [0.0, 0.2],
        },
        "hourly_units": {
            "time": "unixtime",
            "temperature_2m": "°C",
            "precipitation": "mm",
        },
    }
    weather_path = tmp_path / "unequal-arrays.json"
    weather_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="different lengths"):
        load_weather(weather_path)


def test_load_weather_rejects_duplicate_timestamps(tmp_path: Path) -> None:
    payload = {
        "hourly": {
            "time": [1704067200, 1704067200],
            "temperature_2m": [3.0, 3.5],
            "precipitation": [0.0, 0.2],
        },
        "hourly_units": {
            "time": "unixtime",
            "temperature_2m": "°C",
            "precipitation": "mm",
        },
    }
    weather_path = tmp_path / "duplicate-hours.json"
    weather_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate timestamps"):
        load_weather(weather_path)


def test_load_weather_rejects_non_hourly_timestamp(tmp_path: Path) -> None:
    payload = {
        "hourly": {
            "time": [1704067201],
            "temperature_2m": [3.0],
            "precipitation": [0.0],
        },
        "hourly_units": {
            "time": "unixtime",
            "temperature_2m": "°C",
            "precipitation": "mm",
        },
    }
    weather_path = tmp_path / "non-hourly.json"
    weather_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="whole UTC hours"):
        load_weather(weather_path)


def test_load_weather_rejects_unexpected_temperature_unit(
    tmp_path: Path,
) -> None:
    payload = {
        "hourly": {
            "time": [1704067200],
            "temperature_2m": [3.0],
            "precipitation": [0.0],
        },
        "hourly_units": {
            "time": "unixtime",
            "temperature_2m": "C",
            "precipitation": "mm",
        },
    }
    weather_path = tmp_path / "wrong-temperature-unit.json"
    weather_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unexpected units"):
        load_weather(weather_path)


def test_load_weather_rejects_api_error(tmp_path: Path) -> None:
    payload = {
        "error": True,
        "reason": "Invalid date range",
    }
    weather_path = tmp_path / "api-error.json"
    weather_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Open-Meteo returned an error"):
        load_weather(weather_path)


def test_join_taxi_weather_preserves_rows_when_an_hour_is_missing() -> None:
    taxi = pl.DataFrame(
        {
            "trip_key": ["matched", "unmatched"],
            "pickup_utc": [
                datetime(2024, 1, 1, 0, 30, tzinfo=UTC),
                datetime(2024, 1, 1, 1, 0, tzinfo=UTC),
            ],
        }
    )
    weather = pl.DataFrame(
        {
            "weather_hour_utc": [datetime(2024, 1, 1, 0, 0, tzinfo=UTC)],
            "temperature_2m": [3.0],
            "precipitation": [0.0],
        }
    )

    joined = join_taxi_weather(taxi.lazy(), weather).collect()

    assert joined.height == taxi.height
    assert joined.get_column("weather_match").to_list() == [True, False]
    assert joined.get_column("temperature_2m").to_list() == [3.0, None]


def test_join_taxi_weather_rejects_duplicate_weather_hours() -> None:
    taxi = pl.DataFrame(
        {"pickup_utc": [datetime(2024, 1, 1, 0, 30, tzinfo=UTC)]}
    ).lazy()
    weather = pl.DataFrame(
        {
            "weather_hour_utc": [
                datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            ],
            "temperature_2m": [3.0, 3.1],
            "precipitation": [0.0, 0.1],
        }
    )

    with pytest.raises(ValueError, match="duplicate timestamps"):
        join_taxi_weather(taxi, weather)


def test_january_local_pickup_joins_to_february_utc_weather_hour() -> None:
    local_taxi = pl.DataFrame(
        {
            "tpep_pickup_datetime": [datetime(2024, 1, 31, 23, 30)],
            "tpep_dropoff_datetime": [datetime(2024, 1, 31, 23, 45)],
        }
    ).lazy()
    taxi_utc = add_utc_timestamps(local_taxi, "America/New_York")

    weather = pl.DataFrame(
        {
            "weather_hour_utc": [datetime(2024, 2, 1, 4, 0, tzinfo=UTC)],
            "temperature_2m": [2.5],
            "precipitation": [0.1],
        }
    )

    joined = join_taxi_weather(taxi_utc, weather).collect()

    assert joined.height == 1
    assert joined.get_column("weather_match").to_list() == [True]
    assert joined.get_column("pickup_hour_utc").cast(pl.Int64).item() == (
        1706760000000000
    )
