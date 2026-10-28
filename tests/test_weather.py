import json
from pathlib import Path

import polars as pl

from urban_mobility_dq.weather import load_weather


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
