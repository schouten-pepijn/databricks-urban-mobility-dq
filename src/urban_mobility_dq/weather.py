import json
from pathlib import Path

import polars as pl


def load_weather(path: Path) -> pl.DataFrame:
    """Load and validate hourly Open-Meteo observations."""
    payload = json.loads(path.read_text(encoding="utf-8"))

    if payload.get("error", False):
        raise ValueError(f"Open-Meteo returned an error: {payload.get('reason')}")

    hourly = payload["hourly"]
    units = payload["hourly_units"]

    expected_units = {
        "time": "unixtime",
        "temperature_2m": "°C",
        "precipitation": "mm",
    }
    if any(units.get(name) != expected for name, expected in expected_units.items()):
        raise ValueError(f"Unexpected units: {units}")

    epochs = [int(value) for value in hourly["time"]]
    temperatures = hourly["temperature_2m"]
    precipitation = hourly["precipitation"]

    if not (len(epochs) == len(temperatures) == len(precipitation)):
        raise ValueError("Weather time and measurement arrays have different lengths")

    if len(epochs) != len(set(epochs)):
        raise ValueError("Weather data contains duplicate timestamps")

    if any(epoch % 3600 != 0 for epoch in epochs):
        raise ValueError("Weather timestamps must fall on whole UTC hours")

    observations = pl.DataFrame(
        {
            "epoch_seconds": epochs,
            "temperature_2m": temperatures,
            "precipitation": precipitation,
        },
        schema={
            "epoch_seconds": pl.Int64,
            "temperature_2m": pl.Float64,
            "precipitation": pl.Float64,
        },
    )

    return (
        observations.with_columns(
            pl.from_epoch("epoch_seconds", time_unit="s")
            .dt.replace_time_zone("UTC")
            .alias("weather_hour_utc")
        )
        .select("weather_hour_utc", "temperature_2m", "precipitation")
        .sort("weather_hour_utc")
    )
