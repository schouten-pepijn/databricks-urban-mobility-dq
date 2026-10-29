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


def join_taxi_weather(
    taxi: pl.LazyFrame,
    weather: pl.DataFrame,
) -> pl.LazyFrame:
    """Join each taxi pickup to at most one weather observation."""
    if weather.get_column("weather_hour_utc").n_unique() != weather.height:
        raise ValueError("Weather data contains duplicate timestamps")

    taxi_by_hour = taxi.with_columns(
        pl.col("pickup_utc")
        .dt.truncate("1h")
        # Taxi timestamps are nanoseconds; weather timestamps use microseconds.
        .cast(pl.Datetime(time_unit="us", time_zone="UTC"))
        .alias("pickup_hour_utc")
    )
    weather_for_join = weather.lazy().with_columns(
        pl.lit(True).alias("_weather_present")
    )

    return (
        taxi_by_hour.join(
            weather_for_join,
            left_on="pickup_hour_utc",
            right_on="weather_hour_utc",
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.col("_weather_present").fill_null(False).alias("weather_match")
        )
        .drop("_weather_present")
    )


def summarize_weather_metrics(
    joined: pl.DataFrame,
) -> dict[str, int | float | None]:
    """Summarize wather coverage for the joined taxi rows."""
    required_hours = joined.get_column("pickup_hour_utc").n_unique()
    matched = joined.filter(pl.col("weather_match"))
    present_hours = matched.get_column("pickup_hour_utc").n_unique()

    trip_match_rate = matched.height / joined.height if joined.height > 0 else None

    temperature_null_hours = (
        matched.filter(pl.col("temperature_2m").is_null())
        .get_column("pickup_hour_utc")
        .n_unique()
    )
    precipitation_null_hours = (
        matched.filter(pl.col("precipitation").is_null())
        .get_column("pickup_hour_utc")
        .n_unique()
    )

    return {
        "weather_required_hours": required_hours,
        "weather_present_hours": present_hours,
        "weather_missing_hours": required_hours - present_hours,
        "trip_weather_match_rate": trip_match_rate,
        "temperature_null_hours": temperature_null_hours,
        "precipitation_null_hours": precipitation_null_hours,
    }
