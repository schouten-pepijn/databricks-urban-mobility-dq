from pathlib import Path

import polars as pl


def scan_taxi_with_identity(
    parquet_path: Path,
    source_sha256: str,
    batch_id: str,
) -> pl.LazyFrame:
    """Scan taxi rows and add stable source and batch identifiers."""
    return (
        pl.scan_parquet(parquet_path)
        .with_row_index("source_row_number")
        .with_columns(
            pl.col("source_row_number").cast(pl.Int64),
            pl.lit(source_sha256).alias("source_sha256"),
            pl.lit(batch_id).alias("batch_id"),
        )
        .with_columns(
            pl.concat_str(
                pl.col("source_sha256"),
                pl.col("source_row_number").cast(pl.String),
                separator=":",
            ).alias("trip_key")
        )
    )


def classify_taxi(
    rows: pl.LazyFrame,
    month_start_local: datetime,
    month_end_local: datetime,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Split rows into accepted and quarantined sets using taxi v1 rules."""
    reason_code = (
        pl.when(pl.col("tpep_pickup_datetime").is_null())
        .then(pl.lit("pickup_missing"))
        .when(pl.col("pickup_utc").is_null())
        .then(pl.lit("pickup_dst_invalid"))
        .when(
            ~pl.col("tpep_pickup_datetime").is_between(
                month_start_local,
                month_end_local,
                closed="left",
            )
        )
        .then(pl.lit("pickup_outside_month"))
        .when(pl.col("tpep_dropoff_datetime").is_null())
        .then(pl.lit("dropoff_missing"))
        .when(pl.col("dropoff_utc").is_null())
        .then(pl.lit("dropoff_dst_invalid"))
        .when(pl.col("dropoff_utc") <= pl.col("pickup_utc"))
        .then(pl.lit("duration_invalid"))
        .when((pl.col("dropoff_utc") - pl.col("pickup_utc")) > pl.duration(hours=24))
        .then(pl.lit("duration_invalid"))
        .when(pl.col("trip_distance").is_null())
        .then(pl.lit("distance_missing"))
        .when(pl.col("trip_distance") < 0)
        .then(pl.lit("distance_negative"))
        .when(
            pl.col("PULocationID").is_null()
            | (pl.col("PULocationID") <= 0)
            | pl.col("DOLocationID").is_null()
            | (pl.col("DOLocationID") <= 0)
        )
        .then(pl.lit("location_missing_or_invalid"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )

    classified = rows.with_columns(
        reason_code.alias("reason_code"),
        (pl.col("total_amount") < 0)
        .fill_null(False)
        .alias("negative_total_amount_warning"),
    )

    accepted = classified.filter(pl.col("reason_code").is_null())
    quarantine = classified.filter(pl.col("reason_code").is_not_null())

    return accepted, quarantine
