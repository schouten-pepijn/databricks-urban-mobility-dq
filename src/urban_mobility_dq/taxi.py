from datetime import datetime
from pathlib import Path

import polars as pl


def count_parquet_rows(path: Path) -> int:
    """Count the number of rows in a parquet file."""
    count = pl.scan_parquet(path).select(pl.len().alias("row_count")).collect().item()
    return int(count)


def reconcile_taxi_row_counts(
    raw_path: Path,
    accepted_path: Path,
    quarantine_path: Path,
) -> dict[str, int]:
    """Count the raw and staged files and verify that no rows were lost."""
    counts = {
        "raw": count_parquet_rows(raw_path),
        "accepted": count_parquet_rows(accepted_path),
        "quarantine": count_parquet_rows(quarantine_path),
    }

    if counts["raw"] != counts["accepted"] + counts["quarantine"]:
        raise ValueError(
            "Row-count reconciliation failed: "
            f"raw={counts['raw']}, accepted={counts['accepted']}, "
            f"quarantine={counts['quarantine']}"
        )

    return counts


def sink_classified_taxi(
    accepted: pl.LazyFrame,
    quarantine: pl.LazyFrame,
    stage_root: Path,
    batch_id: str,
) -> tuple[Path, Path]:
    """Write both outputs, then mark the batch complete; assumes one batch writer."""
    accepted_path = (
        stage_root / "accepted" / f"batch_id={batch_id}" / "part-000.parquet"
    )
    quarantine_path = (
        stage_root / "quarantine" / f"batch_id={batch_id}" / "part-000.parquet"
    )
    completion_marker = (
        stage_root / "_complete" / f"batch_id={batch_id}" / "_SUCCESS"
    )

    # The marker means both outputs were fully written, so committed data is immutable.
    if completion_marker.exists():
        raise FileExistsError(f"Staged output already exists for batch {batch_id}")

    # Files without a marker are leftovers from an incomplete attempt and can be rebuilt.
    accepted_path.unlink(missing_ok=True)
    quarantine_path.unlink(missing_ok=True)

    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    completion_marker.parent.mkdir(parents=True, exist_ok=True)

    accepted.sink_parquet(accepted_path)
    quarantine.sink_parquet(quarantine_path)
    completion_marker.touch(exist_ok=False)

    return accepted_path, quarantine_path


def add_utc_timestamps(
    rows: pl.LazyFrame,
    source_timezone: str,
) -> pl.LazyFrame:
    """Interpret naive taxi times in their source timezone and derive UTC times."""
    # These source values are local clock readings, so attach their timezone first.
    # Ambiguous or nonexistent DST times become null for the classifier to quarantine.
    return rows.with_columns(
        pl.col("tpep_pickup_datetime")
        .dt.replace_time_zone(
            source_timezone,
            ambiguous="null",
            non_existent="null",
        )
        .dt.convert_time_zone("UTC")
        .alias("pickup_utc"),
        pl.col("tpep_dropoff_datetime")
        .dt.replace_time_zone(
            source_timezone,
            ambiguous="null",
            non_existent="null",
        )
        .dt.convert_time_zone("UTC")
        .alias("dropoff_utc"),
    )


def scan_taxi_with_identity(
    parquet_path: Path,
    source_sha256: str,
    batch_id: str,
) -> pl.LazyFrame:
    """Scan taxi rows and add stable source and batch identifiers."""
    return (
        pl.scan_parquet(parquet_path)
        # Add the row number before any later filters so it points to the source row.
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
    # Keep these checks in contract order: the first matching rule is the primary reason.
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
        # A negative amount is useful to measure, but the contract does not reject it.
        (pl.col("total_amount") < 0)
        .fill_null(False)
        .alias("negative_total_amount_warning"),
    )

    accepted = classified.filter(pl.col("reason_code").is_null())
    quarantine = classified.filter(pl.col("reason_code").is_not_null())

    # The two filters partition the input: each row belongs to exactly one output.
    return accepted, quarantine
