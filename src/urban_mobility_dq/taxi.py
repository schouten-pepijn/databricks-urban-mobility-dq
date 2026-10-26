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
