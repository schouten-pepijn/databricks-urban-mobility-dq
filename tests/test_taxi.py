from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from urban_mobility_dq.taxi import (
    add_utc_timestamps,
    classify_taxi,
    reconcile_taxi_row_counts,
    scan_taxi_with_identity,
    sink_classified_taxi,
)

MONTH_START = datetime(2024, 1, 1)
MONTH_END = datetime(2024, 2, 1)

# Explicit types let the fixtures include nulls without Polars guessing a type.
ROW_SCHEMA = {
    "test_case": pl.String,
    "tpep_pickup_datetime": pl.Datetime("ns", time_zone=None),
    "tpep_dropoff_datetime": pl.Datetime("ns", time_zone=None),
    "pickup_utc": pl.Datetime("ns", time_zone="UTC"),
    "dropoff_utc": pl.Datetime("ns", time_zone="UTC"),
    "PULocationID": pl.Int32,
    "DOLocationID": pl.Int32,
    "trip_distance": pl.Float64,
    "total_amount": pl.Float64,
}


def valid_row(test_case: str = "valid", **overrides: object) -> dict[str, object]:
    """Return one valid January taxi row, allowing individual fields to be changed."""
    row: dict[str, object] = {
        "test_case": test_case,
        "tpep_pickup_datetime": datetime(2024, 1, 15, 12, 0),
        "tpep_dropoff_datetime": datetime(2024, 1, 15, 12, 30),
        "pickup_utc": datetime(2024, 1, 15, 17, 0, tzinfo=UTC),
        "dropoff_utc": datetime(2024, 1, 15, 17, 30, tzinfo=UTC),
        "PULocationID": 161,
        "DOLocationID": 236,
        "trip_distance": 2.5,
        "total_amount": 18.5,
    }
    row.update(overrides)
    return row


def classify_rows(
    rows: list[dict[str, object]],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run the lazy classifier on typed synthetic rows and collect both outputs."""
    source = pl.DataFrame(rows, schema=ROW_SCHEMA).lazy()
    accepted, quarantine = classify_taxi(source, MONTH_START, MONTH_END)
    return accepted.collect(), quarantine.collect()


def write_row_count_parquet(path: Path, row_count: int) -> Path:
    """Write a small Parquet file with the requested number of rows."""
    pl.DataFrame({"row_number": list(range(row_count))}).write_parquet(path)
    return path


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        pytest.param(
            {"tpep_pickup_datetime": None, "pickup_utc": None},
            "pickup_missing",
            id="missing-pickup",
        ),
        pytest.param(
            {
                "tpep_pickup_datetime": datetime(2024, 11, 3, 1, 30),
                "pickup_utc": None,
            },
            "pickup_dst_invalid",
            # This is also outside January; the earlier DST rule must win.
            id="dst-invalid-pickup-before-out-of-month-rule",
        ),
        pytest.param(
            {
                "tpep_pickup_datetime": datetime(2024, 2, 1),
                "pickup_utc": datetime(2024, 2, 1, 5, tzinfo=UTC),
            },
            "pickup_outside_month",
            # The interval ends at February 1, so this timestamp is excluded.
            id="month-end-is-exclusive",
        ),
        pytest.param(
            {"tpep_dropoff_datetime": None, "dropoff_utc": None},
            "dropoff_missing",
            id="missing-dropoff",
        ),
        pytest.param(
            {
                "tpep_dropoff_datetime": datetime(2024, 11, 3, 1, 30),
                "dropoff_utc": None,
            },
            "dropoff_dst_invalid",
            id="dst-invalid-dropoff",
        ),
        pytest.param(
            {
                "tpep_dropoff_datetime": datetime(2024, 1, 15, 12, 0),
                "dropoff_utc": datetime(2024, 1, 15, 17, 0, tzinfo=UTC),
            },
            "duration_invalid",
            id="dropoff-not-after-pickup",
        ),
        pytest.param(
            {
                "tpep_dropoff_datetime": datetime(2024, 1, 16, 13, 0),
                "dropoff_utc": datetime(2024, 1, 16, 18, 0, tzinfo=UTC),
            },
            "duration_invalid",
            id="duration-over-24-hours",
        ),
        pytest.param(
            {"trip_distance": None},
            "distance_missing",
            id="missing-distance",
        ),
        pytest.param(
            {"trip_distance": -0.1},
            "distance_negative",
            id="negative-distance",
        ),
        pytest.param(
            {"PULocationID": None},
            "location_missing_or_invalid",
            id="missing-pickup-location",
        ),
        pytest.param(
            {"DOLocationID": 0},
            "location_missing_or_invalid",
            id="non-positive-dropoff-location",
        ),
        pytest.param(
            {
                "tpep_pickup_datetime": None,
                "pickup_utc": None,
                "trip_distance": -0.1,
            },
            "pickup_missing",
            id="first-rule-wins-when-several-rules-fail",
        ),
    ],
)
def test_invalid_rows_are_quarantined_with_primary_reason(
    overrides: dict[str, object],
    expected_reason: str,
) -> None:
    row = valid_row()
    row.update(overrides)
    accepted, quarantine = classify_rows([row])

    assert accepted.height == 0
    assert quarantine.height == 1
    assert quarantine.get_column("reason_code").item() == expected_reason
    assert accepted.height + quarantine.height == 1


def test_valid_rows_and_negative_amount_warnings_are_accepted() -> None:
    # A negative or missing total amount is retained; only the negative gets a warning.
    rows = [
        valid_row("positive-amount"),
        valid_row("negative-amount", total_amount=-2.0),
        valid_row("null-amount", total_amount=None),
        valid_row(
            "month-start-boundary",
            tpep_pickup_datetime=MONTH_START,
            tpep_dropoff_datetime=datetime(2024, 1, 1, 0, 15),
            pickup_utc=datetime(2024, 1, 1, 5, tzinfo=UTC),
            dropoff_utc=datetime(2024, 1, 1, 5, 15, tzinfo=UTC),
        ),
    ]

    accepted, quarantine = classify_rows(rows)

    assert accepted.height == len(rows)
    assert quarantine.height == 0
    warnings = dict(
        accepted.select("test_case", "negative_total_amount_warning")
        .sort("test_case")
        .iter_rows()
    )
    assert warnings == {
        "month-start-boundary": False,
        "negative-amount": True,
        "null-amount": False,
        "positive-amount": False,
    }
    assert accepted.height + quarantine.height == len(rows)


def test_sink_classified_taxi_writes_separate_immutable_outputs(
    tmp_path: Path,
) -> None:
    rows = [
        valid_row("accepted"),
        valid_row("quarantined", trip_distance=-1.0),
    ]
    accepted, quarantine = classify_rows(rows)

    accepted_path, quarantine_path = sink_classified_taxi(
        accepted.lazy(),
        quarantine.lazy(),
        tmp_path / "stage",
        "test-batch",
    )

    # Read both Parquet files back to prove each output was written correctly.
    assert pl.read_parquet(accepted_path).get_column("test_case").to_list() == [
        "accepted"
    ]
    assert pl.read_parquet(quarantine_path).get_column("test_case").to_list() == [
        "quarantined"
    ]
    assert (
        tmp_path / "stage" / "_complete" / "batch_id=test-batch" / "_SUCCESS"
    ).is_file()

    # A batch's staged files are immutable, so a repeat cannot overwrite them.
    with pytest.raises(FileExistsError, match="test-batch"):
        sink_classified_taxi(
            accepted.lazy(),
            quarantine.lazy(),
            tmp_path / "stage",
            "test-batch",
        )


def test_sink_classified_taxi_retries_incomplete_batch(tmp_path: Path) -> None:
    stage_root = tmp_path / "stage"
    batch_id = "retry-batch"
    accepted_path = (
        stage_root / "accepted" / f"batch_id={batch_id}" / "part-000.parquet"
    )
    completion_marker = stage_root / "_complete" / f"batch_id={batch_id}" / "_SUCCESS"
    accepted = pl.DataFrame({"value": [1]}).lazy()
    invalid_quarantine = pl.DataFrame({"value": [2]}).lazy().select("missing")

    # The second sink fails, so the first output is left uncommitted without a marker.
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        sink_classified_taxi(accepted, invalid_quarantine, stage_root, batch_id)

    assert accepted_path.is_file()
    assert not completion_marker.exists()

    # A retry replaces leftover files and marks the batch complete after both writes.
    accepted_path, quarantine_path = sink_classified_taxi(
        accepted,
        pl.DataFrame({"value": [2]}).lazy(),
        stage_root,
        batch_id,
    )

    assert pl.read_parquet(accepted_path).get_column("value").to_list() == [1]
    assert pl.read_parquet(quarantine_path).get_column("value").to_list() == [2]
    assert completion_marker.is_file()


def test_sink_classified_taxi_propagates_write_failure(tmp_path: Path) -> None:
    rows = pl.DataFrame({"value": [1]}).lazy()

    # Simulate a filesystem failure and ensure callers see the original error.
    with (
        patch.object(
            pl.LazyFrame,
            "sink_parquet",
            side_effect=OSError("simulated disk error"),
        ),
        pytest.raises(OSError, match="simulated disk error"),
    ):
        sink_classified_taxi(rows, rows, tmp_path / "stage", "failed-batch")


def test_reconcile_taxi_row_counts_accepts_balanced_files(tmp_path: Path) -> None:
    raw_path = write_row_count_parquet(tmp_path / "raw.parquet", 3)
    accepted_path = write_row_count_parquet(tmp_path / "accepted.parquet", 2)
    quarantine_path = write_row_count_parquet(tmp_path / "quarantine.parquet", 1)

    counts = reconcile_taxi_row_counts(raw_path, accepted_path, quarantine_path)

    assert counts == {"raw": 3, "accepted": 2, "quarantine": 1}


def test_reconcile_taxi_row_counts_rejects_unbalanced_files(tmp_path: Path) -> None:
    raw_path = write_row_count_parquet(tmp_path / "raw.parquet", 3)
    accepted_path = write_row_count_parquet(tmp_path / "accepted.parquet", 1)
    quarantine_path = write_row_count_parquet(tmp_path / "quarantine.parquet", 1)

    with pytest.raises(ValueError, match="Row-count reconciliation failed"):
        reconcile_taxi_row_counts(raw_path, accepted_path, quarantine_path)


def test_identical_source_rows_keep_distinct_physical_keys(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "duplicate_rows.parquet"
    duplicate = valid_row("same-source-row")
    pl.DataFrame([duplicate, duplicate], schema=ROW_SCHEMA).write_parquet(source_path)

    identified = scan_taxi_with_identity(
        source_path,
        source_sha256="a" * 64,
        batch_id="duplicate-test",
    )
    with_utc = add_utc_timestamps(identified, "America/New_York")
    accepted, quarantine = classify_taxi(with_utc, MONTH_START, MONTH_END)

    accepted_rows = accepted.collect()

    assert accepted_rows.height == 2
    assert quarantine.collect().height == 0
    assert accepted_rows.get_column("source_row_number").to_list() == [0, 1]
    assert accepted_rows.get_column("trip_key").n_unique() == 2


def test_nonexistent_dst_pickup_is_quarantined() -> None:
    row = valid_row(
        "dst-gap",
        tpep_pickup_datetime=datetime(2024, 3, 10, 2, 30),
    )
    source = pl.DataFrame([row], schema=ROW_SCHEMA).lazy()

    with_utc = add_utc_timestamps(source, "America/New_York").collect()
    assert with_utc.get_column("pickup_utc").item() is None

    accepted, quarantine = classify_taxi(
        with_utc.lazy(),
        MONTH_START,
        MONTH_END,
    )

    assert accepted.collect().is_empty()
    assert quarantine.collect().get_column("reason_code").to_list() == [
        "pickup_dst_invalid"
    ]
