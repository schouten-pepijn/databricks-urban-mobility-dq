from datetime import UTC, datetime

import polars as pl
import pytest

from urban_mobility_dq.taxi import classify_taxi

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
