from datetime import datetime

import polars as pl

from urban_mobility_dq.taxi import add_utc_timestamps

sample = pl.DataFrame(
    {
        "tpep_pickup_datetime": [
            datetime(2024, 1, 1, 0, 0),  # ordinary winter time
            datetime(2024, 3, 10, 2, 30),  # nonexistent in New York
            datetime(2024, 11, 3, 1, 30),  # ambiguous in New York
        ],
        "tpep_dropoff_datetime": [
            datetime(2024, 1, 1, 0, 30),
            datetime(2024, 3, 10, 3, 30),
            datetime(2024, 11, 3, 2, 30),
        ],
    }
).lazy()

converted = add_utc_timestamps(sample, "America/New_York")
print(converted.select("pickup_utc", "dropoff_utc").collect())
