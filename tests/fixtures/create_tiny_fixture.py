from pathlib import Path

import polars as pl

fixture_path = Path(__file__).with_name("tiny.parquet")
fixture = pl.DataFrame(
    {"probe_id": [1, 2, 3], "label": ["alpha", "bravo", "charlie"]},
    schema={
        "probe_id": pl.Int64,
        "label": pl.String,
    },
)

fixture.write_parquet(fixture_path)
print(f"Created {fixture_path} ({fixture_path.stat().st_size} bytes)")
