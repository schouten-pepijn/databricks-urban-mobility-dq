import json
import os
from datetime import datetime
from pathlib import Path

import polars as pl

from urban_mobility_dq.contracts import validate_file_schema
from urban_mobility_dq.taxi import (
    add_utc_timestamps,
    classify_taxi,
    scan_taxi_with_identity,
)
from urban_mobility_dq.weather import (
    join_taxi_weather,
    load_weather,
    summarize_weather_metrics,
)

manifest = json.loads(Path("source_manifest.json").read_text(encoding="utf-8"))
source_dir = (
    Path(os.environ["LOCALAPPDATA"])
    / "urban-mobility-dq"
    / "source"
    / manifest["source_month"]
)
taxi_path = source_dir / manifest["taxi"]["file_name"]
weather_path = source_dir / manifest["weather"]["file_name"]

extra_columns = validate_file_schema(taxi_path, Path("contracts/taxi_v1.yaml"))
weather = load_weather(weather_path)

taxi = scan_taxi_with_identity(
    taxi_path,
    source_sha256=manifest["taxi"]["sha256"],
    batch_id=f"coverage-{manifest['source_month']}",
)
taxi_utc = add_utc_timestamps(taxi, "America/New_York")
accepted, quarantine = classify_taxi(
    taxi_utc,
    datetime(2024, 1, 1),
    datetime(2024, 2, 1),
)

accepted_rows = accepted.select(pl.len().alias("rows")).collect().item()
quarantine_rows = quarantine.select(pl.len().alias("rows")).collect().item()
joined = join_taxi_weather(accepted, weather).collect()

if joined.height != accepted_rows:
    raise ValueError(
        f"Weather join changed the accepted row count: "
        f"accepted={accepted_rows}, joined={joined.height}"
    )

print(
    {
        "extra_taxi_columns": extra_columns,
        "accepted_rows": accepted_rows,
        "quarantine_rows": quarantine_rows,
        **summarize_weather_metrics(joined),
    }
)
