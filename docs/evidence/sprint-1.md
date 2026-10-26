# Sprint 1 — Twee bronnen als bevroren input

- Status: afgerond op 2026-10-26.
- Source month: `2024-01`.
- Manifest: `source_manifest.json`, contract version `1.0.0`.
- TLC source file: `yellow_tripdata_2024-01.parquet`.
  - Bytes: `49,961,641`.
  - SHA-256: `C4D59DA7BBC8ABAEEEB1727947EE93D9891A71ACB42854BD80DB1571B2030510`.
  - Local file hash and byte count matched the manifest.
- Weather source file: `open_meteo_era5_nyc_2024-01.json`.
  - Bytes: `15,939`.
  - SHA-256: `6D21F1A7809FDCF8EA0BA805B826C791D400E3AD1CE0150B3BA2663BD7444910`.
  - Local file hash and byte count matched the manifest.
- Weather request: Open-Meteo Archive API, ERA5, NYC point `40.7580/-73.9855`, `2024-01-01` through `2024-02-01`, hourly temperature and precipitation, UTC, Unix timestamps.
- Source rights: `docs/evidence/source_rights.md` records personal, noncommercial learning use and the attribution requirements. Public redistribution has not been approved.

## Validation results

- Taxi schema contained all six required columns:
  - `tpep_pickup_datetime`: `Datetime(ns, no timezone)`
  - `tpep_dropoff_datetime`: `Datetime(ns, no timezone)`
  - `PULocationID`: `Int32`
  - `DOLocationID`: `Int32`
  - `trip_distance`: `Float64`
  - `total_amount`: `Float64`
- Weather validation passed: 768 unique and contiguous UTC-hour timestamps from `2024-01-01T00:00:00Z` through `2024-02-01T23:00:00Z`; temperature unit `°C`, precipitation unit `mm`.
- Source-derived fixtures:
  - First 100 taxi rows: 100 rows, `10,240` bytes.
  - January 1 pickup slice: 81,013 rows, `1,758,911` bytes; first pickup `00:00:00`, last pickup `23:59:55` in the source's local timestamp representation.
- Unity Catalog Volume upload was read back from a Databricks Job driver. Both uploaded source files matched their local byte counts and SHA-256 hashes.

## Negative hash-mismatch proof

- A temporary copy of the manifest was created under `%TEMP%`; only the taxi SHA-256 was changed to an intentionally incorrect value.
- Command:

  ```powershell
  uv run python scratch/scratch.py --manifest "$env:URBAN_DQ_BAD_MANIFEST"
  ```

- Observed result: `ValueError: Source file does not match the manifest: yellow_tripdata_2024-01.parquet`.
- The failure occurred in `verify_source_file` before taxi schema validation, weather validation, or fixture creation.
- The project `source_manifest.json` and original source files were not modified.

## Scope and evidence gaps

- This sprint validates frozen inputs and file access; it does not produce Silver, quarantine, Delta tables, or a published ETL batch.
- Job run IDs, timestamps, and Git commit were not captured in the reported evidence.
- The script console output and mismatch traceback were reported in the project chat; no separate terminal log was retained.

## Next

Sprint 2: define the versioned taxi row contract and implement file-schema validation before row classification and quarantine.
