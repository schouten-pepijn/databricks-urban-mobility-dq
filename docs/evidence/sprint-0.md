# Sprint 0 — Omgeving en rechten vaststellen

- Status: completed, 2026-10-25.
- Repository commit: not captured.
- Databricks profile: `databricks-dev`; cloud: AWS.
- Workspace: Free Edition; serverless compute is quota-limited and has no monetary charge.
- Catalog: `mobility_lab`, managed catalog using Default Storage; isolation mode is `ISOLATED`.
- Workspace binding: current workspace has `BINDING_TYPE_READ_WRITE` access.
- Schema and volume: `mobility_lab.urban_dq_dev.landing` created as a managed volume.
- Job compute: serverless, Standard V6 environment; Python 3.12, Scala 2.13, Java 17.
- Local environment: Python 3.12.14, Polars 2.0.0, PyArrow 26.0.0.
- Effective access observed: schema and volume creation succeeded; CLI listed the volume; synthetic Parquet upload succeeded; Job driver read and wrote the file, and the source/readback hashes matched.
- Fixture: synthetic Parquet, 966 bytes. This was a file-access smoke check, not TLC contract validation.
- Smoke check result: passed, based on the reported matching source and readback hashes. Job run ID, timestamps, and hash value were not captured.
- Cost: no per-run monetary estimate for Free Edition; fair-use quotas apply.
- Sprint 1 has not started; source files, licenses, and the source manifest have not yet been validated.
