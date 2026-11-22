# Bank Failure Early-Warning Pipeline (BFT65B)

Work in progress. Part 1 (acquisition and storage) is complete; later parts will be added.

## Sources and AI usage

### Sources
- [1] BFT65B course starter script `fdic_download_starter.py`, National Economics University, 2026. Used as the basis of `src/p1_download.py` (retry with back-off, `limit/offset` paging, per-quarter manifest, completeness check, UTF-8 stdout setting). Lines taken from it are marked `# SOURCE:` in the code.
- [2] N. T. Long, "Big Data, Spark & Machine Learning at Scale", lecture slides, NEU, 2026. Used for the column-storage/Parquet idea, lazy Polars scans and the Bronze/Silver layering (Part 1).
- [3] FDIC BankFind API field dictionary, `api.fdic.gov/banks/docs/risview_properties.yaml`. Used to check the 33 selected fields.

### AI usage
- **Tool:** Claude (Anthropic).
- **Used for:** drafting `src/p1_download.py`, `src/p1_build_storage.py`, `src/p1_storage_benchmark.py`, `src/p1_field_dictionary.py`, the first draft of the report text for Part 1, and the shortened definitions in report Table A1.
- **How I checked and changed it:** I read and ran each script. I compared downloaded row counts with the API totals, and field titles with the FDIC dictionary. I tested resumability (re-run on a complete download skipped all 170 quarters; after deleting one quarter only that quarter was downloaded again). I re-added the UTF-8 stdout line from the course starter after a `UnicodeEncodeError` when redirecting output on Windows.
- **Error found and fixed:** in the first version of the benchmark script, `failures.parquet` and `institutions.parquet` were copied into the by-quarter layout, so reading `*.parquet` failed with a schema error (extra column `NAME`). I fixed it by excluding those two files from the copy and re-ran the benchmark.