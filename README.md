# Bank Failure Early-Warning Pipeline (BFT65B)

Data pipeline for a bank-failure early-warning system, built on US FDIC Call Report data (1984Q1–2026Q2).
 
## Quick start (Windows, PowerShell)

Tested with Python 3.13.7 on Windows 10 (Intel Core i5-6300U, 8 GB RAM, SSD). Internet access is needed for the download and for `p2_exits.py`.

```powershell
git clone https://github.com/linhlinh120905/bank-failure-pipeline
cd bank-failure-pipeline
python -m venv .venv
.venv\Scripts\activate          # if blocked: Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
pip install -r requirements.txt
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8   # once per terminal, for accented text
```

Run the scripts in this order (each reads the output of the previous ones):

| Step | Command | Time (approx.) | Output |
|---|---|---|---|
| 1 | `python src/p1_download.py` | 20–30 min first run; seconds when complete (resumable) | `data/raw/` (170 quarterly Parquet files, `failures.parquet`, `institutions.parquet`, `manifest_financials.json`, `download_report.json`) |
| 2 | `python src/p1_build_storage.py` | seconds | `data/parquet/financials/year=YYYY/` (43 files, zstd) |
| 3 | `python src/p1_storage_benchmark.py` | 3–5 min, memory heavy: close other applications | `reports/storage_benchmark.json` |
| 4 | `python src/p1_field_dictionary.py` | seconds | `reports/field_dictionary.json` |
| 5 | `python src/p2_exits.py` | 1–2 min (calls the FDIC `history` API once per code, cached in `reports/changec1_codes.json`) | `data/parquet/exits.parquet`, `reports/exits_summary.json` |
| 6 | `python src/p2_quality_report.py` | 1–4 min | `reports/quality_report.json` |
| 7 | `python src/p2_build_panel.py` | 1–2 min, needs about 1 GB free RAM | `data/parquet/panel/`, `reports/panel_summary.json` |

Quick test of the download only: `python src/p1_download.py --start 2023-03-31 --end 2023-12-31`.
A single `run_all.py` will be added when all parts are finished.

## Expected results (to check a rerun)

- Financials: 170 quarters, 1,682,738 rows, 24,064 distinct `CERT`, 0 duplicate (`CERT`, `REPDTE`), 0 cells coerced to null; Failures 4,117 rows; Institutions 27,838 rows.
- Storage: 43 yearly files, about 156.5 MB (CSV about 443 MB); row count read back equals the source.
- Exits (24,064 banks): merger 15,588; active 4,313; failure 2,841; closure code but not in Failures 623; voluntary closure 504; other 195.
- Panel: 1,682,738 − 1,092 rows after the failure date = 1,681,646 rows; 0 rows after `FAILDATE`; 0 duplicate bank-quarters.

## Repository layout

```
config.py                 shared constants (paths, field lists, seed = 42)
src/p1_*.py               Part 1: download, storage layout, benchmark, field dictionary
src/p2_*.py               Part 2: exits, quality report, panel
reports/*.json            small result files used in the report (no raw data)
data/                     NOT committed (see .gitignore): raw, parquet, logs
requirements.txt          pinned versions (pip freeze)
```

Data files are never committed. Parts 1–2 contain no random component; the seed in `config.py` (42) is used from the modelling parts on.

## Design decisions fixed so far

- **Label:** y = 1 if the bank fails (FDIC `RESTYPE` = FAILURE, charter closed) within (q, q + 4 quarters]. Mergers are not failures. `RESTYPE` = ASSISTANCE (mostly banks whose charter survives) is used only in a sensitivity analysis.
- **Split (fixed before any modelling):** development 1984Q1–2006Q4; gap 2007Q1–2007Q4; test 2008Q1–2025Q2, used once. Labels use quarters up to 2025Q2 only.
- **Point-in-time:** the panel row for quarter q contains only what was reported for q, plus quarterly flows computed from q and q−1. Failure dates, exit types and other later information live in a separate exit table and enter only through the label.

## Known limitations

- The data contain no filing dates, so the report of quarter q is assumed available at the end of q (a one-quarter lag is tested later as a sensitivity check). The API returns current values, so restatements cannot be undone.
- The `CHANGEC1` code groups are my reading of the descriptions returned by the FDIC `history` API, not an official definition; 623 closure-coded banks are absent from the Failures table and mostly stopped reporting in 1985–1994.
- Development contains almost only the 1984–1992 crisis (about 95% of its failures), test contains the 2008 crisis.
- Benchmarks were run on one 8 GB laptop with high memory use, so absolute timings are indicative.

## Sources and AI usage

### Sources
- [1] BFT65B course starter script `fdic_download_starter.py`, National Economics University, 2026. Used as the basis of `src/p1_download.py` (retry with back-off, `limit/offset` paging, per-quarter manifest, completeness check, UTF-8 stdout setting). Lines taken from it are marked `# SOURCE:` in the code.
- [2] N. T. Long, "Big Data, Spark & Machine Learning at Scale", lecture slides, NEU, 2026. Used for the column-storage/Parquet idea, lazy Polars scans, the Bronze/Silver layering (Part 1) and window-function logic (Part 2).
- [3] FDIC BankFind API field dictionary, `api.fdic.gov/banks/docs/risview_properties.yaml`. Used to check the 33 selected fields.
- [4] Claude (Anthropic): see "AI usage".
- [5] FDIC dictionaries for Failures and Institutions (`failure_properties`, `institution_properties`).
- [6] FDIC API `history` endpoint, field `CHANGECODE_DESC` (descriptions of `CHANGEC1` codes).
- [7] N. T. Long, "Data Cleaning", lecture slides, NEU, 2026 (IQR rule, duplicate and missing-value checks).
- [8] N. T. Long, "Time Series for Finance", lecture slides, NEU, 2026 (point-in-time features, leakage).

### AI usage
- **Tool:** Claude (Anthropic).
- **Part 1, used for:** drafting `src/p1_download.py`, `src/p1_build_storage.py`, `src/p1_storage_benchmark.py`, `src/p1_field_dictionary.py`, the first draft of the report text for Part 1, and the shortened definitions in report Table A1.
- **Part 1, how I checked and changed it:** I read and ran each script. I compared downloaded row counts with the API totals, and field titles with the FDIC dictionary. I tested resumability (re-run on a complete download skipped all 170 quarters; after deleting one quarter only that quarter was downloaded again). I re-added the UTF-8 stdout line from the course starter after a `UnicodeEncodeError` when redirecting output on Windows. I added `RESTYPE1` and `RESDATE` to the Failures download after reading the FDIC dictionary.
- **Part 1, error found and fixed:** in the first version of the benchmark script, `failures.parquet` and `institutions.parquet` were copied into the by-quarter layout, so reading `*.parquet` failed with a schema error (extra column `NAME`). I fixed it by excluding those two files from the copy and re-ran the benchmark.
- **Part 2, used for:** drafting `src/p2_exits.py`, `src/p2_quality_report.py`, `src/p2_build_panel.py` and the first draft of the report text for Part 2.
- **Part 2, how I checked and changed it:** I read and ran each script and checked the output. After the first run I regrouped the `CHANGEC1` codes using the descriptions returned by the API, added a voluntary-closure group, and verified that the exit groups sum to 24,064 banks. I confirmed that the year-to-date conversion gives almost no negative quarterly interest items, and that the panel row count equals the source minus the removed rows.
- **Part 2, issue found by the audit:** integer columns with nulls were saved as text such as "223.0" by my download step; they are cast back to integers before joins.