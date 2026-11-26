# Bank Failure Early-Warning Pipeline (BFT65B)

Data pipeline for a bank-failure early-warning system, built on US FDIC Call Report data (1984Q1–2026Q2).
 
 **Status:** all parts are complete (Parts 1–5 in this repository; Part 6, the conclusions, is in the report).

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
| 8 | `python src/p3_polars_core.py` | under 1 min | `reports/p3_polars_results.json`, `reports/p3_polars_plans.txt`, `data/p3_spark_input.parquet` (for Colab) |
| 9 | `python src/p3_benchmark.py` | 4–8 min; close other applications | `reports/p3_benchmark.json` |
| 10 | `python src/p3_scaling.py` | 10–20 min; writes about 750 MB of temporary data (deleted at the end) | `reports/p3_scaling.json` |
| 11 | `python src/p4_features.py` | about 1 min | `data/parquet/features/` (one file per year), `reports/feature_catalog.json`, `reports/features_summary.json` |
| 12 | `python src/p4_pipeline.py` | under 1 min | `reports/preprocessor_dev_fit.json`, `data/preprocessor_dev.joblib` (not committed) |
| 13 | `python src/p4_leakage_audit.py` | 1–2 min | `reports/leakage_audit.json` (7 of 7 tests must pass) |
| 14 | python src/p5_validation.py | 20–60 min | reports/p5_dev_selection.json, data/p5_spark_dev.parquet |
| 15 | python src/p5_test_once.py --run | 2–5 min, ONCE | reports/p5_test_result.json |

Spark (Part 3): open `notebooks/p3_spark_colab.ipynb` in Google Colab, upload `data/p3_spark_input.parquet`, `data/p3_spark_exits.parquet` and `reports/p3_polars_results.json`, run all cells; it downloads `p3_spark_results.json` and `p3_spark_plans.txt` (stored in `reports/`). Spark was run on Colab (Spark 4.0.5, Java 21, local mode) because Java and Hadoop are fragile on Windows and the laptop has only 8 GB.
Quick test of the download only: `python src/p1_download.py --start 2023-03-31 --end 2023-12-31`.
Everything above in one command: `python run_all.py` (options: `--list`, `--skip-download`, `--with-benchmarks`, `--with-test`). The Colab notebooks (Spark) are run manually. To reproduce the single test run, delete `reports/p5_test_result.json` and run `python src/p5_test_once.py --run`; the result stores the hash of the frozen selection file.
Everything above in one command: python run_all.py (options: --list, --skip-download, --with-benchmarks, --with-test). The Colab notebooks (Spark) are run manually. To reproduce the single test run, delete reports/p5_test_result.json and run python src/p5_test_once.py --run; the result stores the hash of the frozen selection file.
Spark MLlib (Part 5): open `notebooks/p5_spark_mllib_colab.ipynb` in Google Colab, upload `data/p5_spark_dev.parquet` and `reports/p5_dev_selection.json`, run all cells; it downloads `p5_spark_results.json` (stored in `reports/`).
## Expected results (to check a rerun)

- Financials: 170 quarters, 1,682,738 rows, 24,064 distinct `CERT`, 0 duplicate (`CERT`, `REPDTE`), 0 cells coerced to null; Failures 4,117 rows; Institutions 27,838 rows.
- Storage: 43 yearly files, about 156.5 MB (CSV about 443 MB); row count read back equals the source.
- Exits (24,064 banks): merger 15,588; active 4,313; failure 2,841; closure code but not in Failures 623; voluntary closure 504; other 195.
- Panel: 1,682,738 − 1,092 rows after the failure date = 1,681,646 rows; 0 rows after `FAILDATE`; 0 duplicate bank-quarters.
Quick test of the download only: `python src/p1_download.py --start 2023-03-31 --end 2023-12-31`.
Part 4: feature table 1,664,117 rows (development 1,193,917, gap 34,754, test 435,446); failures by failure date: development 2,289, 2007 3, test 545; leakage audit 7/7 passed.
- Part 5: development pooled PR-AUC 0.377 (gradient boosting) and 0.307 (logistic regression); test PR-AUC 0.595 and 0.607 (positive rate 0.0049); 521 (LR) and 529 (GB) of 545 failed banks flagged at least once.
## Repository layout

```
config.py                 shared constants (paths, field lists, seed = 42)
run_all.py                runs the whole pipeline in order (python run_all.py --list)
src/p1_*.py               Part 1: download, storage layout, benchmark, field dictionary
src/p2_*.py               Part 2: exits, quality report, panel
src/p3_*.py               Part 3: Polars operations, benchmark, scaling experiments
src/p4_*.py               Part 4: features, preprocessing pipeline, leakage audit
src/p5_*.py               Part 5: validation, single test evaluation
notebooks/                Colab notebooks (Spark Part 3, MLlib Part 5), saved with output
reports/*.json            small result files used in the report (no raw data)
data/                     NOT committed (see .gitignore): raw, parquet, logs
requirements.txt          pinned versions (pip freeze)
```

Data files are never committed. Parts 1–2 contain no random component; the seed in `config.py` (42) is used from the modelling parts on.

## Design decisions fixed so far

- **Label:** y = 1 if the bank fails (FDIC `RESTYPE` = FAILURE, charter closed) within (q, q + 4 quarters]. Mergers are not failures. `RESTYPE` = ASSISTANCE (mostly banks whose charter survives) is used only in a sensitivity analysis.
- **Split (fixed before any modelling):** development 1984Q1–2006Q4; gap 2007Q1–2007Q4; test 2008Q1–2025Q2, used once. Labels use quarters up to 2025Q2 only.
- **Point-in-time:** the panel row for quarter q contains only what was reported for q, plus quarterly flows computed from q and q−1. Failure dates, exit types and other later information live in a separate exit table and enter only through the label.
- **Features:** 29 CAMELS ratios; zeros in RBC1AAJ/RBCRWAJ are treated as missing; missing flags, rbc_total and P3ASSET are excluded from the default model (their missingness identifies the period).
- **Modelling:** baseline logistic regression against gradient boosting, expanding-window validation with a four-quarter gap; the alarm threshold minimises the normalised cost on development out-of-fold predictions, with the cost of a missed failure taken from the `COST` field and the cost of a false alarm set to 0.02 times that (an assumption, tested from 0.005 to 0.10); the test period is evaluated once (`p5_test_once.py` refuses to run twice) after the selection file is committed.
## Known limitations

- The data contain no filing dates, so the report of quarter q is assumed available at the end of q (a one-quarter lag is tested later as a sensitivity check). The API returns current values, so restatements cannot be undone.
- The `CHANGEC1` code groups are my reading of the descriptions returned by the FDIC `history` API, not an official definition; 623 closure-coded banks are absent from the Failures table and mostly stopped reporting in 1985–1994.
- Development contains almost only the 1984–1992 crisis (about 95% of its failures), test contains the 2008 crisis.
- Benchmarks were run on one 8 GB laptop with high memory use, so absolute timings are indicative.
- On the test period the boosted model is not better than the logistic baseline, performance in 2022–2025 is weak (11 of 36 positive rows flagged), and the false-alarm cost ratio is an assumption.
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
- [9] N. T. Long, "Machine Learning Pipeline", lecture slides, NEU, 2026 (fit on training data only). 
 [10] N. T. Long, "Data Transform", lecture slides, NEU, 2026 (scaling).

### AI usage
- **Tool:** Claude (Anthropic).
- **Part 1, used for:** drafting `src/p1_download.py`, `src/p1_build_storage.py`, `src/p1_storage_benchmark.py`, `src/p1_field_dictionary.py`, the first draft of the report text for Part 1, and the shortened definitions in report Table A1.
- **Part 1, how I checked and changed it:** I read and ran each script. I compared downloaded row counts with the API totals, and field titles with the FDIC dictionary. I tested resumability (re-run on a complete download skipped all 170 quarters; after deleting one quarter only that quarter was downloaded again). I re-added the UTF-8 stdout line from the course starter after a `UnicodeEncodeError` when redirecting output on Windows. I added `RESTYPE1` and `RESDATE` to the Failures download after reading the FDIC dictionary.
- **Part 1, error found and fixed:** in the first version of the benchmark script, `failures.parquet` and `institutions.parquet` were copied into the by-quarter layout, so reading `*.parquet` failed with a schema error (extra column `NAME`). I fixed it by excluding those two files from the copy and re-ran the benchmark.
- **Part 2, used for:** drafting `src/p2_exits.py`, `src/p2_quality_report.py`, `src/p2_build_panel.py` and the first draft of the report text for Part 2.
- **Part 2, how I checked and changed it:** I read and ran each script and checked the output. After the first run I regrouped the `CHANGEC1` codes using the descriptions returned by the API, added a voluntary-closure group, and verified that the exit groups sum to 24,064 banks. I confirmed that the year-to-date conversion gives almost no negative quarterly interest items, and that the panel row count equals the source minus the removed rows.
- **Part 2, issue found by the audit:** integer columns with nulls were saved as text such as "223.0" by my download step; they are cast back to integers before joins.
- **Part 3, used for:** drafting `src/p3_ops.py`, `src/p3_polars_core.py`, `src/p3_benchmark.py`, `src/p3_scaling.py`, `notebooks/p3_spark_colab.ipynb` and the first draft of the report text for Part 3.
- **Part 3, how I checked and changed it:** I read and ran each script; all five operations agree across Polars, pandas and Spark. I added the missing UTF-8 output setting to the benchmark script, removed the Java 17 install step from the notebook after it failed on Colab (Java 21 is preinstalled), and ran the benchmarks with other applications closed.
- **Part 3, issue found:** `LazyFrame.profile()` is not available in Polars 2.1.0, so the profile output was replaced by a note and per-operation timings come from the benchmark.
- **Part 4, used for:** drafting `src/p4_features.py`, `src/p4_pipeline.py`, `src/p4_leakage_audit.py`, the list of features with their motivations, and the first draft of the report text for Part 4.
- **Part 4, how I checked and changed it:** I read and ran each script. I checked that the number of failures per period adds up to the 2,841 failed banks of Part 2, and that the leakage audit passes 7 of 7 tests. I reviewed the feature list and its motivations myself.
- **Part 5, used for:** drafting `src/p5_common.py`, `src/p5_validation.py`, `src/p5_test_once.py`, `notebooks/p5_spark_mllib_colab.ipynb`, `run_all.py` and the first draft of the report text for Parts 5 and 6.
- **Part 5, how I checked and changed it:** I read and ran each script. The model, parameters and threshold were chosen on development data only and committed before the single test run, which I ran once. I checked that the result file stores the hash of the frozen selection. I noticed that the logistic regression is not worse than gradient boosting on the test period (overlapping confidence intervals) and report it as it is, without changing the model afterwards.