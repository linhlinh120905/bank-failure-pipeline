# SOURCE: so sánh CSV vs Parquet (dung lượng, đọc vài cột, lọc theo điều kiện) theo bài
#   giảng "Big Data, Spark & ML at Scale", Problem 1 (slide 8-12) và Solution 3b [2].
# AI: Claude viết khung đo: 5 layout, mỗi phép đo chạy 3 lần lấy trung vị, đo
#   thêm RAM đỉnh bằng tiến trình con, dùng cùng một công cụ (Polars) cho mọi định dạng.
import json
import platform
import shutil
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import polars as pl
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

BENCH = C.ROOT / "data" / "bench"
REPORTS = C.ROOT / "reports"
YEAR_DIR = C.ROOT / "data" / "parquet" / "financials"
COLS3 = ["CERT", "REPDTE", "ASSET"]
N_RUNS = 3


def size_mb(p: Path) -> float:
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
    return p.stat().st_size / 1e6


def median_time(fn, n=N_RUNS) -> float:
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return round(st.median(ts), 3)


def build_layouts():
    """Dựng các layout ứng viên từ cùng một dữ liệu (Bronze)."""
    if BENCH.exists():
        shutil.rmtree(BENCH)
    BENCH.mkdir(parents=True)
    df = pl.scan_parquet(sorted(C.RAW.glob("REPDTE=*.parquet"))).sort(["CERT", "REPDTE"]).collect()
    df.write_csv(BENCH / "raw.csv")
    df.write_parquet(BENCH / "single_snappy.parquet", compression="snappy")
    df.write_parquet(BENCH / "single_zstd.parquet", compression="zstd", compression_level=3)
    shutil.copytree(C.RAW, BENCH / "by_quarter_zstd",
                    ignore=shutil.ignore_patterns("*.json", "failures.parquet",
                                                  "institutions.parquet"))
    return df.height

# Mỗi layout: (tên, đường dẫn đọc, loại). Dùng chung Polars để so công bằng giữa định dạng.
def layouts():
    return {
        "CSV raw": ("csv", BENCH / "raw.csv"),
        "Parquet 1 file, snappy": ("pq", BENCH / "single_snappy.parquet"),
        "Parquet 1 file, zstd": ("pq", BENCH / "single_zstd.parquet"),
        "Parquet by quarter (170 files), zstd": ("pq", BENCH / "by_quarter_zstd" / "*.parquet"),
        "Parquet by year (43 files), zstd [chosen]": ("pq", YEAR_DIR / "**" / "*.parquet"),
    }


def scan(kind, path):
    if kind == "csv":
        # schema cố định như Parquet để so công bằng (CSV không tự lưu kiểu)
        schema = {c: (pl.Utf8 if c in C.STR_COLS else pl.Float64) for c in C.FIN_FIELDS}
        return pl.scan_csv(path, schema_overrides=schema)
    return pl.scan_parquet(str(path))


def measure(kind, path):
    t_full = median_time(lambda: scan(kind, path).collect())
    t_cols = median_time(lambda: scan(kind, path).select(COLS3).collect())
    t_filter = median_time(
        lambda: scan(kind, path)
        .filter((pl.col("REPDTE") >= "20080101") & (pl.col("REPDTE") <= "20121231"))
        .select(COLS3).collect())
    return t_full, t_cols, t_filter


def peak_ram_mb(kind, path) -> float:
    """RAM đỉnh khi đọc toàn bộ, đo trong tiến trình con riêng để không bị nhiễu."""
    code = (
        "import polars as pl, psutil, sys\n"
        "kind, path = sys.argv[1], sys.argv[2]\n"
        f"cols = {C.FIN_FIELDS!r}; strc = {C.STR_COLS!r}\n"
        "lf = pl.scan_csv(path, schema_overrides={c:(pl.Utf8 if c in strc else pl.Float64) for c in cols})"
        " if kind=='csv' else pl.scan_parquet(path)\n"
        "df = lf.collect()\n"
        "p = psutil.Process()\n"
        "mi = p.memory_info()\n"
        "print(getattr(mi, 'peak_wset', mi.rss) / 1e6)\n"
    )
    out = subprocess.run([sys.executable, "-c", code, kind, str(path)],
                         capture_output=True, text=True)
    try:
        return round(float(out.stdout.strip().splitlines()[-1]), 1)
    except (ValueError, IndexError):
        return float("nan")


def main():
    n_rows = build_layouts()
    print(f"Dựng xong các layout ({n_rows:,} dòng). Bắt đầu đo, mỗi phép đo chạy {N_RUNS} lần...\n")
    results = {}
    for name, (kind, path) in layouts().items():
        base = path.parent if "*" in str(path) else path
        sz = size_mb(YEAR_DIR if "year" in name else base)
        t_full, t_cols, t_filter = measure(kind, path)
        ram = peak_ram_mb(kind, path)
        results[name] = {"size_mb": round(sz, 1), "read_all_s": t_full,
                         "read_3cols_s": t_cols, "filter_2008_2012_s": t_filter,
                         "peak_ram_mb": ram}
        print(f"{name:<44} {sz:>8.1f} MB | all {t_full:>6.3f}s | 3 cols {t_cols:>6.3f}s | "
              f"filter {t_filter:>6.3f}s | RAM {ram} MB")

    machine = {"cpu": platform.processor(),
               "cores_physical": psutil.cpu_count(logical=False),
               "cores_logical": psutil.cpu_count(logical=True),
               "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
               "os": platform.platform(), "python": platform.python_version(),
               "polars": pl.__version__, "runs_per_measure": N_RUNS}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "storage_benchmark.json").write_text(
        json.dumps({"machine": machine, "rows": n_rows, "results": results}, indent=1))
    print("\nĐã lưu reports/storage_benchmark.json")
    shutil.rmtree(BENCH)   # xóa bản CSV tạm cho nhẹ ổ đĩa
    print("Đã xóa data/bench/")


if __name__ == "__main__":
    main()