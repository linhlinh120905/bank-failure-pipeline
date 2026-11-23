# SOURCE: lazy scan_parquet, explain() và projection/predicate pushdown theo bài giảng Spark [2],
#   Solution 3b ("Eager vs. Lazy", plan.explain()).
# AI: Claude viết khung chạy 5 phép, lưu kế hoạch thực thi, kiểm tra khớp pandas và
#   xuất dữ liệu cho Spark.
import json
import platform
import sys
import time
from pathlib import Path

import polars as pl
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3_ops as P  # noqa: E402

C = P.C
sys.stdout.reconfigure(encoding="utf-8")
REPORTS = C.ROOT / "reports"


def main():
    pl.Config.set_tbl_formatting("ASCII_FULL")
    REPORTS.mkdir(exist_ok=True)
    lf, ex = pl.scan_parquet(P.PANEL_GLOB), pl.scan_parquet(str(P.EXITS))
    plans, results, times = [], {}, {}

    print("=== 5 phép bằng Polars lazy (một lần chạy, chỉ để tạo kết quả tham chiếu) ===")
    for op in P.OPS:
        q = P.pl_query(op, lf, ex)
        naive, opt = q.explain(optimized=False), q.explain()
        plans.append(f"########## {op} : KẾ HOẠCH THÔ (chưa tối ưu)\n{naive}\n"
                     f"########## {op} : KẾ HOẠCH ĐÃ TỐI ƯU\n{opt}\n")
        t = time.perf_counter()
        r = q.collect()
        times[op] = round(time.perf_counter() - t, 3)
        results[op] = r.to_dicts()
        ok = P.same(P.ck_pl(op, r), P.pd_query(op, *P.load_pandas(op, tuned=True)))
        print(f"{op:<16} Polars {times[op]:>6.2f}s | khớp pandas: {ok}")
        print(f"--- kế hoạch đã tối ưu của {op} ---\n{opt}\n")

        prof_txt = ("########## profile: LazyFrame.profile() is not available in Polars 2.1.0 "
                "(AttributeRemovedError); per-operation timings are measured in p3_benchmark.py\n")
    (REPORTS / "p3_polars_plans.txt").write_text("\n".join(plans) + "\n" + prof_txt, encoding="utf-8")

    # Dữ liệu nhỏ cho Spark (tải lên Colab; KHÔNG commit)
    slim = lf.select(P.ALL_COLS).sort("cert", "qi").collect()
    slim.write_parquet(C.ROOT / "data" / "p3_spark_input.parquet", compression="zstd")
    pl.read_parquet(P.EXITS).select("cert", "exit_type").write_parquet(
        C.ROOT / "data" / "p3_spark_exits.parquet")
    size = (C.ROOT / "data" / "p3_spark_input.parquet").stat().st_size / 1e6
    print(f"Đã xuất data/p3_spark_input.parquet: {slim.height:,} dòng, {slim.width} cột, {size:.1f} MB")

    try:
        threads = pl.thread_pool_size()
    except Exception:
        threads = None
    machine = {"cpu_cores_physical": psutil.cpu_count(logical=False),
               "cpu_cores_logical": psutil.cpu_count(logical=True),
               "ram_gb": round(psutil.virtual_memory().total / 1e9, 1), "os": platform.platform(),
               "python": platform.python_version(), "polars": pl.__version__,
               "polars_thread_pool": threads}
    out = {"machine": machine, "panel_rows": slim.height, "single_run_seconds": times,
           "results": results}
    (REPORTS / "p3_polars_results.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print("Đã lưu reports/p3_polars_results.json và reports/p3_polars_plans.txt")


if __name__ == "__main__":
    main()