# SOURCE: so sánh pandas (eager) với Polars (lazy), đo thời gian và RAM, chạy nhiều lần lấy trung vị
#   theo yêu cầu đề; ý tưởng "đọc ít cột, lọc sớm" theo bài giảng Spark [2], Problem 1 và 3;
#   đo RAM đỉnh bằng tiến trình con như ở p1_storage_benchmark.py.
# AI: Claude viết bộ đo 4 engine x 5 phép, mỗi lần một tiến trình riêng, bỏ lần chạy
#   đầu, đối chiếu kết quả giữa các engine. 
import argparse
import json
import platform
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3_ops as P  # noqa: E402

C = P.C
sys.stdout.reconfigure(encoding="utf-8")
N_RUNS = 3          # số lần đo tính điểm; thêm 1 lần chạy đầu bị bỏ (khởi động bộ nhớ đệm đĩa)


def worker(engine, op):
    """Chạy một phép một lần trong tiến trình riêng, in một dòng JSON. Thời gian nhập thư viện không tính."""
    if engine.startswith("pandas"):
        import pandas  # noqa: F401
        import pyarrow.parquet  # noqa: F401  (nạp trước để pandas không bị tính thời gian nạp pyarrow)
    else:
        import polars  # noqa: F401
    proc = psutil.Process()
    base = proc.memory_info().rss
    t = time.perf_counter()
    ck = P.run(engine, op)
    dt = time.perf_counter() - t
    mi = proc.memory_info()
    peak = getattr(mi, "peak_wset", mi.rss)        # Windows: đỉnh working set
    num = {k: (float(v) if v is not None else float("nan")) for k, v in ck.items()}
    print(json.dumps({"t": dt, "base_mb": base / 1e6, "peak_mb": peak / 1e6, "ck": num}))


def run_once(engine, op):
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", engine, op],
                       capture_output=True, text=True)
    lines = r.stdout.strip().splitlines()
    try:
        return json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        raise RuntimeError(f"{engine}/{op} lỗi:\n{r.stderr[-900:]}")


def main():
    mem = psutil.virtual_memory()
    print(f"Bộ nhớ đang dùng trước khi đo: {mem.percent}% (nên dưới ~70%; đóng bớt ứng dụng nếu cao hơn)")
    res, check = {}, {}
    for op in P.OPS:
        for eng in P.ENGINES:
            runs = [run_once(eng, op) for _ in range(N_RUNS + 1)][1:]      # bỏ lần đầu
            ts = [x["t"] for x in runs]
            delta = [x["peak_mb"] - x["base_mb"] for x in runs]
            res[f"{op}|{eng}"] = {
                "t_median": round(st.median(ts), 3), "t_min": round(min(ts), 3),
                "t_max": round(max(ts), 3), "ram_delta_mb_median": round(st.median(delta), 0),
                "ram_peak_mb_median": round(st.median([x["peak_mb"] for x in runs]), 0)}
            check[f"{op}|{eng}"] = runs[-1]["ck"]
            print(f"{op:<16} {eng:<16} {res[f'{op}|{eng}']['t_median']:>7.3f}s "
                  f"(min {min(ts):.3f}, max {max(ts):.3f}) | RAM +{res[f'{op}|{eng}']['ram_delta_mb_median']:.0f} MB")
    agree = {k: P.same(v, check[f"{k.split('|')[0]}|polars_lazy"]) for k, v in check.items()}
    print("\nMọi engine cho cùng kết quả với polars_lazy:", all(agree.values()))
    if not all(agree.values()):
        print("  KHÁC:", [k for k, v in agree.items() if not v])

    print("\n=== Trung vị thời gian (s) và hệ số nhanh hơn so với pandas_naive ===")
    print(f"{'phép':<16}" + "".join(f"{e:>17}" for e in P.ENGINES) + f"{'naive/lazy':>13}")
    for op in P.OPS:
        row = [res[f"{op}|{e}"]["t_median"] for e in P.ENGINES]
        print(f"{op:<16}" + "".join(f"{v:>17.3f}" for v in row) + f"{row[0] / row[3]:>12.1f}x")
    print("\n=== RAM tăng thêm khi chạy (MB, trung vị) ===")
    for op in P.OPS:
        print(f"{op:<16}" + "".join(f"{res[f'{op}|{e}']['ram_delta_mb_median']:>17.0f}" for e in P.ENGINES))

    machine = {"cores_physical": psutil.cpu_count(logical=False),
               "cores_logical": psutil.cpu_count(logical=True),
               "ram_gb": round(mem.total / 1e9, 1), "memory_used_pct_before": mem.percent,
               "os": platform.platform(), "python": platform.python_version(),
               "runs_scored": N_RUNS, "warmup_runs_dropped": 1}
    (C.ROOT / "reports" / "p3_benchmark.json").write_text(
        json.dumps({"machine": machine, "results": res, "all_agree": all(agree.values())}, indent=1),
        encoding="utf-8")
    print("\nĐã lưu reports/p3_benchmark.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=2, metavar=("ENGINE", "OP"))
    a = ap.parse_args()
    worker(*a.worker) if a.worker else main()