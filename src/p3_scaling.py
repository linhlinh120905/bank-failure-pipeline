# SOURCE: định luật Amdahl S(N) = 1/((1-p) + p/N) và cách ước lượng phần song song p từ số đo theo
#   bài giảng Spark [2], Problem 4 (Hand Calc 3); shuffle là bước tốn nhất và khi nào cần cluster
#   theo [2], Problem 5-6; đo RAM đỉnh trong tiến trình con như p3_benchmark.py.
# AI: Claude viết hai thí nghiệm: nhân dữ liệu 1x-8x (có chốt chặn RAM) và đổi số
#   luồng Polars kèm ước lượng Amdahl. 
import argparse
import json
import math
import os
import shutil
import statistics as st
import subprocess
import sys
from pathlib import Path

import polars as pl
import psutil
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3_ops as P  # noqa: E402

C = P.C
sys.stdout.reconfigure(encoding="utf-8")

SLIM = C.ROOT / "data" / "p3_spark_input.parquet"        # panel rút gọn 12 cột, 1.681.646 dòng
SCALE_DIR = C.ROOT / "data" / "scale"
BENCH = Path(__file__).resolve().parent / "p3_benchmark.py"
OFFSET = 1_000_000          # đổi cert của mỗi bản sao để chúng là ngân hàng khác nhau
SCALES = [1, 2, 4, 8]
OPS_S = ["yearly_ratios", "lag_features", "ytd_to_quarter", "join_exits"]
ENG_S = ["pandas_tuned", "polars_lazy"]
N_SCORED = 2                # 1 lần khởi động (bỏ) + 2 lần tính điểm
BUDGET_GB = 6.0             # RAM dành cho dữ liệu trên laptop 8 GB (phần còn lại cho hệ điều hành)


def make_data(k):
    """Nhân bản panel và bảng exits k lần (cert + i*OFFSET); ghi từng bản một để tiết kiệm RAM."""
    SCALE_DIR.mkdir(parents=True, exist_ok=True)
    pf, ef = SCALE_DIR / f"panel_x{k}.parquet", SCALE_DIR / f"exits_x{k}.parquet"
    if pf.exists() and ef.exists():
        return pf, ef
    base = pl.read_parquet(SLIM)
    writer = None
    for i in range(k):
        t = base.with_columns((pl.col("cert") + i * OFFSET).alias("cert")).to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(str(pf), t.schema, compression="zstd")
        writer.write_table(t)
    writer.close()
    ex = pl.read_parquet(P.EXITS).select("cert", "exit_type")
    pl.concat([ex.with_columns((pl.col("cert") + i * OFFSET).alias("cert")) for i in range(k)]
              ).write_parquet(ef, compression="zstd")
    return pf, ef


def run_once(engine, op, env_extra):
    env = {**os.environ, **env_extra}
    r = subprocess.run([sys.executable, str(BENCH), "--worker", engine, op],
                       capture_output=True, text=True, env=env)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        raise RuntimeError(f"{engine}/{op} lỗi:\n{r.stderr[-600:]}")


def measure(engine, op, env_extra):
    runs = [run_once(engine, op, env_extra) for _ in range(N_SCORED + 1)][1:]
    return {"t_median": round(st.median(x["t"] for x in runs), 3),
            "ram_delta_mb": round(st.median(x["peak_mb"] - x["base_mb"] for x in runs), 0),
            "ck": runs[-1]["ck"]}


def scale_experiment(max_scale):
    res, skipped, dead, prev = {}, [], set(), {}
    for k in [s for s in SCALES if s <= max_scale]:
        pf, ef = make_data(k)
        rows = pq.ParquetFile(str(pf)).metadata.num_rows
        env = {"P3_PANEL": str(pf), "P3_EXITS": str(ef)}
        print(f"\n--- nhân {k}x: {rows:,} dòng ---")
        for op in OPS_S:
            cks = {}
            for eng in ENG_S:
                key = f"{op}|{eng}"
                avail = psutil.virtual_memory().available / 1e6
                if key in dead:
                    continue
                if key in prev:
                    pk, pd_ = prev[key]
                    pred = pd_ * (k / pk) * 1.15
                    if pred > 0.8 * avail:
                        dead.add(key)
                        skipped.append({"key": key, "scale": k, "predicted_mb": round(pred),
                                        "available_mb": round(avail)})
                        print(f"{key:<32} BỎ QUA: dự đoán cần {pred:.0f} MB > 80% bộ nhớ còn trống ({avail:.0f} MB)")
                        continue
                m = measure(eng, op, env)
                cks[eng] = m.pop("ck")
                res[f"{key}|{k}"] = {"scale": k, "rows": rows, **m}
                prev[key] = (k, max(m["ram_delta_mb"], 1.0))
                print(f"{key:<32} {m['t_median']:>7.3f}s | RAM +{m['ram_delta_mb']:.0f} MB")
            if len(cks) == 2:
                ok = P.same(cks["pandas_tuned"], cks["polars_lazy"], rtol=1e-6)
                res[f"{op}|agree|{k}"] = bool(ok)
                if not ok:
                    print(f"  CẢNH BÁO: hai engine cho kết quả khác nhau ở {op} nhân {k}x")
    return res, skipped


def thread_experiment(scale):
    pf, ef = make_data(scale)
    env_base = {"P3_PANEL": str(pf), "P3_EXITS": str(ef)}
    out, ops = {}, ["yearly_ratios", "lag_features", "ytd_to_quarter"]
    print(f"\n--- đổi số luồng Polars (nhân {scale}x) ---")
    for n in [1, 2, 4]:
        rep = subprocess.run([sys.executable, "-c", "import polars as pl; print(pl.thread_pool_size())"],
                             env={**os.environ, "POLARS_MAX_THREADS": str(n)},
                             capture_output=True, text=True).stdout.strip()
        for op in ops:
            m = measure("polars_lazy", op, {**env_base, "POLARS_MAX_THREADS": str(n)})
            m.pop("ck")
            out[f"{op}|{n}"] = {"threads_requested": n, "threads_reported": rep, **m}
            print(f"{op:<16} {n} luồng (Polars báo {rep}): {m['t_median']:>7.3f}s")
    amdahl = {}
    for op in ops:
        t1 = out[f"{op}|1"]["t_median"]
        for n in [2, 4]:
            s = t1 / out[f"{op}|{n}"]["t_median"]
            p = min(1.0, max(0.0, (1 - 1 / s) / (1 - 1 / n))) if s > 1 else 0.0
            amdahl[f"{op}|{n}"] = {"speedup": round(s, 2), "parallel_fraction_p": round(p, 2),
                                   "max_speedup_inf_cores": round(1 / (1 - p), 1) if p < 1 else None}
    print("\nƯớc lượng Amdahl (p = phần song song được; S_max = 1/(1-p) khi số lõi vô hạn):")
    for k, v in amdahl.items():
        print(f"{k:<24} tăng tốc {v['speedup']}x | p ≈ {v['parallel_fraction_p']} | S_max ≈ {v['max_speedup_inf_cores']}")
    return out, amdahl


def extrapolate(res):
    print(f"\n=== Ngoại suy với ngân sách RAM {BUDGET_GB:.0f} GB (từ lần đo lớn nhất của mỗi cặp) ===")
    ex, base_rows = {}, 1_681_646
    for op in OPS_S:
        for eng in ENG_S:
            pts = sorted((v["rows"], v["t_median"], v["ram_delta_mb"]) for k, v in res.items()
                         if k.startswith(f"{op}|{eng}|") and isinstance(v, dict))
            if len(pts) < 2:
                continue
            (r0, t0, _), (r1, t1, m1) = pts[0], pts[-1]
            slope = math.log(t1 / t0) / math.log(r1 / r0)
            mb_per_m = m1 / (r1 / 1e6)
            max_rows = BUDGET_GB * 1000 / mb_per_m * 1e6
            ex[f"{op}|{eng}"] = {"time_slope": round(slope, 2), "ram_mb_per_million_rows": round(mb_per_m),
                                 "max_rows_in_budget_millions": round(max_rows / 1e6, 1),
                                 "times_current_panel": round(max_rows / base_rows, 1)}
            print(f"{op:<16} {eng:<13} độ dốc thời gian {slope:.2f} | {mb_per_m:.0f} MB/triệu dòng "
                  f"| vừa ~{max_rows / 1e6:.0f} triệu dòng (~{max_rows / base_rows:.1f}x panel hiện tại)")
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-scale", type=int, default=8)
    ap.add_argument("--keep", action="store_true", help="giữ data/scale/ sau khi chạy")
    a = ap.parse_args()
    mem = psutil.virtual_memory()
    print(f"Bộ nhớ đang dùng: {mem.percent}%, còn trống {mem.available / 1e9:.1f} GB "
          f"(nên dưới ~65%; đóng ứng dụng khác)")
    res, skipped = scale_experiment(a.max_scale)
    thr, amdahl = thread_experiment(min(4, a.max_scale))
    extra = extrapolate(res)
    out = {"machine": {"cores_physical": psutil.cpu_count(logical=False),
                       "cores_logical": psutil.cpu_count(logical=True),
                       "ram_gb": round(mem.total / 1e9, 1), "memory_used_pct_before": mem.percent,
                       "polars": pl.__version__},
           "scale": res, "skipped": skipped, "threads": thr, "amdahl": amdahl, "extrapolation": extra}
    (C.ROOT / "reports" / "p3_scaling.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nĐã lưu reports/p3_scaling.json")
    if not a.keep:
        shutil.rmtree(SCALE_DIR, ignore_errors=True)
        print("Đã xóa data/scale/")


if __name__ == "__main__":
    main()