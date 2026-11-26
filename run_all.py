# AI: Claude (Anthropic) viết bộ chạy tuần tự toàn bộ pipeline. 
"""Chạy lại pipeline từ đầu đến cuối. Ví dụ: python run_all.py --list | python run_all.py | python run_all.py --with-benchmarks"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORE = [("download", "src/p1_download.py"), ("storage", "src/p1_build_storage.py"), ("fields", "src/p1_field_dictionary.py"),
        ("exits", "src/p2_exits.py"), ("quality", "src/p2_quality_report.py"), ("panel", "src/p2_build_panel.py"),
        ("polars", "src/p3_polars_core.py"), ("features", "src/p4_features.py"), ("pipeline", "src/p4_pipeline.py"),
        ("audit", "src/p4_leakage_audit.py"), ("validation", "src/p5_validation.py")]
BENCH = [("storage_benchmark", "src/p1_storage_benchmark.py"), ("benchmark", "src/p3_benchmark.py"), ("scaling", "src/p3_scaling.py")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="chỉ liệt kê các bước")
    ap.add_argument("--skip-download", action="store_true", help="bỏ bước tải (dữ liệu đã có trong data/raw)")
    ap.add_argument("--with-benchmarks", action="store_true", help="chạy thêm 3 benchmark nặng (khoảng 30 phút, đóng ứng dụng khác)")
    ap.add_argument("--with-test", action="store_true", help="chạy đánh giá test MỘT LẦN (p5_test_once.py --run)")
    a = ap.parse_args()
    steps = [(n, s, []) for n, s in CORE if not (a.skip_download and n == "download")]
    if a.with_benchmarks:
        steps += [(n, s, []) for n, s in BENCH]
    if a.with_test:
        steps.append(("test", "src/p5_test_once.py", ["--run"]))
    for i, (n, s, extra) in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {n}: python {s} {' '.join(extra)}".rstrip())
    if a.list:
        return
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for i, (n, s, extra) in enumerate(steps, 1):
        if n == "test" and (ROOT / "reports" / "p5_test_result.json").exists():
            print("Bỏ qua test: reports/p5_test_result.json đã có (kết quả của lần chạy test duy nhất). "
                  "Muốn tái lập: xóa file đó rồi chạy lại, script sẽ kiểm tra mã băm của p5_dev_selection.json.")
            continue
        t = time.time()
        print(f"\n===== [{i}/{len(steps)}] {n} =====", flush=True)
        r = subprocess.run([sys.executable, s, *extra], cwd=ROOT, env=env)
        if r.returncode != 0:
            sys.exit(f"Bước '{n}' lỗi (mã {r.returncode}). Dừng.")
        print(f"----- {n} xong trong {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
