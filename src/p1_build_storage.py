# SOURCE: ý tưởng lưu Parquet theo cột, nén, và lazy scan_parquet (projection/predicate
#   pushdown) theo bài giảng "Big Data, Spark & ML at Scale", Part I (Problem 1) và
#   Part II (Solution 3b) [2]; lớp Bronze (raw) -> Silver (đã chuẩn hóa) theo
#   Medallion architecture, Solution 2a [2].
# AI: Claude viết khung gộp theo năm, zstd, sort theo CERT/REPDTE và kiểm tra
#   số dòng sau khi ghi. 
import shutil
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

OUT = C.ROOT / "data" / "parquet" / "financials"   # Silver: partition theo năm


def main():
    files = sorted(C.RAW.glob("REPDTE=*.parquet"))
    print(f"Đọc {len(files)} file quý từ {C.RAW}")
    t0 = time.perf_counter()

    # Lazy: chỉ mô tả kế hoạch, chưa đọc dữ liệu cho tới khi collect()
    lf = (
        pl.scan_parquet(files)
        .with_columns(pl.col("REPDTE").str.slice(0, 4).alias("year"))
        .sort(["CERT", "REPDTE"])
    )
    df = lf.collect()   # ~1,68 triệu dòng x 34 cột, khoảng 450MB RAM: vừa với máy 8GB

    if OUT.exists():
        shutil.rmtree(OUT)   # dựng lại từ đầu để chạy lại luôn ra cùng kết quả
    OUT.mkdir(parents=True)

    # Ghi từng năm vào thư mục year=YYYY/ (kiểu Hive)
    for year, part in df.partition_by("year", as_dict=True, maintain_order=True).items():
        y = year[0] if isinstance(year, tuple) else year
        d = OUT / f"year={y}"
        d.mkdir(parents=True, exist_ok=True)
        part.drop("year").write_parquet(d / "part-0.parquet",
                                        compression="zstd", compression_level=3)

    # Kiểm tra: số dòng đọc lại phải bằng số dòng gốc
    n_out = pl.scan_parquet(str(OUT / "**" / "*.parquet")).select(pl.len()).collect().item()
    size_mb = sum(f.stat().st_size for f in OUT.rglob("*.parquet")) / 1e6
    n_files = len(list(OUT.rglob("*.parquet")))
    print(f"Dòng gốc: {df.height:,} | dòng sau khi ghi: {n_out:,} | "
          f"{'KHỚP' if n_out == df.height else 'LỆCH!'}")
    print(f"{n_files} file, {size_mb:.1f} MB, mất {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()