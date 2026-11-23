# SOURCE: ý tưởng chỉ dùng thông tin đã biết tại quý q (point-in-time) và không nhìn tương lai theo
#   bài giảng L03 "Time Series for Finance", P5 và P7 [8]; phép shift theo từng ngân hàng
#   (over("cert")) tương đương window function partitionBy/orderBy trong bài giảng Spark,
#   Problem 7 [2]; ý nghĩa FAILDATE theo dictionary FDIC failure_properties.md [5].
#   AI: Claude viết quy tắc cắt dòng sau FAILDATE, phép đổi YTD thành số từng quý và
#   các phép kiểm tra sau khi dựng panel. 
# Tôi đã đọc code, chạy và đối chiếu waterfall: 1.682.738 - 1.092 = 1.681.646, đọc lại khớp, 0 dòng sau FAILDATE.
import json
import shutil
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

FIN_GLOB = str(C.ROOT / "data" / "parquet" / "financials" / "**" / "*.parquet")
EXITS = C.ROOT / "data" / "parquet" / "exits.parquet"
PANEL = C.ROOT / "data" / "parquet" / "panel"
SUMMARY = C.ROOT / "reports" / "panel_summary.json"
FLOW = ["NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR", "NTLNLS"]   # số cộng dồn từ đầu năm


def to_int(name):
    return pl.col(name).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)


def main():
    pl.Config.set_tbl_formatting("ASCII_FULL")
    pl.Config.set_tbl_rows(60)

    # 1) Nạp Financials (cert là số nguyên, q là ngày cuối quý) và bảng exits (chỉ để cắt dòng)
    ex = pl.read_parquet(EXITS).select("cert", "first_q", "fail_date", "failed")
    fin = (pl.scan_parquet(FIN_GLOB)
           .with_columns(to_int("CERT").alias("cert"),
                         pl.col("REPDTE").str.to_date("%Y%m%d").alias("q"))
           .drop("CERT", "REPDTE").collect())
    n_in = fin.height
    fin = fin.with_columns(pl.col("q").dt.year().cast(pl.Int64).alias("yr"),
                           pl.col("q").dt.month().cast(pl.Int64).alias("mo"))
    fin = fin.with_columns((pl.col("yr") * 4 + (pl.col("mo") - 1) // 3).alias("qi"))   # chỉ số quý liên tục
    fin = (fin.join(ex, on="cert", how="left")
           .with_columns(pl.col("failed").fill_null(False))
           .sort("cert", "qi"))

    # 2) Cắt dòng SAU FAILDATE, chỉ với ngân hàng có báo cáo vào hoặc trước FAILDATE
    in_window = pl.col("failed") & (pl.col("first_q") <= pl.col("fail_date"))
    after = in_window & (pl.col("q") > pl.col("fail_date"))
    dropped = fin.filter(after)
    n_after, n_after_banks = dropped.height, dropped["cert"].n_unique()
    prewin = fin.filter(pl.col("failed") & (pl.col("first_q") > pl.col("fail_date")))
    n_pre_rows, n_pre_banks = prewin.height, prewin["cert"].n_unique()
    fin = fin.filter(~after)
    n_left_after = fin.filter(in_window & (pl.col("q") > pl.col("fail_date"))).height   # phải bằng 0

    # 3) YTD -> số từng quý, chỉ dùng quý hiện tại và quý liền trước (không nhìn tương lai)
    consecutive = (pl.col("qi") - pl.col("qi").shift(1).over("cert")) == 1
    flows = []
    for c in FLOW:
        ytd = pl.col(c)
        flows.append(pl.when(pl.col("mo") == 3).then(ytd)                       # Q1: giữ nguyên
                     .when(consecutive).then(ytd - ytd.shift(1).over("cert"))   # quý sau: trừ quý trước cùng năm
                     .otherwise(None)                                           # thiếu quý trước: để rỗng
                     .alias(f"{c}_Q"))
    fin = fin.with_columns(flows)
    fin = fin.with_columns((pl.col("qi") - pl.col("qi").shift(1).over("cert")).alias("gap_before"))
    fin = fin.rename({c: f"{c}_YTD" for c in FLOW})

    # 4) Kiểm tra
    n_dup = fin.height - fin.select("cert", "qi").unique().height
    flow_checks = {}
    for c in FLOW:
        s = fin[f"{c}_Q"]
        nn = s.drop_nulls()
        flow_checks[c] = {"null_rate": round(float(s.is_null().mean()), 4),
                          "share_negative": round(float((nn < 0).mean()), 4)}
    n_gap_rows = int(fin.select((pl.col("gap_before") > 1).sum()).item())
    zero_year = (fin.group_by("yr")
                 .agg((pl.col("RBC1AAJ") == 0).mean().round(3).alias("rbc1aaj_zero"),
                      (pl.col("RBCRWAJ") == 0).mean().round(3).alias("rbcrwaj_zero"),
                      pl.col("P3ASSET").is_null().mean().round(3).alias("p3asset_null"))
                 .sort("yr"))

    # 5) Ghi panel theo năm (không chứa FAILDATE, loại thoát hay bất kỳ thông tin tương lai nào)
    out = fin.drop("mo", "failed", "first_q", "fail_date")
    if PANEL.exists():
        shutil.rmtree(PANEL)
    for key, part in out.partition_by("yr", as_dict=True, maintain_order=True).items():
        y = key[0] if isinstance(key, tuple) else key
        d = PANEL / f"year={y}"
        d.mkdir(parents=True, exist_ok=True)
        part.drop("yr").write_parquet(d / "part-0.parquet", compression="zstd", compression_level=3)
    n_back = pl.scan_parquet(str(PANEL / "**" / "*.parquet")).select(pl.len()).collect().item()
    size_mb = sum(f.stat().st_size for f in PANEL.rglob("*.parquet")) / 1e6

    print("\n=== WATERFALL SỐ DÒNG ===")
    print(f"Dòng Financials ban đầu                         : {n_in:,}")
    print(f"- Dòng sau FAILDATE bị cắt ({n_after_banks} ngân hàng)        : {n_after:,}")
    print(f"= Dòng của panel                                 : {out.height:,}")
    print(f"Đọc lại từ đĩa                                   : {n_back:,} ({'KHỚP' if n_back == out.height else 'LỆCH!'})")
    print(f"Ngân hàng đổ vỡ TRƯỚC khi có dữ liệu, giữ nguyên : {n_pre_banks} ngân hàng, {n_pre_rows} dòng")
    print(f"Dòng còn lại sau FAILDATE (phải = 0)            : {n_left_after}")
    print(f"Trùng (cert, qi) (phải = 0)                     : {n_dup}")
    print(f"Dòng có khoảng trống quý trước đó (gap_before>1): {n_gap_rows:,}")
    print(f"Dung lượng panel: {size_mb:.1f} MB, cột: {out.width}")
    print("\nSố từng quý: tỷ lệ rỗng và tỷ lệ âm (kỳ vọng ít âm với INTINC, EINTEXP, NONIX nếu YTD đúng):")
    print(pl.DataFrame([{"field": f"{k}_Q", **v} for k, v in flow_checks.items()]))
    print("\nTỷ lệ số 0 của RBC1AAJ, RBCRWAJ và tỷ lệ rỗng của P3ASSET theo năm:")
    print(zero_year)

    SUMMARY.parent.mkdir(exist_ok=True)
    SUMMARY.write_text(json.dumps({
        "rows_in": n_in, "rows_dropped_after_faildate": n_after, "banks_with_dropped_rows": n_after_banks,
        "rows_panel": out.height, "rows_read_back": n_back, "rows_left_after_faildate": n_left_after,
        "prewindow_failed_banks_kept": n_pre_banks, "prewindow_failed_rows_kept": n_pre_rows,
        "duplicate_cert_qi": n_dup, "rows_with_gap_before": n_gap_rows,
        "flow_columns": FLOW, "flow_checks": flow_checks, "size_mb": round(size_mb, 1),
        "columns": out.columns}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã lưu panel tại {PANEL} và {SUMMARY.name}")


if __name__ == "__main__":
    main()