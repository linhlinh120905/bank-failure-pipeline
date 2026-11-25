# SOURCE: đặc trưng chỉ dùng dữ liệu đến quý q, nhãn nhìn về quý sau, cửa sổ lag không nhìn tương lai theo bài
#   giảng L03 "Time Series for Finance" [8] (P5, "Leakage Hunt"); lag theo từng ngân hàng tương đương window
#   partitionBy/orderBy trong bài giảng Spark [2] (Problem 7); nhãn y(i,q)=1{đổ vỡ trong (q, q+4 quý]} theo đề;
#   khung CAMELS (vốn, chất lượng tài sản, lợi nhuận, thanh khoản, quy mô) theo hướng dẫn của bài.
# AI: Claude  viết danh mục 29 đặc trưng, hàm dựng đặc trưng theo từng năm (chỉ đọc năm trước và năm
#   hiện tại), nhãn y/y_broad và bản tổng kết. 
import json
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

PANEL_DIR = C.ROOT / "data" / "parquet" / "panel"
EXITS = C.ROOT / "data" / "parquet" / "exits.parquet"
OUT_DIR = C.ROOT / "data" / "parquet" / "features"
REPORTS = C.ROOT / "reports"

HORIZON = 4                      # nhãn nhìn 4 quý về sau
FIRST_QI = 1984 * 4              # qi = năm*4 + (quý-1)
DEV_END = 2006 * 4 + 3           # development kết thúc 2006Q4
GAP_END = 2007 * 4 + 3           # khoảng đệm 2007Q1-2007Q4 (4 quý = horizon nhãn)
LAST_QI = 2025 * 4 + 1           # quý cuối có nhãn đầy đủ: 2025Q2 (+4 quý = 2026Q2, quý cuối của dữ liệu)

RAW = ["ASSET", "EQ", "RBC1AAJ", "RBCRWAJ", "NCLNLS", "LNLSGR", "LNLSNET", "LNATRES", "P9ASSET", "NAASSET",
       "ORE", "NTLNLS_Q", "ELNATR_Q", "NETINC_Q", "NONIX_Q", "INTINC_Q", "EINTEXP_Q", "NONII_Q", "ROA", "NIMY",
       "DEP", "CHBAL", "SC", "BRO", "LNRE", "LNCI", "LNCON"]

# (tên, nhóm, công thức, lý do, dùng trong mô hình mặc định)
CATALOG = [
    ("eq_assets", "Capital", "EQ / ASSET", "Đệm vốn trước thua lỗ; vốn mỏng đi trước đổ vỡ", True),
    ("lev_ratio", "Capital", "RBC1AAJ (0 coi là thiếu)", "Tỷ lệ đòn bẩy Tier 1 theo quy định", True),
    ("eq_assets_chg4", "Capital", "eq_assets - cùng chỉ số 4 quý trước", "Vốn bị bào mòn trong một năm", True),
    ("nclnls_loans", "Asset quality", "NCLNLS / LNLSGR", "Tỷ trọng nợ không sinh lãi trong tổng dư nợ", True),
    ("nclnls_assets", "Asset quality", "NCLNLS / ASSET", "Nợ xấu so với toàn bộ bảng cân đối", True),
    ("p9_assets", "Asset quality", "P9ASSET / ASSET", "Nợ quá hạn 90+ ngày so với tài sản", True),
    ("nonaccrual_assets", "Asset quality", "NAASSET / ASSET", "Tài sản ngừng ghi nhận lãi", True),
    ("ore_assets", "Asset quality", "ORE / ASSET", "Bất động sản thu hồi: dấu hiệu tổn thất tín dụng trước đó", True),
    ("reserve_loans", "Asset quality", "LNATRES / LNLSGR", "Dự phòng tổn thất so với dư nợ", True),
    ("texas_ratio", "Asset quality", "(NCLNLS + ORE) / (EQ + LNATRES), kẹp [0,10]; 10 nếu mẫu số <= 0",
     "Tài sản xấu so với khả năng hấp thụ tổn thất", True),
    ("nco_assets_q", "Asset quality", "NTLNLS_Q / ASSET", "Xóa nợ ròng của quý", True),
    ("provision_assets_q", "Asset quality", "ELNATR_Q / ASSET", "Chi phí dự phòng của quý", True),
    ("nclnls_loans_chg4", "Asset quality", "nclnls_loans - cùng chỉ số 4 quý trước", "Chất lượng nợ xấu đi trong một năm", True),
    ("roa", "Earnings", "ROA (như báo cáo, đã quy năm)", "Khả năng sinh lời", True),
    ("roa_q_ann", "Earnings", "400 * NETINC_Q / ASSET", "Lợi nhuận quý gần nhất (từ số liệu từng quý)", True),
    ("nim", "Earnings", "NIMY", "Biên lãi thuần", True),
    ("overhead_ratio", "Earnings / Management", "NONIX_Q / (INTINC_Q - EINTEXP_Q + NONII_Q), kẹp [0,10]; 10 nếu mẫu số <= 0",
     "Hiệu quả chi phí (đại diện cho quản trị)", True),
    ("roa_chg4", "Earnings", "ROA - cùng chỉ số 4 quý trước", "Biến động khả năng sinh lời", True),
    ("loans_deposits", "Liquidity", "LNLSNET / DEP", "Tỷ lệ cho vay trên tiền gửi", True),
    ("cash_assets", "Liquidity", "CHBAL / ASSET", "Tiền mặt và tiền gửi tại tổ chức khác", True),
    ("securities_assets", "Liquidity", "SC / ASSET", "Chứng khoán nắm giữ", True),
    ("brokered_deposits", "Liquidity", "BRO / DEP", "Phụ thuộc tiền gửi môi giới, nhạy cảm với lãi suất", True),
    ("log_assets", "Size", "ln(ASSET)", "Quy mô", True),
    ("asset_growth4", "Size / growth", "ASSET / ASSET(-4 quý) - 1", "Tăng trưởng nhanh thường đi kèm rủi ro", True),
    ("loan_growth4", "Size / growth", "LNLSGR / LNLSGR(-4 quý) - 1", "Tăng trưởng tín dụng nhanh", True),
    ("loans_assets", "Composition", "LNLSGR / ASSET", "Mức độ tập trung vào cho vay", True),
    ("re_loans_share", "Composition", "LNRE / LNLSGR", "Tập trung bất động sản", True),
    ("ci_loans_share", "Composition", "LNCI / LNLSGR", "Tỷ trọng cho vay thương mại và công nghiệp", True),
    ("consumer_share", "Composition", "LNCON / LNLSGR", "Tỷ trọng cho vay tiêu dùng", True),
    # tùy chọn: có trong bảng, KHÔNG dùng trong mô hình mặc định vì mức thiếu phụ thuộc giai đoạn
    ("lev_missing", "Optional", "1 nếu RBC1AAJ rỗng hoặc bằng 0", "Cờ thiếu; nhận diện giai đoạn", False),
    ("rbc_total", "Optional", "RBCRWAJ (0 coi là thiếu)", "Thiếu 100% giai đoạn 1984-89", False),
    ("rbc_missing", "Optional", "1 nếu RBCRWAJ rỗng hoặc bằng 0", "Cờ thiếu; nhận diện giai đoạn", False),
]
FEATURES = [c[0] for c in CATALOG if c[4]]
OPTIONAL = [c[0] for c in CATALOG if not c[4]]
LABELS = ["y", "y_broad"]
KEYS = ["cert", "qi", "q", "year", "period"]


def build_chunk(df: pl.DataFrame) -> pl.DataFrame:
    """Đặc trưng cho MỌI dòng của `df` (cột cert, qi, q + RAW). Mỗi đặc trưng của dòng (cert, qi) chỉ dùng các
    dòng cùng ngân hàng ở quý qi hoặc trước đó (shift dương); thử nghiệm 'poison' trong p4_leakage_audit.py kiểm tra."""
    df = df.sort("cert", "qi")
    A = pl.col("ASSET")
    LN = pl.col("LNLSGR")
    DEP = pl.col("DEP")
    null = pl.lit(None, dtype=pl.Float64)

    def div(num, den):
        return pl.when(den > 0).then(num / den).otherwise(null)

    def capped(num, den, cap=10.0):   # quy tắc cố định theo miền giá trị, không học từ dữ liệu
        return (pl.when(num.is_null() | den.is_null()).then(null)
                .when(den > 0).then((num / den).clip(0.0, cap))
                .otherwise(pl.lit(cap)))

    def zero_is_missing(c):
        return pl.when(pl.col(c) != 0).then(pl.col(c)).otherwise(null)

    f = df.with_columns(
        div(pl.col("EQ"), A).alias("eq_assets"),
        zero_is_missing("RBC1AAJ").alias("lev_ratio"),
        div(pl.col("NCLNLS"), LN).alias("nclnls_loans"),
        div(pl.col("NCLNLS"), A).alias("nclnls_assets"),
        div(pl.col("P9ASSET"), A).alias("p9_assets"),
        div(pl.col("NAASSET"), A).alias("nonaccrual_assets"),
        div(pl.col("ORE"), A).alias("ore_assets"),
        div(pl.col("LNATRES"), LN).alias("reserve_loans"),
        capped(pl.col("NCLNLS") + pl.col("ORE"), pl.col("EQ") + pl.col("LNATRES")).alias("texas_ratio"),
        div(pl.col("NTLNLS_Q"), A).alias("nco_assets_q"),
        div(pl.col("ELNATR_Q"), A).alias("provision_assets_q"),
        pl.col("ROA").alias("roa"),
        (400.0 * pl.col("NETINC_Q") / A).alias("roa_q_ann"),
        pl.col("NIMY").alias("nim"),
        capped(pl.col("NONIX_Q"), pl.col("INTINC_Q") - pl.col("EINTEXP_Q") + pl.col("NONII_Q")).alias("overhead_ratio"),
        div(pl.col("LNLSNET"), DEP).alias("loans_deposits"),
        div(pl.col("CHBAL"), A).alias("cash_assets"),
        div(pl.col("SC"), A).alias("securities_assets"),
        div(pl.col("BRO"), DEP).alias("brokered_deposits"),
        A.log().alias("log_assets"),
        div(LN, A).alias("loans_assets"),
        div(pl.col("LNRE"), LN).alias("re_loans_share"),
        div(pl.col("LNCI"), LN).alias("ci_loans_share"),
        div(pl.col("LNCON"), LN).alias("consumer_share"),
        (pl.col("RBC1AAJ").is_null() | (pl.col("RBC1AAJ") == 0)).cast(pl.Float64).alias("lev_missing"),
        zero_is_missing("RBCRWAJ").alias("rbc_total"),
        (pl.col("RBCRWAJ").is_null() | (pl.col("RBCRWAJ") == 0)).cast(pl.Float64).alias("rbc_missing"),
    )
    ok = (pl.col("qi") - pl.col("qi").shift(4).over("cert")) == 4          # đủ 4 quý liên tiếp

    def lag4(c):
        return pl.when(ok).then(pl.col(c).shift(4).over("cert"))            # null nếu thiếu quý

    f = f.with_columns(
        (pl.col("eq_assets") - lag4("eq_assets")).alias("eq_assets_chg4"),
        (pl.col("nclnls_loans") - lag4("nclnls_loans")).alias("nclnls_loans_chg4"),
        (pl.col("ROA") - lag4("ROA")).alias("roa_chg4"),
        pl.when(lag4("ASSET") > 0).then(A / lag4("ASSET") - 1).otherwise(null).alias("asset_growth4"),
        pl.when(lag4("LNLSGR") > 0).then(LN / lag4("LNLSGR") - 1).otherwise(null).alias("loan_growth4"),
    )
    return f.select(["cert", "qi", "q", (pl.col("qi") // 4).alias("year")] + FEATURES + OPTIONAL)


def add_labels(feat: pl.DataFrame, ex: pl.DataFrame) -> pl.DataFrame:
    """Ghép nhãn: y = 1 nếu FAILURE xảy ra trong (q, q+4 quý]; y_broad thêm ASSISTANCE và ngân hàng có mã đóng cửa
    nhưng không có trong Failures (phân tích độ nhạy). Sáp nhập không bao giờ là nhãn 1."""
    def evq(c):
        return pl.col(c).dt.year().cast(pl.Int64) * 4 + (pl.col(c).dt.month().cast(pl.Int64) - 1) // 3

    def within(c):
        e = evq(c)
        return ((e > pl.col("qi")) & (e <= pl.col("qi") + HORIZON)).fill_null(False)

    f = feat.join(ex, on="cert", how="left")
    f = f.with_columns(
        within("fail_date").cast(pl.Int8).alias("y"),
        (within("fail_date") | within("assist_date")
         | ((pl.col("exit_type") == "closure_code_not_in_failures") & within("end_date"))
         ).fill_null(False).cast(pl.Int8).alias("y_broad"),
        pl.when(pl.col("qi") <= DEV_END).then(pl.lit("dev"))
        .when(pl.col("qi") <= GAP_END).then(pl.lit("gap")).otherwise(pl.lit("test")).alias("period"),
    )
    return f.select(KEYS + LABELS + FEATURES + OPTIONAL)


def chunk_files(y):
    return [str(f) for yy in (y - 1, y) if (f := PANEL_DIR / f"year={yy}" / "part-0.parquet").exists()]


def read_chunk(y):
    """Năm y và năm y-1 của panel (đủ cho lag 4 quý); không đọc bất kỳ dữ liệu nào sau năm y."""
    return pl.scan_parquet(chunk_files(y)).select(["cert", "qi", "q"] + RAW).collect()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    pl.Config.set_tbl_formatting("ASCII_FULL")
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(12)
    ex = pl.read_parquet(EXITS).select("cert", "fail_date", "assist_date", "end_date", "exit_type")
    t0 = time.perf_counter()
    for y in range(FIRST_QI // 4, LAST_QI // 4 + 1):
        feat = build_chunk(read_chunk(y)).filter((pl.col("year") == y) & (pl.col("qi") <= LAST_QI))
        out = add_labels(feat, ex)
        out.write_parquet(OUT_DIR / f"y{y}.parquet", compression="zstd")
        print(f"{y}: {out.height:>7,} dòng | y=1: {int(out['y'].sum()):>5,}", flush=True)
    print(f"Dựng xong trong {time.perf_counter() - t0:.0f}s")

    lf = pl.scan_parquet(str(OUT_DIR / "*.parquet"))
    panel_rows = pl.scan_parquet(str(PANEL_DIR / "**" / "*.parquet")).select(pl.len()).collect().item()
    by_period = (lf.group_by("period").agg(
        rows=pl.len(), banks=pl.col("cert").n_unique(), positive_rows=pl.col("y").sum(),
        positive_rows_broad=pl.col("y_broad").sum(), first_qi=pl.col("qi").min(), last_qi=pl.col("qi").max())
        .sort("period").collect())
    pos_banks = (lf.filter(pl.col("y") == 1).group_by("period")
                 .agg(banks_with_positive_rows=pl.col("cert").n_unique()).sort("period").collect())
    n_feat_rows = int(by_period["rows"].sum())
    print("\n=== WATERFALL SỐ DÒNG ===")
    print(f"Panel: {panel_rows:,} | bỏ quý sau 2025Q2 (nhãn chưa đủ): {panel_rows - n_feat_rows:,} | bảng đặc trưng: {n_feat_rows:,}")
    print(by_period)
    print(pos_banks)

    exf = pl.read_parquet(EXITS).filter(pl.col("failed"))
    fd = pl.col("fail_date")
    fails = exf.select(
        dev=(fd <= pl.date(2006, 12, 31)).sum(),
        gap_2007=((fd >= pl.date(2007, 1, 1)) & (fd <= pl.date(2007, 12, 31))).sum(),
        test=((fd >= pl.date(2008, 1, 1)) & (fd <= pl.date(2025, 6, 30))).sum(),
        after_2025q2=(fd > pl.date(2025, 6, 30)).sum()).row(0, named=True)
    print("\nSố vụ đổ vỡ (RESTYPE = FAILURE) theo NGÀY đổ vỡ, trang 1 của report:", fails)

    dev = lf.filter(pl.col("period") == "dev")
    rows = []
    for name, grp, formula, why, default in CATALOG:
        r = dev.select(null_rate=pl.col(name).is_null().mean(), p01=pl.col(name).quantile(0.01),
                       median=pl.col(name).median(), p99=pl.col(name).quantile(0.99)).collect().row(0, named=True)
        rows.append({"feature": name, "group": grp, "default": default, **{k: r[k] for k in r}})
    stats = pl.DataFrame(rows)
    print("\n=== Đặc trưng trong development (tỷ lệ rỗng, p01, trung vị, p99) ===")
    print(stats.with_columns(pl.col("null_rate", "p01", "median", "p99").round(3)))

    (REPORTS / "feature_catalog.json").write_text(json.dumps(
        [{"name": n, "group": g, "formula": f, "motivation": w, "default": d} for n, g, f, w, d in CATALOG],
        indent=1, ensure_ascii=False), encoding="utf-8")
    (REPORTS / "features_summary.json").write_text(json.dumps(
        {"panel_rows": panel_rows, "feature_rows": n_feat_rows, "by_period": by_period.to_dicts(),
         "banks_with_positive_rows": pos_banks.to_dicts(), "failures_by_failure_date": fails,
         "dev_feature_stats": stats.to_dicts(), "n_default_features": len(FEATURES),
         "split": {"dev_end_qi": DEV_END, "gap_end_qi": GAP_END, "last_label_qi": LAST_QI}},
        indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print("\nĐã lưu reports/feature_catalog.json và reports/features_summary.json")


if __name__ == "__main__":
    main()
