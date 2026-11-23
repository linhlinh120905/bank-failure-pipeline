# SOURCE: ý tưởng kiểm tra giá trị rỗng, trùng lặp và ngoại lai theo quy tắc IQR lấy từ bài giảng
#   L02 "Data Cleaning" (03_L02_Data_clearning.pdf, phần Outlier IQR, slide 19-21) [7]; ngưỡng
#   3×IQR theo từng năm và phép thử YTD (so Q4 với Q1) là lựa chọn của tôi; ý nghĩa field theo
#   dictionary FDIC [3], [5]; lazy scan_parquet theo bài giảng [2].
# # AI: Claude viết báo cáo chất lượng tự động gồm 11 mục kiểm tra (mục [11] kiểm tra bước nhảy của trung vị theo năm).
#   Tôi đã đọc code, chạy, đối chiếu mục [7] (YTD) với phép kiểm tra một ngân hàng ở Part 1 và đối chiếu số liệu với report.
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

FIN_GLOB = str(C.ROOT / "data" / "parquet" / "financials" / "**" / "*.parquet")
EXITS = C.ROOT / "data" / "parquet" / "exits.parquet"
OUT = C.ROOT / "reports" / "quality_report.json"

NUM = [c for c in C.FIN_FIELDS if c not in C.STR_COLS]          # 30 cột số
FLOW = ["NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR", "NTLNLS"]
RATIO = ["ROA", "ROE", "NIMY", "RBC1AAJ", "RBCRWAJ"]
CONTROL = ["ASSET", "ROA"]   # đối chứng: số dư và tỷ lệ annualized không phải YTD


def to_int(name):
    return pl.col(name).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)


def base_lazy():
    return (pl.scan_parquet(FIN_GLOB)
            .with_columns(to_int("CERT").alias("cert"),
                          pl.col("REPDTE").str.to_date("%Y%m%d").alias("q"))
            .with_columns(pl.col("q").dt.year().alias("yr"), pl.col("q").dt.month().alias("mo")))


def r3(v):
    return None if v is None else round(float(v), 4)


def sec_basic(lf):
    n_rows = lf.select(pl.len()).collect().item()
    cert_bad = lf.select(pl.col("cert").is_null().sum()).collect().item()
    dup = (lf.group_by("cert", "REPDTE").len().filter(pl.col("len") > 1)
           .select(pl.len()).collect().item())
    per_q = lf.group_by("q").len().sort("q").collect()
    r = {"rows": n_rows, "cert_not_integer": int(cert_bad), "duplicate_cert_repdte": int(dup),
         "quarters": per_q.height, "banks_per_quarter_min": int(per_q["len"].min()),
         "banks_per_quarter_median": float(per_q["len"].median()),
         "banks_per_quarter_max": int(per_q["len"].max()),
         "first_quarter_banks": int(per_q["len"][0]), "last_quarter_banks": int(per_q["len"][-1])}
    print("\n[1] Cơ bản:", json.dumps(r, ensure_ascii=False))
    return r


def sec_nulls(lf):
    cols = NUM + ["STNAME"]
    overall = lf.select([pl.col(c).is_null().mean().alias(c) for c in cols]).collect().row(0, named=True)
    zeros = lf.select([(pl.col(c) == 0).mean().alias(c) for c in NUM]).collect().row(0, named=True)
    by_year = (lf.group_by("yr").agg([pl.col(c).is_null().mean().alias(c) for c in cols])
               .sort("yr").collect())
    years = by_year["yr"].to_list()
    rows, by_year_json = [], {}
    for c in cols:
        rates = by_year[c].to_list()
        by_year_json[c] = {str(y): round(v, 3) for y, v in zip(years, rates)}
        rows.append({"field": c, "null_rate": r3(overall[c]),
                     "zero_share_of_non_null": r3(zeros.get(c)),
                     "first_year_null_below_50pct": next((y for y, v in zip(years, rates) if v < 0.5), None),
                     "n_years_all_null": sum(1 for v in rates if v >= 0.99)})
    df = pl.DataFrame(rows)
    print("\n[2] Rỗng và số 0 theo field (null_rate trên toàn panel):")
    print(df)
    return {"fields": rows, "null_rate_by_year": by_year_json}


def sec_flags(lf):
    f = lf.select(
        (pl.col("ASSET").is_null() | (pl.col("ASSET") <= 0)).sum().alias("asset_null_or_le0"),
        (pl.col("DEP") < 0).sum().alias("dep_negative"),
        (pl.col("EQ") > pl.col("ASSET")).sum().alias("eq_gt_asset"),
        (pl.col("NCLNLS") < 0).sum().alias("nclnls_negative"),
        (pl.col("LNLSGR") < 0).sum().alias("lnlsgr_negative"),
        (pl.col("ROA").abs() > 100).sum().alias("abs_roa_gt_100"),
        (pl.col("RBC1AAJ").abs() > 100).sum().alias("abs_rbc1aaj_gt_100"),
    ).collect().row(0, named=True)
    r = {k: int(v) for k, v in f.items()}
    print("\n[3] Giá trị vô lý (chỉ đánh dấu, không xóa):", json.dumps(r))
    return r


def sec_gaps(ex):
    g = ex.with_columns(
        (((pl.col("last_q").dt.year() - pl.col("first_q").dt.year()).cast(pl.Int64) * 4
          + (pl.col("last_q").dt.month().cast(pl.Int64)
             - pl.col("first_q").dt.month().cast(pl.Int64)) // 3 + 1)
         - pl.col("n_quarters").cast(pl.Int64)).alias("missing_q"))
    with_gap = g.filter(pl.col("missing_q") > 0)
    dist = (g.group_by(pl.col("missing_q").clip(0, 5).alias("missing_q_capped5")).len()
            .sort("missing_q_capped5"))
    by_type = with_gap.group_by("exit_type").len().sort("len", descending=True)
    print("\n[4] Quý bị thiếu giữa chừng của từng ngân hàng (5 = từ 5 quý trở lên):")
    print(dist)
    print(by_type)
    return {"banks_with_gaps": with_gap.height, "missing_quarters_total": int(g["missing_q"].sum()),
            "distribution": dist.to_dicts(), "gap_banks_by_exit_type": by_type.to_dicts()}


def sec_outliers(lf):
    rows = []
    for c in NUM:
        d = lf.select(pl.col("REPDTE").str.slice(0, 4).alias("y"), pl.col(c)).collect()
        d = d.with_columns(q1=pl.col(c).quantile(0.25).over("y"),
                           q3=pl.col(c).quantile(0.75).over("y"))
        d = d.with_columns(iqr=pl.col("q3") - pl.col("q1"))
        ok = d.filter(pl.col(c).is_not_null() & (pl.col("iqr") > 0))
        n_out = ok.filter((pl.col(c) < pl.col("q1") - 3 * pl.col("iqr"))
                          | (pl.col(c) > pl.col("q3") + 3 * pl.col("iqr"))).height
        rows.append({"field": c, "n_checked": ok.height,
                     "share_outside_3iqr_within_year": r3(n_out / ok.height) if ok.height else None,
                     "min": r3(d[c].min()), "max": r3(d[c].max())})
    print("\n[5] Ngoại lai: tỷ lệ ngoài [Q1-3·IQR, Q3+3·IQR] tính riêng từng năm (chỉ đánh dấu):")
    print(pl.DataFrame(rows).sort("share_outside_3iqr_within_year", descending=True, nulls_last=True))
    return rows


def sec_units(lf):
    out = []
    for c in RATIO:
        r = lf.select(pl.col(c).min().alias("min"), pl.col(c).quantile(0.01).alias("p01"),
                      pl.col(c).median().alias("p50"), pl.col(c).quantile(0.99).alias("p99"),
                      pl.col(c).max().alias("max")).collect().row(0, named=True)
        out.append({"field": c, **{k: r3(v) for k, v in r.items()}})
    print("\n[6] Thang đo của các tỷ lệ (kỳ vọng đơn vị %):")
    print(pl.DataFrame(out))
    return out


def sec_ytd(lf):
    cols = FLOW + CONTROL
    d = lf.filter(pl.col("mo").is_in([3, 12])).select("cert", "yr", "mo", *cols).collect()
    a = d.filter(pl.col("mo") == 3).drop("mo")
    b = d.filter(pl.col("mo") == 12).drop("mo")
    j = a.join(b, on=["cert", "yr"], suffix="_q4")
    rows = []
    for c in cols:
        v = j.filter(pl.col(c).is_not_null() & pl.col(f"{c}_q4").is_not_null() & (pl.col(c) > 0))
        n = v.height
        rows.append({"field": c, "role": "control" if c in CONTROL else "flow",
                     "n_bank_years": n,
                     "share_q4_gt_q1": r3((v[f"{c}_q4"] > v[c]).mean()) if n else None,
                     "median_q4_over_q1": r3((v[f"{c}_q4"] / v[c]).median()) if n else None})
    print("\n[7] Phép thử YTD trên toàn panel (Q1 > 0). Nếu là YTD: median Q4/Q1 gần 4, share_q4_gt_q1 gần 1:")
    print(pl.DataFrame(rows))
    return rows

def sec_stability(lf):
    cols = ["ASSET", "EQ", "DEP", "LNLSGR", "ROA", "ROE", "NIMY", "RBC1AAJ", "RBCRWAJ"]
    d = lf.group_by("yr").agg([pl.col(c).median().alias(c) for c in cols]).sort("yr").collect()
    jumps = []
    for c in cols:
        t = (d.select("yr", pl.col(c).alias("v")).with_columns(prev=pl.col("v").shift(1))
             .filter((pl.col("v") > 0) & (pl.col("prev") > 0))
             .with_columns(ratio=pl.col("v") / pl.col("prev")))
        f = t.filter((pl.col("ratio") > 1.5) | (pl.col("ratio") < 1 / 1.5))
        jumps.append({"field": c, "years": f["yr"].to_list(),
                      "ratio": [round(x, 2) for x in f["ratio"].to_list()]})
    print("\n[11] Ổn định theo thời gian: trung vị từng năm; ứng viên bước nhảy khi trung vị đổi > 50% so với năm trước:")
    print(d.select("yr", *[pl.col(c).round(2) for c in ["ASSET", "ROA", "NIMY", "RBC1AAJ", "RBCRWAJ"]]))
    print(pl.DataFrame(jumps))
    return {"yearly_median": d.to_dicts(), "yoy_jumps_over_50pct": jumps}

def sec_after_faildate(lf, ex):
    fb = ex.filter(pl.col("failed")).select("cert", "fail_date", "restype1")
    j = lf.select("cert", "q").collect().join(fb, on="cert")
    before = (j.filter(pl.col("q") <= pl.col("fail_date")).group_by("cert")
              .agg(last_before=pl.col("q").max(), n_before=pl.len()))
    after = (j.filter(pl.col("q") > pl.col("fail_date")).group_by("cert")
             .agg(first_after=pl.col("q").min(), last_after=pl.col("q").max(), n_after=pl.len()))
    d = (after.join(before, on="cert", how="left").join(fb, on="cert", how="left")
         .with_columns((pl.col("last_after") - pl.col("fail_date")).dt.total_days().alias("max_days_after"),
                       (pl.col("first_after") - pl.col("last_before")).dt.total_days().alias("hole_days")))
    d = d.with_columns(
        pl.when(pl.col("max_days_after") <= 100).then(pl.lit("a) <=100 days"))
        .when(pl.col("max_days_after") <= 366).then(pl.lit("b) 101-366 days"))
        .otherwise(pl.lit("c) >366 days")).alias("bucket"))
    by_bucket = (d.group_by("bucket").agg(pl.len().alias("banks"), pl.col("n_after").sum().alias("rows"))
                 .sort("bucket"))
    by_type = (d.group_by("restype1").agg(pl.len().alias("banks"), pl.col("n_after").sum().alias("rows"))
               .sort("banks", descending=True))
    worst = (d.sort("max_days_after", descending=True).head(15)
             .select("cert", "restype1", "fail_date", "last_before", "first_after", "last_after",
                     "n_before", "n_after", "max_days_after", "hole_days"))
    r = {"banks_failed": fb.height, "banks_with_rows_after_faildate": d.height,
         "rows_after_faildate": int(d["n_after"].sum()),
         "banks_with_hole_over_1y": d.filter(pl.col("hole_days") > 366).height,
         "banks_with_no_report_before_faildate": d.filter(pl.col("n_before").is_null()).height,
         "by_bucket": by_bucket.to_dicts(), "by_restype1": by_type.to_dicts()}
    print("\n[8] Dòng báo cáo SAU FAILDATE (ngân hàng đổ vỡ):")
    print({k: v for k, v in r.items() if not k.startswith("by_")})
    print(by_bucket)
    print(by_type)
    print("15 trường hợp lệch nhiều nhất (hole_days > 366 nghĩa là báo cáo ngắt quãng rồi lại có):")
    print(worst)
    return r


def sec_exits(ex):
    other = (ex.filter(pl.col("exit_type") == "other_exit")
             .group_by(pl.col("active").is_null().alias("not_in_institutions"), "changec1")
             .len().sort("len", descending=True))
    unlisted = (ex.filter(pl.col("exit_type") == "closure_code_not_in_failures")
                .group_by("changec1", (pl.col("last_q").dt.year() // 5 * 5).alias("last_report_5y"))
                .len().sort("changec1", "last_report_5y"))
    r = {"banks_with_multiple_failure_rows": ex.filter(pl.col("n_failure_rows") > 1).height,
         "failed_and_also_assisted": ex.filter(pl.col("failed") & pl.col("assisted")).height,
         "other_exit_breakdown": other.to_dicts(), "closure_code_not_in_failures": unlisted.to_dicts()}
    print("\n[9] Điểm bất thường của bảng exits:")
    print({k: v for k, v in r.items() if not isinstance(v, list)})
    print("other_exit chia theo (không có trong Institutions?, CHANGEC1):")
    print(other)
    print("Mã đóng cửa nhưng không có trong Failures, theo mã và giai đoạn của báo cáo cuối:")
    print(unlisted)
    return r


def sec_formats(lf):
    def dz(df, col):
        s = df[col].drop_nulls().cast(pl.Utf8)
        return {"n_non_null": int(s.len()), "ends_with_dot0": int(s.str.ends_with(".0").sum())}
    fl = pl.read_parquet(C.RAW / "failures.parquet")
    ins = pl.read_parquet(C.RAW / "institutions.parquet")
    fin = lf.select(pl.col("CERT").is_not_null().sum().alias("n"),
                    pl.col("CERT").str.ends_with(".0").sum().alias("dot0")).collect().row(0, named=True)
    r = {"failures.CERT": dz(fl, "CERT"), "institutions.CERT": dz(ins, "CERT"),
         "institutions.CHANGEC1": dz(ins, "CHANGEC1"), "institutions.ACTIVE": dz(ins, "ACTIVE"),
         "financials.CERT": {"n_non_null": int(fin["n"]), "ends_with_dot0": int(fin["dot0"])}}
    print("\n[10] Định dạng số nguyên bị lưu thành chữ dạng '223.0':")
    print(json.dumps(r, indent=1))
    return r


def main():
    pl.Config.set_tbl_formatting("ASCII_FULL")
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(20)
    ex = pl.read_parquet(EXITS)
    lf = base_lazy()
    rep = {"basic": sec_basic(lf), "nulls": sec_nulls(lf), "flags": sec_flags(lf),
           "quarter_gaps": sec_gaps(ex), "outliers_3iqr": sec_outliers(lf),
           "ratio_scales": sec_units(lf), "ytd_test": sec_ytd(lf),
           "after_faildate": sec_after_faildate(lf, ex), "exits_checks": sec_exits(ex),
           "id_formats": sec_formats(lf), "stability": sec_stability(lf)}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nĐã lưu {OUT}")


if __name__ == "__main__":
    main()
  