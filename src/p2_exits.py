# SOURCE: ý nghĩa FAILDATE, RESTYPE, RESTYPE1, ACTIVE, CHANGEC1, ENDEFYMD lấy từ dictionary
#   FDIC failure_properties.md và institution_properties.md [5]; mô tả mã CHANGEC1 lấy từ
#   API FDIC "history", field CHANGECODE_DESC [6]; lazy scan_parquet theo bài giảng [2].
# AI: Claude viết quy tắc phân loại thoát, các bảng kiểm tra chéo và bộ nhớ đệm
#  Tôi đã đọc code, chạy, gom lại nhóm mã CHANGEC1 theo mô tả từ API và kiểm tra tổng các nhóm = 24.064 ngân hàng.
import json
import sys
from pathlib import Path

import polars as pl
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

FIN_GLOB = str(C.ROOT / "data" / "parquet" / "financials" / "**" / "*.parquet")
OUT = C.ROOT / "data" / "parquet" / "exits.parquet"
REPORTS = C.ROOT / "reports"
CODES_JSON = REPORTS / "changec1_codes.json"
SUMMARY_JSON = REPORTS / "exits_summary.json"

# Quy tắc theo mô tả mã 
MERGER_CODES = [213, 221, 222, 223, 224, 225]   # merger/absorption/consolidation, assisted or not
CLOSURE_CODES = [211, 215, 216, 217, 230, 235]  # failure-like closing, receivership, payoff
VOLUNTARY_CODES = [240]                          # closing - voluntary
NON_EXIT_CODES = [110, 440, 470, 510, 520, 610, 810, 811, 820]  # rename, relocation, acquirer side...


def to_int(name):
    """Chuỗi như '223.0' hoặc '1234' -> số nguyên; giá trị lạ -> null."""
    return pl.col(name).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)


def to_date(name):
    """Thử dạng M/D/YYYY rồi dạng YYYY-MM-DD; không đọc được -> null (sẽ được đếm)."""
    s = pl.col(name)
    return pl.coalesce(s.str.to_date("%m/%d/%Y", strict=False),
                       s.str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False))


def code_descriptions(codes):
    """Tra CHANGECODE_DESC qua API history (1 mẫu mỗi mã); lưu đệm vào reports/."""
    cache = json.loads(CODES_JSON.read_text(encoding="utf-8")) if CODES_JSON.exists() else {}
    for c in codes:
        if str(c) in cache:
            continue
        try:
            r = requests.get(f"{C.BASE}/history", timeout=60,
                             params={"filters": f"CHANGECODE:{c}",
                                     "fields": "CHANGECODE,CHANGECODE_DESC", "limit": 1})
            cache[str(c)] = r.json()["data"][0]["data"]["CHANGECODE_DESC"]
        except Exception as exc:  # không lưu mã tra thất bại, lần sau thử lại
            print(f"  (không tra được mã {c}: {type(exc).__name__})")
    REPORTS.mkdir(exist_ok=True)
    CODES_JSON.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    return cache


def main():
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_formatting("ASCII_FULL")
    pl.Config.set_fmt_str_lengths(70)
    REPORTS.mkdir(exist_ok=True)

    # 1) Financials: mỗi ngân hàng báo cáo từ quý nào đến quý nào (lazy, chỉ đọc 2 cột)
    fin = (pl.scan_parquet(FIN_GLOB)
           .select(to_int("CERT").alias("cert"),
                   pl.col("REPDTE").str.to_date("%Y%m%d").alias("q"))
           .group_by("cert")
           .agg(first_q=pl.col("q").min(), last_q=pl.col("q").max(), n_quarters=pl.len())
           .collect())
    final_q = fin["last_q"].max()
    print(f"{fin.height:,} ngân hàng trong Financials; quý cuối của dữ liệu: {final_q}")

    # 2) Institutions (một dòng mỗi ngân hàng)
    raw = pl.read_parquet(C.RAW / "institutions.parquet")
    inst = raw.select(
        to_int("CERT").alias("cert"), to_int("ACTIVE").alias("active"),
        to_int("CHANGEC1").alias("changec1"), to_date("ENDEFYMD").alias("end_date"),
        (pl.col("ENDEFYMD").is_not_null()
         & (pl.col("ENDEFYMD").str.strip_chars() != "")).alias("has_end_raw"))
    n_inst_dup = inst.height - inst["cert"].n_unique()
    n_end_unparsed = inst.filter(pl.col("has_end_raw") & pl.col("end_date").is_null()).height
    inst = inst.drop("has_end_raw").unique(subset=["cert"], keep="first", maintain_order=True)

    # 3) Failures: tách FAILURE (charter đóng) và ASSISTANCE (chủ yếu charter còn)
    f = pl.read_parquet(C.RAW / "failures.parquet").select(
        to_int("CERT").alias("cert"), pl.col("NAME").alias("name"),
        to_date("FAILDATE").alias("fail_date"), pl.col("RESTYPE").alias("restype"),
        pl.col("RESTYPE1").alias("restype1"),
        pl.col("COST").cast(pl.Float64, strict=False).alias("cost"))
    n_nocert_all = f.filter(pl.col("cert").is_null()).height
    n_nocert_1984 = f.filter(pl.col("cert").is_null()
                             & (pl.col("fail_date") >= pl.date(1984, 1, 1))).height
    n_faildate_bad = f.filter(pl.col("fail_date").is_null()).height
    f = f.filter(pl.col("cert").is_not_null())
    fails = (f.filter(pl.col("restype") == "FAILURE").sort("fail_date")
             .group_by("cert", maintain_order=True)
             .agg(fail_date=pl.col("fail_date").first(), restype1=pl.col("restype1").first(),
                  cost=pl.col("cost").first(), n_failure_rows=pl.len()))
    assists = (f.filter(pl.col("restype") == "ASSISTANCE").sort("fail_date")
               .group_by("cert", maintain_order=True)
               .agg(assist_date=pl.col("fail_date").first(), n_assist_rows=pl.len()))
    unmatched = (f.filter(pl.col("fail_date") >= pl.date(1984, 1, 1))
                 .join(fin.select("cert"), on="cert", how="anti"))

    # 4) Nối và phân loại cách thoát
    ex = (fin.join(inst, on="cert", how="left")
          .join(fails, on="cert", how="left").join(assists, on="cert", how="left"))
    n_no_inst = fin.join(inst.select("cert"), on="cert", how="anti").height
    ex = ex.with_columns(pl.col("fail_date").is_not_null().alias("failed"),
                         pl.col("assist_date").is_not_null().alias("assisted"),
                         (pl.col("last_q") == final_q).alias("reports_final_q"))
    ex = ex.with_columns(
        pl.when(pl.col("failed")).then(pl.lit("failure"))
        .when(pl.col("reports_final_q")).then(pl.lit("active"))
        .when(pl.col("changec1").is_in(MERGER_CODES)).then(pl.lit("merger"))
        .when(pl.col("changec1").is_in(CLOSURE_CODES)).then(pl.lit("closure_code_not_in_failures"))
        .when(pl.col("changec1").is_in(VOLUNTARY_CODES)).then(pl.lit("voluntary_closure"))
        .otherwise(pl.lit("other_exit")).alias("exit_type"))
    counts = ex.group_by("exit_type").len().sort("len", descending=True)
    print("\n[A] Cách thoát của từng ngân hàng (quy tắc: Failures = căn cứ đổ vỡ):")
    print(counts)

    # 5) Bảng mã CHANGEC1 với mô tả và nhóm quy tắc
    codes = (ex.filter(pl.col("changec1").is_not_null())
             .group_by("changec1").len().sort("len", descending=True))
    desc = code_descriptions(codes["changec1"].to_list())
    code_list = codes["changec1"].to_list()
    tbl = pl.DataFrame({
        "changec1": code_list, "n_banks": codes["len"].to_list(),
        "description": [desc.get(str(c)) for c in code_list],
        "rule_group": ["merger" if c in MERGER_CODES else "closure" if c in CLOSURE_CODES
                       else "voluntary" if c in VOLUNTARY_CODES
                       else "non_exit_event" if c in NON_EXIT_CODES else "unassigned"
                       for c in code_list]})
    print("\n[B] Mọi mã CHANGEC1 (mô tả từ API history); 'unassigned' = chưa xếp nhóm:")
    print(tbl)

    # 6) Kiểm tra chéo (liên quan trực tiếp tới 'mergers are not failures')
    print("\n[C] Ngân hàng NẰM trong Failures (FAILURE), theo CHANGEC1 của Institutions:")
    print(ex.filter(pl.col("failed")).group_by("changec1").len().sort("len", descending=True))
    n_unlisted = int((ex["exit_type"] == "closure_code_not_in_failures").sum())
    print(f"\n[D] Mã đóng cửa {CLOSURE_CODES} nhưng KHÔNG có trong Failures: {n_unlisted}")
    print("\n[E] ACTIVE so với 'có báo cáo ở quý cuối':")
    print(ex.group_by("active", "reports_final_q").len().sort("len", descending=True))
    g = (ex.filter(pl.col("failed"))
         .with_columns((pl.col("fail_date") - pl.col("last_q")).dt.total_days().alias("gap_days")))
    print("\n[F] Khoảng cách (ngày) từ báo cáo cuối đến FAILDATE, ngân hàng đổ vỡ:")
    print(g.select(n=pl.len(), reports_after_faildate=(pl.col("gap_days") < 0).sum(),
                   gap_over_1y=(pl.col("gap_days") > 366).sum(),
                   median=pl.col("gap_days").median(), p90=pl.col("gap_days").quantile(0.9),
                   min=pl.col("gap_days").min(), max=pl.col("gap_days").max()))
    print(f"\n[G] FAILURE/ASSISTANCE từ 1984 có CERT nhưng KHÔNG có báo cáo trong Financials: {unmatched.height}")
    print(unmatched.select("cert", "name", "fail_date", "restype", "restype1")
          .sort("fail_date").head(40))
    print("\n[H] Chất lượng dữ liệu nguồn:")
    print(f"  Failures: CERT null = {n_nocert_all} (từ 1984: {n_nocert_1984}); FAILDATE không đọc được = {n_faildate_bad}")
    print(f"  Institutions: CERT trùng = {n_inst_dup}; ENDEFYMD không đọc được = {n_end_unparsed}")
    print(f"  Ngân hàng có trong Financials nhưng không có trong Institutions = {n_no_inst}")

    # 7) Lưu kết quả
    ex.write_parquet(OUT, compression="zstd")
    summary = {
        "final_quarter": str(final_q), "n_banks": ex.height,
        "exit_counts": {r["exit_type"]: r["len"] for r in counts.iter_rows(named=True)},
        "merger_codes": MERGER_CODES, "closure_codes": CLOSURE_CODES,
        "n_failed": int(ex["failed"].sum()), "n_assisted": int(ex["assisted"].sum()),
        "closure_code_not_in_failures": n_unlisted,
        "failures_without_financials": unmatched.height,
        "failures_cert_null_all": n_nocert_all, "failures_cert_null_from_1984": n_nocert_1984,
        "failures_faildate_unparsed": n_faildate_bad,
        "institutions_cert_duplicates": n_inst_dup, "institutions_enddate_unparsed": n_end_unparsed,
        "financials_not_in_institutions": n_no_inst,
        "changec1_table": tbl.to_dicts(),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str),
                            encoding="utf-8")
    print(f"\nĐã lưu {OUT.name} ({ex.height:,} ngân hàng) và reports/exits_summary.json")


if __name__ == "__main__":
    main()