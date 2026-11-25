# SOURCE: các lỗi rò rỉ cần kiểm tra (cột biết sau sự kiện, fit trên toàn mẫu, nhìn quý sau, chia ngẫu nhiên) theo
#   bài giảng L03 [8] ("Leakage Hunt"), L02 [9] (Module 1) và đề (Leakage gate); thử nghiệm "poison" (làm hỏng dữ liệu
#   tương lai rồi kiểm tra đặc trưng quá khứ không đổi) là thiết kế của tôi.
# AI: Claude viết 7 phép kiểm tra rò rỉ tự động. 
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p4_features as F  # noqa: E402
import p4_pipeline as PP  # noqa: E402

C = F.C
sys.stdout.reconfigure(encoding="utf-8")
BANNED = {"fail_date", "assist_date", "end_date", "exit_type", "changec1", "active", "restype", "restype1", "cost",
          "failed", "assisted", "reports_final_q", "last_q", "first_q", "n_quarters", "n_failure_rows",
          "n_assist_rows", "name"}
results = []


def record(name, ok, detail):
    results.append({"test": name, "pass": bool(ok), "detail": detail})
    print(f"[{'ĐẠT' if ok else 'LỖI'}] {name}: {detail}", flush=True)


def poison(df, q0):
    """Thay mọi giá trị số ở các quý SAU q0 bằng số rất khác (giữ nguyên khóa cert, qi)."""
    return df.with_columns([pl.when(pl.col("qi") > q0).then(pl.col(c) * 7.0 + 1.0e6).otherwise(pl.col(c)).alias(c)
                            for c in F.RAW])


def main():
    cols = F.FEATURES + F.OPTIONAL
    tbl = PP.load_features()

    # 1) Không có cột biết sau sự kiện trong bảng đặc trưng
    bad = sorted((set(c.lower() for c in tbl.columns) | set(cols)) & BANNED)
    bad_name = [c for c in cols if "fail" in c.lower() or "exit" in c.lower()]
    record("1. Không có cột 'biết sau' trong bảng đặc trưng", not bad and not bad_name,
           f"{len(tbl.columns)} cột; cột cấm: {bad or 'không'}; tên đáng ngờ: {bad_name or 'không'}")

    # 2) Poison: đặc trưng ở quý <= q0 không đổi khi dữ liệu sau q0 bị làm hỏng
    ok_all, n_tests, sensitive_all = True, 0, True
    for y in (1989, 2009, 2019):
        df = F.read_chunk(y)
        base = F.build_chunk(df)
        for q0 in (y * 4, y * 4 + 1, y * 4 + 2):
            alt = F.build_chunk(poison(df, q0))
            same = base.filter(pl.col("qi") <= q0).select(cols).equals(alt.filter(pl.col("qi") <= q0).select(cols))
            moved = not base.filter(pl.col("qi") > q0).select(cols).equals(alt.filter(pl.col("qi") > q0).select(cols))
            ok_all, sensitive_all, n_tests = ok_all and same, sensitive_all and moved, n_tests + 1
    record("2. Thử nghiệm 'poison' (đặc trưng quá khứ không phụ thuộc dữ liệu tương lai)", ok_all and sensitive_all,
           f"{n_tests} phép thử (năm 1989, 2009, 2019 x 3 mốc q0); quá khứ không đổi: {ok_all}; "
           f"phép thử có độ nhạy (tương lai bị đổi thì đặc trưng tương lai đổi): {sensitive_all}")

    # 3) Nhãn: tính lại độc lập bằng phép tính NGÀY (không dùng chỉ số quý)
    ex = pl.read_parquet(F.EXITS).select("cert", "fail_date", "assist_date", "first_q", "failed", "exit_type")
    t = tbl.select("cert", "qi", "q", "y", "y_broad").join(ex, on="cert", how="left")
    exp = ((pl.col("fail_date") > pl.col("q")) & (pl.col("fail_date") <= pl.col("q").dt.offset_by("1y"))).fill_null(False)
    mism = t.filter(exp != (pl.col("y") == 1)).height
    nonfail_pos = t.filter((pl.col("y") == 1) & ~pl.col("failed").fill_null(False)).height
    broad_lt = t.filter(pl.col("y_broad") < pl.col("y")).height
    pos = int(t["y"].sum())
    record("3. Nhãn y khớp phép tính ngày độc lập; chỉ ngân hàng đổ vỡ có y=1; y_broad chứa y",
           mism == 0 and nonfail_pos == 0 and broad_lt == 0 and pos > 0,
           f"sai lệch {mism}; y=1 ở ngân hàng không đổ vỡ {nonfail_pos}; y_broad < y: {broad_lt}; số dòng y=1: {pos:,}")

    # 4) Chia theo thời gian, khoảng đệm, không dòng sau FAILDATE, không quý sau 2025Q2
    g = tbl.group_by("period").agg(lo=pl.col("qi").min(), hi=pl.col("qi").max()).sort("period")
    p = {r["period"]: (r["lo"], r["hi"]) for r in g.iter_rows(named=True)}
    horizon_ok = p["dev"][1] + F.HORIZON <= F.GAP_END
    order_ok = p["dev"][1] < p["gap"][0] <= p["gap"][1] < p["test"][0]
    gap_len = p["gap"][1] - p["gap"][0] + 1
    after_fail = t.filter(pl.col("failed").fill_null(False) & (pl.col("first_q") <= pl.col("fail_date"))
                          & (pl.col("q") > pl.col("fail_date"))).height
    max_qi = int(tbl["qi"].max())
    record("4. Chia theo thời gian: dev < gap(4 quý) < test; không dòng sau FAILDATE; không quý sau 2025Q2",
           horizon_ok and order_ok and gap_len == 4 and after_fail == 0 and max_qi <= F.LAST_QI,
           f"dev {p['dev']}, gap {p['gap']} ({gap_len} quý), test {p['test']}; nhãn của dev nhìn tới "
           f"qi {p['dev'][1] + F.HORIZON} <= hết gap {F.GAP_END}: {horizon_ok}; dòng sau FAILDATE {after_fail}; qi lớn nhất {max_qi}")

    # 5) Fit chỉ trên development: làm hỏng gap/test rồi fit lại -> tham số không đổi
    sub = tbl.select(["period"] + F.FEATURES).filter(pl.int_range(pl.len()) % 4 == 0)   # lấy mẫu 1/4 cho nhanh
    pre0, feats = PP.fit_preprocessor(sub)
    pert = sub.with_columns([pl.when(pl.col("period") != "dev").then(pl.col(c) * 10 + 1000).otherwise(pl.col(c)).alias(c)
                             for c in feats])
    pre1, _ = PP.fit_preprocessor(pert)
    p0, p1 = PP.params(pre0), PP.params(pre1)
    same = all(np.array_equal(p0[k], p1[k], equal_nan=True) for k in p0)
    contaminated = sub.with_columns(pl.lit("dev").alias("period"))               # nếu fit nhầm trên cả gap và test
    p2 = PP.params(PP.fit_preprocessor(contaminated)[0])
    shift = float(np.nanmax([np.nanmax(np.abs(p0[k] - p2[k]) / np.maximum(np.abs(p0[k]), 1e-9)) for k in ("median", "mean")]))
    record("5. Tiền xử lý chỉ fit trên development (làm hỏng gap/test, fit lại: tham số y hệt)", same,
           f"tham số không đổi: {same}; để so sánh, fit nhầm trên cả gap+test sẽ làm trung vị/trung bình lệch tới {shift:.1%}")

    # 6) Trung vị của bộ impute đúng bằng trung vị tính độc lập trên dev
    xd = PP.to_matrix(sub.filter(pl.col("period") == "dev"), feats)
    ok6 = np.allclose(p0["median"], np.nanmedian(xd, axis=0), equal_nan=True)
    record("6. Trung vị impute = trung vị tính độc lập trên dev", ok6, f"{len(feats)} đặc trưng khớp: {ok6}")

    # 7) Chất lượng: không đặc trưng nào rỗng hoàn toàn hoặc hằng số trong dev
    dev = tbl.filter(pl.col("period") == "dev")
    allnull = [c for c in F.FEATURES if dev[c].null_count() == dev.height]
    const = [c for c in F.FEATURES if dev[c].drop_nulls().n_unique() <= 1]
    record("7. Không đặc trưng rỗng hoàn toàn hoặc hằng số trong dev", not allnull and not const,
           f"rỗng hoàn toàn: {allnull or 'không'}; hằng số: {const or 'không'}")

    (C.ROOT / "reports" / "leakage_audit.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    n_pass = sum(r["pass"] for r in results)
    print(f"\nKẾT QUẢ: {n_pass}/{len(results)} phép kiểm tra ĐẠT. Đã lưu reports/leakage_audit.json")
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
