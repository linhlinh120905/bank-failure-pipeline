# SOURCE: lazy scan, projection/predicate pushdown theo bài giảng Spark [2], Solution 3b;
#   window function partitionBy/orderBy với lag theo bài giảng Spark [2], Problem 7;
#   đặc trưng chỉ dùng quá khứ (shift dương, không dùng dữ liệu quý sau) theo bài giảng L03 [8], P5.
# AI: Claude viết 5 phép biến đổi tương đương cho Polars và pandas
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

PANEL_DIR = C.ROOT / "data" / "parquet" / "panel"
PANEL_GLOB = str(PANEL_DIR / "**" / "*.parquet")
EXITS = C.ROOT / "data" / "parquet" / "exits.parquet"
QI0, QI1 = 2008 * 4, 2012 * 4 + 3          # 2008Q1..2012Q4; qi = năm*4 + (quý-1)
FLOWS = ["NETINC", "INTINC", "NONIX"]
OPS = ["filter_project", "yearly_ratios", "lag_features", "ytd_to_quarter", "join_exits"]
ENGINES = ["pandas_naive", "pandas_tuned", "polars_read_all", "polars_lazy"]
COLS = {
    "filter_project": ["cert", "qi", "ASSET", "EQ", "NETINC_Q", "NCLNLS"],
    "yearly_ratios": ["qi", "ASSET", "EQ", "NCLNLS", "NETINC_Q"],
    "lag_features": ["cert", "qi", "ASSET", "ROA"],
    "ytd_to_quarter": ["cert", "qi"] + [f"{f}_YTD" for f in FLOWS] + [f"{f}_Q" for f in FLOWS],
    "join_exits": ["cert", "ASSET"],
}
ALL_COLS = sorted({c for v in COLS.values() for c in v})


def pl_query(op, lf, ex):
    """Phép `op` bằng Polars; lf, ex là LazyFrame; kết quả là bảng nhỏ (chưa collect)."""
    import polars as pl
    if op == "filter_project":
        return (lf.select(COLS[op])
                .filter((pl.col("qi") >= QI0) & (pl.col("qi") <= QI1))
                .select(n=pl.len(), s_asset=pl.col("ASSET").sum(), s_eq=pl.col("EQ").sum(),
                        s_nclnls=pl.col("NCLNLS").sum(), s_netinc=pl.col("NETINC_Q").sum()))
    if op == "yearly_ratios":
        return (lf.select(COLS[op])
                .select((pl.col("qi") // 4).alias("yr"),
                        (pl.col("EQ") / pl.col("ASSET")).alias("eq"),
                        (pl.col("NCLNLS") / pl.col("ASSET")).alias("npl"),
                        (pl.col("NETINC_Q") / pl.col("ASSET")).alias("roaq"))
                .group_by("yr")
                .agg(n=pl.len(), eq=pl.col("eq").mean(), npl=pl.col("npl").mean(),
                     roaq=pl.col("roaq").mean())
                .sort("yr"))
    if op == "lag_features":
        q = (lf.select(COLS[op]).sort("cert", "qi")
             .with_columns(a4=pl.col("ASSET").shift(4).over("cert"),
                           r4=pl.col("ROA").shift(4).over("cert"),
                           q4=pl.col("qi").shift(4).over("cert"))
             .with_columns(ok=(pl.col("qi") - pl.col("q4")) == 4)      # đủ 4 quý liên tiếp
             .with_columns(g=pl.when(pl.col("ok")).then(pl.col("ASSET") / pl.col("a4") - 1),
                           d=pl.when(pl.col("ok")).then(pl.col("ROA") - pl.col("r4"))))
        return q.select(n_valid=pl.col("g").count(), s_growth=pl.col("g").sum(),
                        m_growth=pl.col("g").mean(), s_droa=pl.col("d").sum(),
                        m_droa=pl.col("d").mean())
    if op == "ytd_to_quarter":
        q = lf.select(COLS[op]).sort("cert", "qi")
        cons = (pl.col("qi") - pl.col("qi").shift(1).over("cert")) == 1
        q = q.with_columns([
            pl.when(pl.col("qi") % 4 == 0).then(pl.col(f"{f}_YTD"))
            .when(cons).then(pl.col(f"{f}_YTD") - pl.col(f"{f}_YTD").shift(1).over("cert"))
            .otherwise(None).alias(f"{f}_r") for f in FLOWS])
        aggs = []
        for f in FLOWS:
            r, s = pl.col(f"{f}_r"), pl.col(f"{f}_Q")
            mism = (r.is_null() != s.is_null()) | (r.is_not_null() & s.is_not_null()
                                                   & ((r - s).abs() > 1e-6))
            aggs += [r.count().alias(f"n_{f}"), r.sum().alias(f"s_{f}"),
                     mism.sum().alias(f"mism_{f}")]
        return q.select(aggs)
    if op == "join_exits":
        return (lf.select(COLS[op]).join(ex.select("cert", "exit_type"), on="cert", how="left")
                .group_by("exit_type").agg(n=pl.len(), s=pl.col("ASSET").sum()).sort("exit_type"))
    raise ValueError(op)


def ck_pl(op, r):
    """Kết quả Polars (DataFrame nhỏ) -> dict số để so sánh giữa các engine."""
    if op == "yearly_ratios":
        return {"n_years": r.height, "sum_n": r["n"].sum(), "sum_eq": r["eq"].sum(),
                "sum_npl": r["npl"].sum(), "sum_roaq": r["roaq"].sum()}
    if op == "join_exits":
        out = {}
        for row in r.iter_rows(named=True):
            out[f"n_{row['exit_type']}"] = row["n"]
            out[f"s_{row['exit_type']}"] = row["s"]
        return out
    return r.row(0, named=True)


def pd_query(op, df, ex):
    """Cùng phép `op` bằng pandas; df, ex là DataFrame; trả về dict số giống ck_pl."""
    if op == "filter_project":
        d = df.loc[(df["qi"] >= QI0) & (df["qi"] <= QI1), COLS[op]]
        return {"n": len(d), "s_asset": d["ASSET"].sum(), "s_eq": d["EQ"].sum(),
                "s_nclnls": d["NCLNLS"].sum(), "s_netinc": d["NETINC_Q"].sum()}
    if op == "yearly_ratios":
        d = df[COLS[op]]
        t = d.assign(yr=d["qi"] // 4, eq=d["EQ"] / d["ASSET"], npl=d["NCLNLS"] / d["ASSET"],
                     roaq=d["NETINC_Q"] / d["ASSET"])
        g = t.groupby("yr").agg(n=("eq", "size"), eq=("eq", "mean"), npl=("npl", "mean"),
                                roaq=("roaq", "mean"))
        return {"n_years": len(g), "sum_n": g["n"].sum(), "sum_eq": g["eq"].sum(),
                "sum_npl": g["npl"].sum(), "sum_roaq": g["roaq"].sum()}
    if op == "lag_features":
        d = df[COLS[op]].sort_values(["cert", "qi"])
        g = d.groupby("cert", sort=False)
        ok = (d["qi"] - g["qi"].shift(4)) == 4
        growth = (d["ASSET"] / g["ASSET"].shift(4) - 1).where(ok)
        droa = (d["ROA"] - g["ROA"].shift(4)).where(ok)
        return {"n_valid": int(growth.count()), "s_growth": growth.sum(),
                "m_growth": growth.mean(), "s_droa": droa.sum(), "m_droa": droa.mean()}
    if op == "ytd_to_quarter":
        d = df[COLS[op]].sort_values(["cert", "qi"])
        g = d.groupby("cert", sort=False)
        cons = (d["qi"] - g["qi"].shift(1)) == 1
        q1 = (d["qi"] % 4) == 0
        out = {}
        for f in FLOWS:
            y = d[f"{f}_YTD"]
            r = y.where(q1, (y - g[f"{f}_YTD"].shift(1)).where(cons))
            s = d[f"{f}_Q"]
            mism = (r.isna() != s.isna()) | (r.notna() & s.notna() & ((r - s).abs() > 1e-6))
            out[f"n_{f}"] = int(r.count())
            out[f"s_{f}"] = r.sum()
            out[f"mism_{f}"] = int(mism.sum())
        return out
    if op == "join_exits":
        m = df[COLS[op]].merge(ex[["cert", "exit_type"]], on="cert", how="left")
        g = m.groupby("exit_type", dropna=False).agg(n=("ASSET", "size"), s=("ASSET", "sum"))
        out = {f"n_{k}": int(v) for k, v in g["n"].items()}
        out.update({f"s_{k}": v for k, v in g["s"].items()})
        return out
    raise ValueError(op)


def load_pandas(op, tuned):
    """naive: đọc mọi cột rồi mới tính; tuned: chỉ đọc cột cần (và lọc ngay khi đọc nếu có)."""
    import pandas as pd
    if tuned:
        flt = [("qi", ">=", QI0), ("qi", "<=", QI1)] if op == "filter_project" else None
        df = pd.read_parquet(PANEL_DIR, columns=COLS[op], filters=flt)
    else:
        df = pd.read_parquet(PANEL_DIR)
    return df, pd.read_parquet(EXITS, columns=["cert", "exit_type"])


def run(engine, op):
    """Chạy một phép bằng một engine, trả về dict số (để kiểm tra các engine cho cùng kết quả)."""
    if engine == "polars_lazy":
        import polars as pl
        r = pl_query(op, pl.scan_parquet(PANEL_GLOB), pl.scan_parquet(str(EXITS))).collect()
        return ck_pl(op, r)
    if engine == "polars_read_all":
        import polars as pl
        df, ex = pl.read_parquet(PANEL_GLOB), pl.read_parquet(EXITS)
        return ck_pl(op, pl_query(op, df.lazy(), ex.lazy()).collect())
    df, ex = load_pandas(op, tuned=(engine == "pandas_tuned"))
    return pd_query(op, df, ex)


def same(a, b, rtol=1e-9):
    """Hai dict số có bằng nhau (sai số tương đối rtol) không."""
    if a.keys() != b.keys():
        return False
    for k in a:
        x, y = float(a[k]), float(b[k])
        if abs(x - y) > rtol * max(1.0, abs(x), abs(y)):
            return False
    return True