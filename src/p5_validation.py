# SOURCE: expanding-window CV có purge (thay cho K-Fold ngẫu nhiên) theo bài giảng L03 [8] (P6); PR-AUC, ngưỡng theo chi phí
#   t* = C_FP/(C_FP + C_FN) theo bài giảng Spark [2] (Problem 9); Pipeline fit chỉ trên train theo bài giảng L02 [9].
# AI: Claude (Anthropic) viết quy trình chọn mô hình CHỈ trên development: so sánh Logistic Regression (baseline) với Gradient
#   Boosting, chọn tham số, ước lượng chi phí từ cột COST, chọn ngưỡng, phân tích độ nhạy. Tôi đã đọc, chạy và đối chiếu

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p4_features as F  # noqa: E402
import p4_pipeline as PP  # noqa: E402
import p5_common as K  # noqa: E402

C = F.C
sys.stdout.reconfigure(encoding="utf-8")
REPORTS = C.ROOT / "reports"
SEL = REPORTS / "p5_dev_selection.json"
SPARK_OUT = C.ROOT / "data" / "p5_spark_dev.parquet"


def load_dev():
    cols = ["cert", "qi", "year", "period", "y", "y_broad"] + F.FEATURES + F.OPTIONAL
    tbl = PP.load_features(cols).filter(pl.col("period") == "dev")           # KHÔNG bao giờ nạp gap hoặc test
    assert int(tbl["qi"].max()) <= F.DEV_END, "bảng chứa quý ngoài development"
    return tbl.sort("cert", "qi")


def cost_model():
    ex = pl.read_parquet(F.EXITS).filter(pl.col("failed") & (pl.col("fail_date") <= pl.date(2006, 12, 31)))
    wc = ex.filter(pl.col("cost").is_not_null())
    return {"n_dev_failures": ex.height, "n_with_cost": wc.height,
            "first_failure_date_with_cost": str(wc["fail_date"].min()),
            "mean_cost_kUSD": float(wc["cost"].mean()), "median_cost_kUSD": float(wc["cost"].median()),
            "assumption": "C_FN = chi phí ước tính trung bình mỗi vụ đổ vỡ (cột COST, nghìn USD); C_FP = c x C_FN với c là "
                          "giả định (không có chi phí báo động giả trong dữ liệu); mức cơ sở c = 0,02 (tỷ lệ 10/500 của bài giảng Spark)",
            "c_grid": list(K.C_GRID), "c_base": K.C_BASE}


def lagged(tbl, feats):
    """Đặc trưng của quý TRƯỚC (trễ 1 quý), null nếu thiếu quý liền trước: kiểm tra độ nhạy với độ trễ nộp báo cáo."""
    ok = (pl.col("qi") - pl.col("qi").shift(1).over("cert")) == 1
    return tbl.select([pl.when(ok).then(pl.col(c).shift(1).over("cert")).alias(c) for c in feats])


def fmt(v, d=3):
    return "  n/a " if v is None else f"{v:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="chạy thử nhanh (2 fold, 1 cấu hình, 1/5 dữ liệu)")
    ap.add_argument("--no-sens", action="store_true", help="bỏ phân tích độ nhạy")
    a = ap.parse_args()
    t0 = time.time()
    pl.Config.set_tbl_formatting("ASCII_FULL")
    tbl = load_dev()
    if a.smoke:
        tbl = tbl.filter(pl.int_range(pl.len()) % 5 == 0)
        blocks, lr_grid, hgb_grid = K.VAL_BLOCKS[3:5], dict(list(K.LR_GRID.items())[:1]), dict(list(K.HGB_GRID.items())[:1])
    else:
        blocks, lr_grid, hgb_grid = K.VAL_BLOCKS, K.LR_GRID, K.HGB_GRID
    feats = F.FEATURES
    X = PP.to_matrix(tbl, feats)
    y = tbl["y"].to_numpy().astype(np.int64)
    yb = tbl["y_broad"].to_numpy().astype(np.int64)
    qi = tbl["qi"].to_numpy()
    print(f"Development: {len(y):,} dòng, {int(y.sum()):,} dòng y=1, {len(feats)} đặc trưng, {len(blocks)} fold "
          f"({'THỬ NHANH' if a.smoke else 'đầy đủ'})")

    cm = cost_model()
    print(f"\nChi phí từ dữ liệu: {cm['n_with_cost']}/{cm['n_dev_failures']} vụ đổ vỡ development có COST "
          f"(từ {cm['first_failure_date_with_cost']}); trung bình {cm['mean_cost_kUSD']:,.0f}, trung vị {cm['median_cost_kUSD']:,.0f} nghìn USD")

    # --- Giai đoạn A: so sánh baseline và mô hình chính, chọn tham số trên development ---
    models = {f"lr|{k}": K.make_lr(v) for k, v in lr_grid.items()}
    models.update({f"hgb|{k}": K.make_hgb(**v) for k, v in hgb_grid.items()})
    print("\n=== A. Expanding-window CV trên development ===")
    res = K.cv_many(models, X, y, qi, blocks, PP.make_preprocessor)
    best = {t: max((k for k in res if k.startswith(t + "|")), key=lambda k: res[k]["pooled"]["pr_auc"]) for t in ("lr", "hgb")}
    print("\nPR-AUC gộp (out-of-fold) và ROC-AUC của từng cấu hình:")
    for k in res:
        flag = "  <== chọn" if k in best.values() else ""
        print(f"  {k:<22} PR-AUC {fmt(res[k]['pooled']['pr_auc'])} | ROC-AUC {fmt(res[k]['pooled']['roc_auc'])}{flag}")
    print("\nTheo fold (mô hình được chọn): PR-AUC  [LR | HGB]  và số dòng dương")
    for i, f in enumerate(res[best["lr"]]["folds"]):
        h = res[best["hgb"]]["folds"][i]
        print(f"  {f['block']:<10} dương {f['pos']:>4} | LR {fmt(f['pr_auc'])} | HGB {fmt(h['pr_auc'])} | ROC LR {fmt(f['roc_auc'])} HGB {fmt(h['roc_auc'])}")

    # --- Giai đoạn B: ngưỡng theo chi phí (chỉ trên dự đoán out-of-fold của development) ---
    sel = {"smoke": a.smoke, "blocks": blocks, "cost_model": cm, "selected": {}, "thresholds": {}, "pooled": {},
           "per_fold": {}, "cost_table": {}, "grid_pooled": {k: res[k]["pooled"] for k in res}}
    print(f"\n=== B. Ngưỡng báo động (chi phí chuẩn hóa: không báo động = 1,0; C_FP/C_FN = c) ===")
    for t in ("lr", "hgb"):
        name = best[t]
        m = ~np.isnan(res[name]["oof"])
        table = {str(c): K.best_threshold(y[m], res[name]["oof"][m], c) for c in K.C_GRID}
        sel["selected"][t] = {"name": name, **(hgb_grid[name.split("|")[1]] if t == "hgb" else {"C": lr_grid[name.split("|")[1]]})}
        sel["thresholds"][t] = table[str(K.C_BASE)]["threshold"]
        sel["pooled"][t], sel["per_fold"][t], sel["cost_table"][t] = res[name]["pooled"], res[name]["folds"], table
        print(f"\n{t.upper()} ({name}); ngưỡng lý thuyết t* = c/(1+c) = {K.C_BASE / (1 + K.C_BASE):.4f} nếu xác suất hiệu chuẩn tốt")
        for c in K.C_GRID:
            r = table[str(c)]
            tag = " (cơ sở)" if c == K.C_BASE else ""
            print(f"  c={c:<5} ngưỡng {r['threshold']:.4f} | chi phí {r['cost']:.3f} | recall {r['recall']:.3f} | "
                  f"precision {fmt(r['precision'])} | báo động {r['alarm_rate']:.2%} dòng | tốt hơn không báo động: {r['better_than_no_alarm']}{tag}")
    m = ~np.isnan(res[best["hgb"]]["oof"])
    rules = {}
    for nm, s in (("texas_ratio", X[:, feats.index("texas_ratio")]), ("neg_eq_assets", -X[:, feats.index("eq_assets")])):
        rules[nm] = K.safe_metrics(y[m], np.where(np.isnan(s), -1e9, s)[m])
    sel["simple_rules_pooled"] = rules
    print("\nQuy tắc đơn giản (không fit) trên cùng các dòng validation: " +
          " | ".join(f"{k}: PR-AUC {fmt(v['pr_auc'])}" for k, v in rules.items()) +
          f" | tỷ lệ dương {y[m].mean():.4f}")
    SEL.write_text(json.dumps(sel, indent=1, ensure_ascii=False, default=lambda o: o.item()), encoding="utf-8")
    print(f"\nĐã lưu {SEL.name} (CHƯA có độ nhạy). Mô hình, tham số và ngưỡng đã được cố định tại đây.")

    # --- Giai đoạn C: độ nhạy (chỉ development) ---
    if not a.no_sens and not a.smoke:
        print("\n=== C. Độ nhạy (mô hình HGB đã chọn, chỉ development) ===")
        cfg = hgb_grid[best["hgb"].split("|")[1]]
        one = {"hgb": K.make_hgb(**cfg)}
        variants = {
            "cơ sở": (X, y, None),
            "thêm đặc trưng tùy chọn (cờ thiếu, rbc_total)": (PP.to_matrix(tbl, F.FEATURES + F.OPTIONAL), y, None),
            "bỏ năm 1984 khỏi train": (X, y, 1985 * 4),
            "nhãn mở rộng y_broad": (X, yb, None),
            "đặc trưng trễ 1 quý": (PP.to_matrix(lagged(tbl, feats), feats), y, None),
        }
        sens = {}
        for nm, (Xv, yv, drop) in variants.items():
            r = K.cv_many(one, Xv, yv, qi, blocks, PP.make_preprocessor, drop_before=drop, log=lambda *_: None)["hgb"]
            sens[nm] = r["pooled"]
            print(f"  {nm:<46} PR-AUC {fmt(r['pooled']['pr_auc'])} | ROC-AUC {fmt(r['pooled']['roc_auc'])} | dương {r['pooled']['pos']}")
        sel["sensitivity_pooled"] = sens
        SEL.write_text(json.dumps(sel, indent=1, ensure_ascii=False, default=lambda o: o.item()), encoding="utf-8")

    if not a.smoke:                       # dữ liệu cho demo Spark MLlib (chỉ development, đến 1991Q4)
        tbl.filter(pl.col("qi") <= 1991 * 4 + 3).select(["cert", "qi", "y"] + feats).write_parquet(SPARK_OUT, compression="zstd")
        print(f"Đã xuất {SPARK_OUT.name} cho demo MLlib (chỉ development, qi <= 1991Q4)")
    print(f"\nXong trong {(time.time() - t0) / 60:.1f} phút. Đã lưu reports/p5_dev_selection.json")


if __name__ == "__main__":
    main()
