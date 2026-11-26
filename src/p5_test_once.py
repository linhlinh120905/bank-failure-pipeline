# SOURCE: đánh giá một lần trên giai đoạn test theo đề (test chỉ dùng một lần); metric PR-AUC, recall, precision và chi phí kỳ vọng
#   theo bài giảng Spark [2] (Problem 9); bootstrap theo ngân hàng là thiết kế của tôi.
# AI: Claude (Anthropic) viết script đánh giá test MỘT LẦN: khóa chặn chạy lại, fit trên development, ngưỡng cố định từ development,
#   khoảng tin cậy bootstrap, phát hiện ở mức ngân hàng và thời gian báo trước. Tôi đã đọc, chạy một lần duy nhất và đối chiếu

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p4_features as F  # noqa: E402
import p4_pipeline as PP  # noqa: E402
import p5_common as K  # noqa: E402

C = F.C
sys.stdout.reconfigure(encoding="utf-8")
REPORTS = C.ROOT / "reports"
SEL, RES = REPORTS / "p5_dev_selection.json", REPORTS / "p5_test_result.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="xác nhận: chạy đánh giá test (CHỈ MỘT LẦN)")
    a = ap.parse_args()
    if not a.run:
        print("Script này đánh giá trên TEST và chỉ được chạy MỘT LẦN, sau khi reports/p5_dev_selection.json đã được commit.\n"
              "Chạy: python src/p5_test_once.py --run")
        return
    if RES.exists():
        print(f"DỪNG: {RES.name} đã tồn tại, test đã được dùng. Không chạy lại (mỗi lần xem kết quả test rồi chỉnh sẽ là rò rỉ).")
        sys.exit(2)
    if not SEL.exists():
        print("DỪNG: chưa có reports/p5_dev_selection.json. Chạy p5_validation.py trước.")
        sys.exit(2)
    raw = SEL.read_bytes()
    sel = json.loads(raw)
    if sel.get("smoke"):
        print("DỪNG: p5_dev_selection.json là bản chạy thử (--smoke). Chạy p5_validation.py đầy đủ trước.")
        sys.exit(2)
    t0 = time.time()
    feats = F.FEATURES
    cols = ["cert", "qi", "year", "period", "y"] + feats
    tbl = PP.load_features(cols).filter(pl.col("period").is_in(["dev", "test"])).sort("cert", "qi")   # bỏ gap
    dev, test = tbl.filter(pl.col("period") == "dev"), tbl.filter(pl.col("period") == "test")
    assert int(dev["qi"].max()) <= F.DEV_END and int(test["qi"].min()) > F.GAP_END
    Xd, yd = PP.to_matrix(dev, feats), dev["y"].to_numpy().astype(np.int64)
    Xt, yt = PP.to_matrix(test, feats), test["y"].to_numpy().astype(np.int64)
    cert, year, qi = test["cert"].to_numpy(), test["year"].to_numpy(), test["qi"].to_numpy()
    print(f"Fit trên development: {len(yd):,} dòng ({int(yd.sum()):,} dương). Test: {len(yt):,} dòng ({int(yt.sum()):,} dương).")

    lr_sel, hgb_sel = sel["selected"]["lr"], sel["selected"]["hgb"]
    hgb_cfg = {k: v for k, v in hgb_sel.items() if k != "name"}
    fitted = {
        "lr": Pipeline([("prep", PP.make_preprocessor()), ("model", K.make_lr(lr_sel["C"])())]).fit(Xd, yd),
        "hgb": Pipeline([("prep", PP.make_preprocessor()), ("model", K.make_hgb(**hgb_cfg)())]).fit(Xd, yd),
    }
    out = {"selection_sha256": hashlib.sha256(raw).hexdigest(), "n_dev": len(yd), "n_test": len(yt), "pos_test": int(yt.sum()),
           "test_positive_rate": float(yt.mean()), "models": {}, "rules": {}}
    for tag, pipe in fitted.items():
        p = pipe.predict_proba(Xt)[:, 1]
        thr = sel["thresholds"][tag]
        ev = K.evaluate_at(yt, p, thr)
        ev.update(K.safe_metrics(yt, p))
        ev["recall_at_fpr_5pct"] = K.recall_at_fpr(yt, p, 0.05)
        ev["bootstrap_95ci"] = K.bank_bootstrap(yt, p, cert, thr)
        df = pl.DataFrame({"year": year, "qi": qi, "cert": cert, "y": yt, "p": p})
        ev["by_year"] = (df.group_by("year").agg(rows=pl.len(), positives=pl.col("y").sum(),
                                                  alarms=(pl.col("p") >= thr).sum(),
                                                  tp=((pl.col("p") >= thr) & (pl.col("y") == 1)).sum())
                         .sort("year").to_dicts())
        ex = pl.read_parquet(F.EXITS).select("cert", "fail_date")
        pos = (df.filter(pl.col("y") == 1).join(ex, on="cert")
               .with_columns((pl.col("fail_date").dt.year().cast(pl.Int64) * 4 + (pl.col("fail_date").dt.month().cast(pl.Int64) - 1) // 3).alias("fqi")))
        g = pos.group_by("cert").agg(detected=(pl.col("p") >= thr).any(),
                                     first_alarm=pl.col("qi").filter(pl.col("p") >= thr).min(), fqi=pl.col("fqi").first())
        det = g.filter(pl.col("detected"))
        lead = (det["fqi"] - det["first_alarm"]).to_numpy()
        ev["bank_level"] = {"failed_banks_with_positive_rows": g.height, "detected": det.height,
                            "share_detected": det.height / g.height,
                            "median_lead_quarters": float(np.median(lead)) if len(lead) else None,
                            "share_lead_ge_2_quarters": float((lead >= 2).mean()) if len(lead) else None}
        out["models"][tag] = ev
        if tag == "lr":
            coef = fitted["lr"].named_steps["model"].coef_[0]
            top = np.argsort(-np.abs(coef))[:8]
            out["lr_top_coefficients_standardised"] = {feats[i]: float(coef[i]) for i in top}
    for nm in ("texas_ratio", "eq_assets"):
        s = Xt[:, feats.index(nm)] * (1 if nm == "texas_ratio" else -1)
        out["rules"][nm if nm == "texas_ratio" else "neg_eq_assets"] = K.safe_metrics(yt, np.where(np.isnan(s), -1e9, s))
    out["seconds"] = round(time.time() - t0)
    RES.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=lambda o: o.item()), encoding="utf-8")

    pl.Config.set_tbl_formatting("ASCII_FULL")
    base = out["test_positive_rate"]
    print(f"\n=== KẾT QUẢ TEST (một lần). Tỷ lệ dương test = {base:.4f} (PR-AUC của dự đoán ngẫu nhiên ≈ cùng giá trị) ===")
    for tag, ev in out["models"].items():
        b = ev["bootstrap_95ci"]
        print(f"\n{tag.upper()}: PR-AUC {ev['pr_auc']:.3f} [{b['pr_auc'][0]:.3f}-{b['pr_auc'][1]:.3f}] | ROC-AUC {ev['roc_auc']:.3f} | "
              f"recall tại FPR 5% {ev['recall_at_fpr_5pct']:.3f}")
        print(f"  ngưỡng {ev['threshold']:.4f}: báo động {ev['alarm_rate']:.2%} dòng | TP {ev['tp']} FP {ev['fp']} FN {ev['fn']} | "
              f"recall {ev['recall']:.3f} [{b['recall'][0]:.3f}-{b['recall'][1]:.3f}] | precision {ev['precision'] if ev['precision'] is None else round(ev['precision'], 3)}")
        print("  chi phí chuẩn hóa (không báo động = 1,0) theo c: " + " | ".join(f"c={c}: {v:.3f}" for c, v in ev["cost_vs_no_alarm"].items()))
        bl = ev["bank_level"]
        print(f"  mức ngân hàng: phát hiện {bl['detected']}/{bl['failed_banks_with_positive_rows']} vụ đổ vỡ; báo trước trung vị {bl['median_lead_quarters']} quý")
    print("\nQuy tắc đơn giản: " + " | ".join(f"{k}: PR-AUC {v['pr_auc']:.3f}" for k, v in out["rules"].items()))
    print("LR hệ số lớn nhất (chuẩn hóa):", {k: round(v, 3) for k, v in out["lr_top_coefficients_standardised"].items()})
    print("\nHGB theo năm (dòng, dương, báo động, TP):")
    print(pl.DataFrame(out["models"]["hgb"]["by_year"]))
    print(f"\nĐã lưu {RES.name}. KHÔNG chạy lại script này.")


if __name__ == "__main__":
    main()
