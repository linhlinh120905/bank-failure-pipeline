# SOURCE: Pipeline chỉ fit trên train (SimpleImputer -> scaler trong sklearn Pipeline, mọi tham số học trên train) theo
#   bài giảng L02 "Machine Learning Pipeline" [9] (Module 1-2, "Split first, prepare later") và L02 "Data Cleaning" [7]
#   (SimpleImputer median); StandardScaler theo bài giảng L02 "Data Transform" [10]; giới hạn phân vị (winsorize)
#   học trên train là lựa chọn của tôi, thay cho cắt ngoại lai trên toàn mẫu.
# AI: Claude viết bộ tiền xử lý (impute -> clip phân vị -> chuẩn hóa) và hàm fit chỉ trên giai đoạn
#   development.
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p4_features as F  # noqa: E402

C = F.C
sys.stdout.reconfigure(encoding="utf-8")
FEATURES = F.FEATURES


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Cắt giá trị về [phân vị dưới, phân vị trên] học từ dữ liệu fit (winsorize); transform dùng ngưỡng đã học."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def make_preprocessor():
    """Toàn bộ tiền xử lý nằm trong MỘT Pipeline; gọi .fit() chỉ trên tập train (hoặc train fold khi CV)."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("clip", QuantileClipper(0.01, 0.99)),
        ("scale", StandardScaler()),
    ])


def load_features(columns=None):
    lf = pl.scan_parquet(str(F.OUT_DIR / "*.parquet"))
    if columns:
        lf = lf.select(columns)
    return lf.collect()


def to_matrix(df, features=None):
    return df.select(features or FEATURES).fill_null(float("nan")).to_numpy()


def fit_preprocessor(df, features=None):
    """Fit CHỈ trên các dòng development (cột `period` == "dev"); dòng gap và test không bao giờ được dùng để fit."""
    feats = features or FEATURES
    pre = make_preprocessor().fit(to_matrix(df.filter(pl.col("period") == "dev"), feats))
    return pre, feats


def params(pre):
    return {"median": pre.named_steps["impute"].statistics_, "clip_lo": pre.named_steps["clip"].lo_,
            "clip_hi": pre.named_steps["clip"].hi_, "mean": pre.named_steps["scale"].mean_,
            "scale": pre.named_steps["scale"].scale_}


def main():
    import joblib
    df = load_features(["period"] + FEATURES)
    pre, feats = fit_preprocessor(df)
    p = params(pre)
    n_dev = int((df["period"] == "dev").sum())
    print(f"Đã fit bộ tiền xử lý trên {n_dev:,} dòng development, {len(feats)} đặc trưng (không dùng gap/test).")
    tbl = pl.DataFrame({"feature": feats, **{k: np.round(v, 4) for k, v in p.items()}})
    pl.Config.set_tbl_formatting("ASCII_FULL")
    pl.Config.set_tbl_rows(60)
    print(tbl)
    (C.ROOT / "reports" / "preprocessor_dev_fit.json").write_text(json.dumps(
        {"n_dev_rows": n_dev, "features": feats, **{k: [float(x) for x in v] for k, v in p.items()}},
        indent=1), encoding="utf-8")
    out = C.ROOT / "data" / "preprocessor_dev.joblib"
    joblib.dump(pre, out)
    print(f"Đã lưu reports/preprocessor_dev_fit.json và {out.name} (không commit)")


if __name__ == "__main__":
    main()
