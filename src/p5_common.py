# SOURCE: validation theo thời gian có khoảng đệm (purge) thay cho K-Fold ngẫu nhiên theo bài giảng L03 "Time Series for
#   Finance" [8] (P6: TimeSeriesSplit, purge và embargo); PR-AUC, recall, precision và ngưỡng theo chi phí
#   t* = C_FP / (C_FP + C_FN) theo bài giảng Spark [2] (Problem 9, Hand Calc 9 và 9b); Pipeline fit chỉ trên train theo [9].
# AI: Claude (Anthropic) viết các hàm dùng chung: chia fold expanding-window, so sánh mô hình, bảng chi phí và ngưỡng,
#   bootstrap theo ngân hàng. 
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

SEED = 42
HORIZON = 4
# Các khối validation (theo năm). Train = mọi quý có nhãn kết thúc TRƯỚC khối validation (qi <= v0 - 1 - 4).
VAL_BLOCKS = [(1988, 1988), (1989, 1989), (1990, 1990), (1991, 1991), (1992, 1992),
              (1993, 1996), (1997, 2001), (2002, 2006)]
LR_GRID = {"C=0.01": 0.01, "C=0.1": 0.1, "C=1": 1.0}
HGB_GRID = {
    "d3_i100_lr0.10": dict(max_depth=3, max_iter=100, learning_rate=0.10),
    "d5_i100_lr0.10": dict(max_depth=5, max_iter=100, learning_rate=0.10),
    "d3_i200_lr0.05": dict(max_depth=3, max_iter=200, learning_rate=0.05),
    "d5_i200_lr0.05": dict(max_depth=5, max_iter=200, learning_rate=0.05),
}
C_GRID = (0.005, 0.01, 0.02, 0.05, 0.10)    # C_FP / C_FN: giả định (chi phí báo động giả không có trong dữ liệu)
C_BASE = 0.02                               # mức cơ sở: cùng tỷ lệ 10/500 trong ví dụ của bài giảng Spark [2]


def make_lr(C):
    return lambda: LogisticRegression(C=C, max_iter=1000)       # không class_weight: giữ xác suất gần hiệu chuẩn


def make_hgb(**cfg):
    return lambda: HistGradientBoostingClassifier(random_state=SEED, early_stopping=False, l2_regularization=1.0,
                                                  min_samples_leaf=200, **cfg)


def fold_masks(qi, a, b, horizon=HORIZON):
    v0, v1 = a * 4, b * 4 + 3
    return qi <= v0 - 1 - horizon, (qi >= v0) & (qi <= v1)


def safe_metrics(y, p):
    both = 0 < int(y.sum()) < len(y)
    return {"n": int(len(y)), "pos": int(y.sum()),
            "pr_auc": float(average_precision_score(y, p)) if both else None,
            "roc_auc": float(roc_auc_score(y, p)) if both else None}


def cv_many(models, X, y, qi, blocks, make_prep, drop_before=None, log=print):
    """Expanding-window CV. Với mỗi fold: bộ tiền xử lý (Pipeline) fit CHỈ trên train của fold, dùng chung cho mọi mô hình
    ứng viên (kết quả giống fit một Pipeline đầy đủ cho từng mô hình); không dòng validation nào được dùng để fit."""
    oof = {k: np.full(len(y), np.nan) for k in models}
    folds = {k: [] for k in models}
    for a, b in blocks:
        t0 = time.time()
        tr, va = fold_masks(qi, a, b)
        if drop_before is not None:
            tr = tr & (qi >= drop_before)
        prep = make_prep().fit(X[tr])
        Xt, Xv = prep.transform(X[tr]), prep.transform(X[va])
        for k, mk in models.items():
            p = mk().fit(Xt, y[tr]).predict_proba(Xv)[:, 1]
            oof[k][va] = p
            folds[k].append({"block": f"{a}-{b}", "n_train": int(tr.sum()), "pos_train": int(y[tr].sum()),
                             **{kk: v for kk, v in safe_metrics(y[va], p).items()}})
        log(f"  fold {a}-{b}: train {int(tr.sum()):,} dòng ({int(y[tr].sum()):,} dương), validation {int(va.sum()):,} "
            f"({int(y[va].sum())} dương), {len(models)} mô hình, {time.time() - t0:.0f}s")
    out = {}
    for k in models:
        m = ~np.isnan(oof[k])
        out[k] = {"oof": oof[k], "folds": folds[k], "pooled": safe_metrics(y[m], oof[k][m])}
    return out


def best_threshold(y, p, c):
    """Ngưỡng làm chi phí kỳ vọng nhỏ nhất. Chi phí chuẩn hóa = (FN + c*FP) / số dương; 'không báo động' = 1,0.
    Mỗi dòng (ngân hàng-quý) được coi là một quyết định độc lập (giản lược)."""
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order].astype(float), p[order]
    P = ys.sum()
    tp, fp = np.cumsum(ys), np.cumsum(1.0 - ys)
    cost = ((P - tp) + c * fp) / P
    last = np.append(ps[1:] != ps[:-1], True)
    cm = np.where(last, cost, np.inf)
    k = int(np.argmin(cm))
    if cm[k] >= 1.0:
        return {"threshold": 1.01, "cost": 1.0, "alarms": 0, "tp": 0, "fp": 0, "recall": 0.0, "precision": None,
                "alarm_rate": 0.0, "better_than_no_alarm": False}
    return {"threshold": float(ps[k]), "cost": float(cm[k]), "alarms": k + 1, "tp": int(tp[k]), "fp": int(fp[k]),
            "recall": float(tp[k] / P), "precision": float(tp[k] / (k + 1)), "alarm_rate": float((k + 1) / len(ys)),
            "better_than_no_alarm": True}


def recall_at_fpr(y, p, fpr=0.05):
    thr = np.quantile(p[y == 0], 1 - fpr)
    return float(((p > thr) & (y == 1)).sum() / y.sum())


def evaluate_at(y, p, thr, c_grid=C_GRID):
    alarm = p >= thr
    P, N = int(y.sum()), int(len(y) - y.sum())
    tp, fp = int((alarm & (y == 1)).sum()), int((alarm & (y == 0)).sum())
    return {"threshold": float(thr), "alarms": int(alarm.sum()), "tp": tp, "fp": fp, "fn": P - tp,
            "recall": tp / P, "precision": tp / int(alarm.sum()) if alarm.sum() else None,
            "alarm_rate": float(alarm.mean()), "false_alarm_rate": fp / N,
            "cost_vs_no_alarm": {str(c): (P - tp + c * fp) / P for c in c_grid},
            "cost_alarm_all": {str(c): c * N / P for c in c_grid}}


def bank_bootstrap(y, p, cert, thr, c=C_BASE, B=200, seed=SEED):
    """Khoảng tin cậy 95% bằng bootstrap theo NGÂN HÀNG (các quý của một ngân hàng không độc lập)."""
    rng = np.random.default_rng(seed)
    order = np.argsort(cert, kind="stable")
    _, start = np.unique(cert[order], return_index=True)
    groups = np.split(order, start[1:])
    n = len(groups)
    res = {"pr_auc": [], "recall": [], "cost": []}
    for _ in range(B):
        ii = np.concatenate([groups[k] for k in rng.integers(0, n, n)])
        yy, pp = y[ii], p[ii]
        if yy.sum() == 0:
            continue
        al = pp >= thr
        tp, fp = (al & (yy == 1)).sum(), (al & (yy == 0)).sum()
        res["pr_auc"].append(average_precision_score(yy, pp))
        res["recall"].append(tp / yy.sum())
        res["cost"].append((yy.sum() - tp + c * fp) / yy.sum())
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in res.items()}
