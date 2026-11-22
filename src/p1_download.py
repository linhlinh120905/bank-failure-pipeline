# SOURCE: get_json (retry + exponential back-off), fetch_all (phân trang limit/offset),
#   quarter_ends, manifest resume theo quý và logic kiểm tra đủ dữ liệu được điều chỉnh từ
#   fdic_download_starter.py (BFT65B course starter, NEU 2026) [1].
# AI: Claude (Anthropic) đề xuất 3 chỗ sửa: (1) cast_schema ép kiểu cố định và đếm ô bị
#   coerce, (2) ghi file .tmp rồi replace, nén zstd, (3) ghi download_report.json.
#   đã đọc code, chạy thử 4 quý rồi tải đủ 170 quý, kiểm tra resume bằng cách xóa một quý rồi chạy lại, thêm lại dòng reconfigure UTF-8 của script thầy.
import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402
sys.stdout.reconfigure(encoding="utf-8")  # SOURCE: dòng này lấy từ fdic_download_starter.py [1]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "BFT65B-student-pipeline/1.0"})


def get_json(endpoint, params, max_retries=6, pause=1.0):
    """GET API, thử lại khi lỗi với thời gian chờ tăng dần (2, 4, 8... giây)."""
    url = f"{C.BASE}/{endpoint}"
    for attempt in range(1, max_retries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=180)
            if r.status_code == 200:
                time.sleep(pause)  # lịch sự với server
                return r.json()
            print(f"    HTTP {r.status_code} (lần {attempt}) - thử lại")
        except requests.RequestException as exc:
            print(f"    {type(exc).__name__} (lần {attempt}) - thử lại")
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"Bỏ cuộc: {url} {params}")


def fetch_all(endpoint, fields, filters=None, sort_by="CERT"):
    """Lật trang bằng limit/offset đến khi đủ số dòng API báo. Trả về (DataFrame, total)."""
    rows, offset, total = [], 0, None
    while True:
        params = {"fields": ",".join(fields), "limit": C.PAGE_SIZE, "offset": offset,
                  "sort_by": sort_by, "sort_order": "ASC", "format": "json"}
        if filters:
            params["filters"] = filters
        js = get_json(endpoint, params)
        total = js["meta"]["total"]
        page = [rec["data"] for rec in js["data"]]
        rows.extend(page)
        offset += len(page)
        if not page or offset >= total:
            break
    return pd.DataFrame(rows).drop(columns=["ID"], errors="ignore"), total


def quarter_ends(start, end):
    """Mọi ngày cuối quý (YYYYMMDD) trong [start, end]."""
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    for y in range(s.year, e.year + 1):
        for m, d in ((3, 31), (6, 30), (9, 30), (12, 31)):
            q = date(y, m, d)
            if s <= q <= e:
                out.append(q.strftime("%Y%m%d"))
    return out


def to_text(series):
    """Chuỗi tường minh: giữ None cho ô rỗng, không phụ thuộc kiểu chuỗi mặc định của pandas."""
    return pd.Series([None if pd.isna(v) else str(v) for v in series],
                     index=series.index, dtype="object")


def cast_schema(df):
    """Ép schema cố định cho mọi quý. Trả về (df, số ô bị ép thành rỗng)."""
    df = df.reindex(columns=C.FIN_FIELDS)          # đủ cột, cùng thứ tự
    n_coerced = 0
    for col in C.FIN_FIELDS:
        if col in C.STR_COLS:
            df[col] = to_text(df[col])
        else:
            before = int(df[col].isna().sum())
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            n_coerced += int(df[col].isna().sum()) - before
    return df, n_coerced


def download_financials(quarters):
    C.RAW.mkdir(parents=True, exist_ok=True)
    mpath = C.RAW / "manifest_financials.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    for i, q in enumerate(quarters, 1):
        target = C.RAW / f"REPDTE={q}.parquet"
        if target.exists() and manifest.get(q, {}).get("complete"):
            print(f"[{i}/{len(quarters)}] {q}: đã đủ, bỏ qua")
            continue
        t0 = time.time()
        df, total = fetch_all("financials", C.FIN_FIELDS, filters=f"REPDTE:{q}")
        if df.empty:
            print(f"[{i}/{len(quarters)}] {q}: chưa có dữ liệu")
            continue
        df, n_co = cast_schema(df)
        tmp = target.with_suffix(".tmp")
        df.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(target)                        # không bao giờ để lại file hỏng
        ok = len(df) == total and not df.duplicated(C.KEYS).any()
        manifest[q] = {"rows": len(df), "api_total": total,
                       "complete": bool(ok), "coerced_cells": n_co}
        mpath.write_text(json.dumps(manifest, indent=1))   # lưu sau MỖI quý
        print(f"[{i}/{len(quarters)}] {q}: {len(df):>6,} dòng (API {total:,}) "
              f"{'OK' if ok else 'THIẾU'} {time.time() - t0:5.1f}s")


def download_small_tables():
    """Failures và Institutions: bảng nhỏ, lưu dạng chuỗi, ép kiểu ở Part 2."""
    info = {}
    for name, fields, sort_by in (("failures", C.FAIL_FIELDS, "FAILDATE"),
                                  ("institutions", C.INST_FIELDS, "CERT")):
        df, total = fetch_all(name, fields, sort_by=sort_by)
        df = df.reindex(columns=fields)
        for col in df.columns:
            df[col] = to_text(df[col])
        df.to_parquet(C.RAW / f"{name}.parquet", index=False, compression="zstd")
        info[name] = {"rows": len(df), "api_total": total}
        print(f"{name:<12}: {len(df):>6,} dòng (API {total:,})")
    return info


def check(quarters, small_info):
    """Kiểm tra đủ dữ liệu, in ra và lưu download_report.json (điền trang 1 report)."""
    manifest = json.loads((C.RAW / "manifest_financials.json").read_text())
    files = sorted(C.RAW.glob("REPDTE=*.parquet"))
    panel = pd.concat([pd.read_parquet(f, columns=C.KEYS) for f in files])
    missing = sorted(set(quarters) - set(manifest))
    bad = [q for q, m in manifest.items() if not m["complete"]]
    n_dup = int(panel.duplicated(C.KEYS).sum())
    report = {
        "download_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "method": "FDIC public API via src/p1_download.py",
        "expected_quarters": len(quarters),
        "downloaded_quarters": len(manifest),
        "missing_quarters": missing,
        "incomplete_quarters": bad,
        "financial_rows": int(len(panel)),
        "distinct_cert": int(panel["CERT"].nunique()),
        "duplicate_cert_repdte": n_dup,
        "coerced_cells_total": int(sum(m["coerced_cells"] for m in manifest.values())),
        "small_tables": small_info,
    }
    (C.RAW / "download_report.json").write_text(json.dumps(report, indent=1))
    print("\n=== KIỂM TRA ĐẦY ĐỦ ===")
    for k, v in report.items():
        print(f"{k:<24}: {v}")
    return not (missing or bad or n_dup)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=C.START, help="quý đầu, YYYY-MM-DD")
    ap.add_argument("--end", default=C.END, help="quý cuối, YYYY-MM-DD")
    args = ap.parse_args()
    qs = quarter_ends(args.start, args.end)
    print(f"Tải {len(qs)} quý: {qs[0]} -> {qs[-1]} vào {C.RAW}\n")
    download_financials(qs)
    print()
    small = download_small_tables()
    sys.exit(0 if check(qs, small) else 1)