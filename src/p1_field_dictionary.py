# SOURCE: field dictionary chính thức của FDIC (risview_properties.yaml) [3];
#   danh sách field lấy từ config.py (gốc: fdic_download_starter.py [1]).
# AI: Claude viết code tải và đọc YAML và bảng gán nhóm CAMELS. Nhóm CAMELS là bản nháp: 
# Đã đối chiếu 33 field với file YAML gốc: title khớp, nhóm CAMELS giữ nguyên,
# S đổi tên nhóm cho cơ cấu cho vay.
import json
import sys
from pathlib import Path

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

URL = "https://api.fdic.gov/banks/docs/risview_properties.yaml"
OUT = C.ROOT / "reports" / "field_dictionary.json"

# Bản nháp nhóm CAMELS cho từng field 
GROUP = {
    "CERT": "Key", "REPDTE": "Key", "STNAME": "Descriptor",
    "EQ": "Capital", "RBC1AAJ": "Capital", "RBCRWAJ": "Capital",
    "LNATRES": "Asset quality", "NCLNLS": "Asset quality", "P3ASSET": "Asset quality",
    "P9ASSET": "Asset quality", "NAASSET": "Asset quality", "NTLNLS": "Asset quality",
    "ORE": "Asset quality",
    "LNLSGR": "Concentration (loan mix)", "LNLSNET": "Concentration (loan mix)", "LNRE": "Concentration (loan mix)",
    "LNCI": "Concentration (loan mix)", "LNCON": "Concentration (loan mix)",
    "NETINC": "Earnings", "INTINC": "Earnings", "EINTEXP": "Earnings", "NONII": "Earnings",
    "NONIX": "Earnings", "ELNATR": "Earnings", "ROA": "Earnings", "ROE": "Earnings",
    "NIMY": "Earnings",
    "DEP": "Liquidity / funding", "DEPDOM": "Liquidity / funding", "BRO": "Liquidity / funding",
    "CHBAL": "Liquidity / funding", "SC": "Liquidity / funding",
    "ASSET": "Size",
}


def find_entries(node, wanted, found):
    """Duyệt đệ quy YAML, lấy mục (dict) có khóa trùng tên field."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k in wanted and isinstance(v, dict) and k not in found:
                found[k] = v
            find_entries(v, wanted, found)
    elif isinstance(node, list):
        for v in node:
            find_entries(v, wanted, found)


def main():
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    doc = yaml.safe_load(r.text)
    found = {}
    find_entries(doc, set(C.FIN_FIELDS), found)

    rows = []
    for f in C.FIN_FIELDS:
        e = found.get(f, {})
        unit_info = {k: v for k, v in e.items() if "unit" in str(k).lower()}
        rows.append({"field": f, "group": GROUP.get(f, "UNASSIGNED"),
                     "title": e.get("title"),
                     "description": ("" if str(e.get("description", "")).strip() in ("", "None")
                                     else " ".join(str(e["description"]).split())) or None,
                     "type": e.get("type"), **unit_info})
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

    missing = [r["field"] for r in rows if r["description"] is None]
    print(f"Tìm thấy mô tả cho {len(rows) - len(missing)}/{len(rows)} field")
    print("Thiếu:", missing if missing else "không")
    print("Chưa gán nhóm:", [r["field"] for r in rows if r["group"] == "UNASSIGNED"] or "không")
    print()
    for r in rows:
        print(f"{r['field']:<8} {r['group']:<26} {(r['title'] or '')[:34]:<34} {(r['description'] or '')[:40]}")
    print(f"\nĐã lưu {OUT}")


if __name__ == "__main__":
    main()