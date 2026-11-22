# SOURCE: BASE, PAGE_SIZE và danh sách FIN_FIELDS / FAIL_FIELDS / INST_FIELDS lấy từ
#   fdic_download_starter.py (BFT65B course starter, NEU 2026) [1];
# AI: Claude  đề xuất cách gom hằng số vào config.py. 

from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"            # parquet theo quý 

BASE = "https://api.fdic.gov/banks"
PAGE_SIZE = 10_000
START, END = "1984-03-31", "2026-06-30"   # phạm vi bắt buộc của đề (170 quý)

KEYS = ["CERT", "REPDTE"]
STR_COLS = ["CERT", "REPDTE", "STNAME"]    # cột chuỗi; mọi cột khác là số

FIN_FIELDS = [
    "CERT", "REPDTE", "STNAME",
    "ASSET", "DEP", "DEPDOM", "EQ", "BRO", "CHBAL", "SC",
    "LNLSGR", "LNLSNET", "LNRE", "LNCI", "LNCON", "LNATRES",
    "NCLNLS", "P3ASSET", "P9ASSET", "NAASSET", "NTLNLS", "ORE",
    "NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR",
    "ROA", "ROE", "NIMY",
    "RBC1AAJ", "RBCRWAJ",
]
FAIL_FIELDS = ["CERT", "NAME", "CITYST", "FAILDATE", "FAILYR", "COST",
               "QBFASSET", "QBFDEP", "RESTYPE", "RESTYPE1", "RESDATE"]
INST_FIELDS = ["CERT", "NAME", "STNAME", "BKCLASS", "ACTIVE", "ESTYMD",
               "ENDEFYMD", "CHANGEC1", "INSDATE"]