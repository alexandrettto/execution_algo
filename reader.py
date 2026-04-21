# -*- coding: utf-8 -*-

"""
Chunked reconstruction of CNYRUB_TOM order book (top-5 levels)
from MOEX OrderLog files inside a folder.

Just edit CONFIG section below and run:

    python build_cny_book_top5.py
"""
from pathlib import Path
from bisect import bisect_left, insort
from collections import defaultdict
import gc
import re

import zipfile
import gzip
import io

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:
    raise ImportError(
        "pyarrow is required for writing one parquet file incrementally. "
        "Install it with: pip install pyarrow"
    ) from e

# ============================================================
# CONFIG — ИЗМЕНИ ЗДЕСЬ ПУТИ
# ============================================================

INPUT_DIR = Path(r"C:\Users\ИллюкАА\Documents\algotrading")        # папка с OrderLog файлами
OUTPUT_DIR = Path(r"C:\Users\ИллюкАА\Documents\algotrading\CNY_BOOK_TOP5")  # куда сохранять результат


FILE_PATTERN = "OrderLog"

CSV_SEPARATOR = ","
CSV_ENCODING = None

CHUNKSIZE = 300_000
TOP_N = 5

SAVE_ONLY_IF_L1_CHANGED = False

# ============================================================
# ORDER BOOK
# ============================================================

class OrderBookReconstructor:
    def __init__(self):
        self.active_orders = {}
        self.bids = defaultdict(int)
        self.asks = defaultdict(int)

        self.bid_prices = []   # ascending
        self.ask_prices = []   # ascending

    def _add_price_level(self, side, price):
        arr = self.bid_prices if side == "B" else self.ask_prices
        i = bisect_left(arr, price)
        if i == len(arr) or arr[i] != price:
            insort(arr, price)

    def _remove_price_level_if_empty(self, side, price):
        book = self.bids if side == "B" else self.asks
        if book.get(price, 0) > 0:
            return

        arr = self.bid_prices if side == "B" else self.ask_prices
        i = bisect_left(arr, price)
        if i < len(arr) and arr[i] == price:
            arr.pop(i)

        book.pop(price, None)

    def _add_to_book(self, side, price, volume):
        if volume <= 0:
            return

        book = self.bids if side == "B" else self.asks
        if book[price] == 0:
            self._add_price_level(side, price)
        book[price] += volume

    def _remove_from_book(self, side, price, volume):
        if volume <= 0:
            return

        book = self.bids if side == "B" else self.asks
        if price not in book:
            return

        book[price] -= volume
        if book[price] <= 0:
            self._remove_price_level_if_empty(side, price)

    def on_add(self, orderno, side, price, volume):
        if orderno in self.active_orders:
            old = self.active_orders[orderno]
            self._remove_from_book(old["side"], old["price"], old["volume"])

        self.active_orders[orderno] = {
            "side": side,
            "price": price,
            "volume": volume,
        }
        self._add_to_book(side, price, volume)

    def on_trade(self, orderno, volume):
        if orderno not in self.active_orders:
            return

        old = self.active_orders[orderno]
        dec = min(volume, old["volume"])

        self._remove_from_book(old["side"], old["price"], dec)

        remaining = old["volume"] - dec
        if remaining <= 0:
            self.active_orders.pop(orderno, None)
        else:
            old["volume"] = remaining

    def on_cancel(self, orderno, volume):
        if orderno not in self.active_orders:
            return

        old = self.active_orders[orderno]
        dec = min(volume, old["volume"])

        self._remove_from_book(old["side"], old["price"], dec)
        self.active_orders.pop(orderno, None)

    def best_bid(self):
        if not self.bid_prices:
            return None, None
        px = self.bid_prices[-1]
        return px, self.bids[px]

    def best_ask(self):
        if not self.ask_prices:
            return None, None
        px = self.ask_prices[0]
        return px, self.asks[px]

    def top_n(self, n):
        bids = [(px, self.bids[px]) for px in reversed(self.bid_prices[-n:])]
        asks = [(px, self.asks[px]) for px in self.ask_prices[:n]]
        return bids, asks

    def snapshot(self, n):
        bid_px, bid_vol = self.best_bid()
        ask_px, ask_vol = self.best_ask()

        if bid_px is not None and ask_px is not None:
            mid = (bid_px + ask_px) / 2
            spread = ask_px - bid_px
        else:
            mid = np.nan
            spread = np.nan

        if (
            bid_px is not None
            and ask_px is not None
            and bid_vol is not None
            and ask_vol is not None
            and (bid_vol + ask_vol) > 0
        ):
            microprice = (
                ask_px * bid_vol + bid_px * ask_vol
            ) / (bid_vol + ask_vol)

            imbalance = (
                bid_vol - ask_vol
            ) / (bid_vol + ask_vol)
        else:
            microprice = np.nan
            imbalance = np.nan

        snap = {
            "best_bid": bid_px if bid_px is not None else np.nan,
            "best_bid_vol": bid_vol if bid_vol is not None else np.nan,
            "best_ask": ask_px if ask_px is not None else np.nan,
            "best_ask_vol": ask_vol if ask_vol is not None else np.nan,
            "mid": mid,
            "spread": spread,
            "microprice": microprice,
            "imbalance_l1": imbalance,
        }

        bids, asks = self.top_n(n)

        for i in range(n):
            if i < len(bids):
                snap[f"bid_px_{i+1}"] = bids[i][0]
                snap[f"bid_vol_{i+1}"] = bids[i][1]
            else:
                snap[f"bid_px_{i+1}"] = np.nan
                snap[f"bid_vol_{i+1}"] = np.nan

            if i < len(asks):
                snap[f"ask_px_{i+1}"] = asks[i][0]
                snap[f"ask_vol_{i+1}"] = asks[i][1]
            else:
                snap[f"ask_px_{i+1}"] = np.nan
                snap[f"ask_vol_{i+1}"] = np.nan

        return snap


# ============================================================
# HELPERS
# ============================================================

def safe_stem(path: Path) -> str:
    """
    Normalize filename stem for output parquet name.
    """
    stem = path.stem
    stem = re.sub(r"[^\w\-.]+", "_", stem)
    return stem


def normalize_chunk(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "NO", "SECCODE", "BUYSELL", "TIME", "ORDERNO",
        "ACTION", "PRICE", "VOLUME"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.loc[df["SECCODE"] == "CNYRUB_TOM"].copy()

    if out.empty:
        return out

    if "TRADENO" not in out.columns:
        out["TRADENO"] = np.nan
    if "TRADEPRICE" not in out.columns:
        out["TRADEPRICE"] = np.nan

    numeric_cols = [
        "NO", "TIME", "ORDERNO", "ACTION",
        "PRICE", "VOLUME", "TRADENO", "TRADEPRICE"
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["BUYSELL"] = out["BUYSELL"].astype(str)
    out = out[out["BUYSELL"].isin(["B", "S"])]

    out = out.dropna(subset=["NO", "ORDERNO", "ACTION", "VOLUME"])
    if out.empty:
        return out

    out["NO"] = out["NO"].astype("int64")
    out["ORDERNO"] = out["ORDERNO"].astype("int64")
    out["ACTION"] = out["ACTION"].astype("int64")
    out["VOLUME"] = out["VOLUME"].astype("int64")

    out = out.sort_values("NO").reset_index(drop=True)

    return out

def build_output_name(zip_path: Path, inner_member: str) -> str:
    """
    Пример:
    202601.zip + 20260105/ordlog.exp.gz
    ->
    202601_20260105_CNYRUB_TOM_top5.parquet
    """
    zip_stem = safe_stem(zip_path)
    day_part = Path(inner_member).parts[0] if len(Path(inner_member).parts) > 1 else Path(inner_member).stem
    day_part = safe_stem(Path(day_part))
    return f"{zip_stem}_{day_part}_CNYRUB_TOM_top5.parquet"

def read_gz_csv_chunks_from_zip(zip_path: Path, inner_member: str, chunksize: int):
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(inner_member) as gz_file:
            with gzip.open(gz_file, "rt", encoding=CSV_ENCODING or "utf-8", newline="") as f:
                reader = pd.read_csv(
                    f,
                    chunksize=chunksize,
                    sep=CSV_SEPARATOR,
                    low_memory=False,
                )
                for chunk in reader:
                    yield chunk

def count_lines_in_gz_inside_zip(zip_path: Path, inner_member: str) -> int:
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(inner_member) as gz_file:
            with gzip.open(gz_file, "rt", encoding=CSV_ENCODING or "utf-8", newline="") as f:
                for _ in f:
                    n += 1
    return max(n - 1, 0)  # minus header

def find_month_zip_files():
    files = sorted(
        f for f in INPUT_DIR.rglob("*")
        if f.is_file()
        and f.suffix.lower() == ".zip"
    )
    if not files:
        raise FileNotFoundError(f"No .zip files found in {INPUT_DIR}")
    return files


def iter_ordlog_members(zip_path: Path):
    """
    Ищет внутри monthly zip файлы вида .../ordlog.exp.gz
    """
    with zipfile.ZipFile(zip_path) as zf:
        members = sorted(
            name for name in zf.namelist()
            if name.lower().endswith("ordlog.exp.gz")
        )
    return members


def build_rows_from_chunk(chunk: pd.DataFrame, ob: OrderBookReconstructor, prev_l1):
    rows = []

    for row in chunk.itertuples(index=False):
        action = int(row.ACTION)

        if action == 1:
            if pd.notna(row.PRICE) and row.PRICE > 0 and row.VOLUME > 0:
                ob.on_add(
                    int(row.ORDERNO),
                    row.BUYSELL,
                    float(row.PRICE),
                    int(row.VOLUME)
                )

        elif action == 2:
            if row.VOLUME > 0:
                ob.on_trade(
                    int(row.ORDERNO),
                    int(row.VOLUME)
                )

        elif action == 0:
            if row.VOLUME >= 0:
                ob.on_cancel(
                    int(row.ORDERNO),
                    int(row.VOLUME)
                )

        snap = ob.snapshot(TOP_N)

        l1 = (
            snap["best_bid"],
            snap["best_bid_vol"],
            snap["best_ask"],
            snap["best_ask_vol"],
        )

        if SAVE_ONLY_IF_L1_CHANGED and l1 == prev_l1:
            continue

        prev_l1 = l1

        snap.update({
            "NO": row.NO,
            "TIME": row.TIME,
            "ACTION": row.ACTION,
            "ORDERNO": row.ORDERNO,
            "BUYSELL": row.BUYSELL,
            "PRICE_EVENT": row.PRICE,
            "VOLUME_EVENT": row.VOLUME,
            "TRADENO": getattr(row, "TRADENO", np.nan),
            "TRADEPRICE": getattr(row, "TRADEPRICE", np.nan),
            "ACTIVE_ORDERS": len(ob.active_orders),
        })

        rows.append(snap)

    return rows, prev_l1


# ============================================================
# MAIN
# ============================================================

def process_one_ordlog_member(zip_path: Path, inner_member: str):
    print(f"\nProcessing: {zip_path.name} :: {inner_member}")

    output_name = build_output_name(zip_path, inner_member)
    output_path = OUTPUT_DIR / output_name

    ob = OrderBookReconstructor()
    prev_l1 = None
    writer = None

    total_rows = count_lines_in_gz_inside_zip(zip_path, inner_member)
    pbar = tqdm(total=total_rows, desc=f"{zip_path.name}:{inner_member}", unit="rows")

    saved_rows = 0
    cny_rows = 0

    try:
        for raw_chunk in read_gz_csv_chunks_from_zip(
            zip_path=zip_path,
            inner_member=inner_member,
            chunksize=CHUNKSIZE,
        ):
            pbar.update(len(raw_chunk))

            chunk = normalize_chunk(raw_chunk)
            del raw_chunk

            if chunk.empty:
                gc.collect()
                continue

            cny_rows += len(chunk)

            rows, prev_l1 = build_rows_from_chunk(chunk, ob, prev_l1)
            del chunk

            if rows:
                out = pd.DataFrame(rows)

                table = pa.Table.from_pandas(out, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)

                writer.write_table(table)
                saved_rows += len(out)

                del out
                del table

            gc.collect()

    finally:
        pbar.close()
        if writer is not None:
            writer.close()

    print(f"Saved rows: {saved_rows:,}")
    print(f"CNY rows processed: {cny_rows:,}")
    print(f"Output: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zip_files = find_month_zip_files()

    print("Found monthly zip files:")
    for z in zip_files:
        print(" ", z)

    for zip_path in zip_files:
        members = iter_ordlog_members(zip_path)

        if not members:
            print(f"[WARN] No ordlog.exp.gz inside {zip_path.name}")
            continue

        for inner_member in members:
            process_one_ordlog_member(zip_path, inner_member)

    print("\nDone.")
    
if __name__ == "__main__":
    main()