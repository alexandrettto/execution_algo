def normalize_output_df(out: pd.DataFrame) -> pd.DataFrame:
    float_cols = [
        "best_bid", "best_bid_vol", "best_ask", "best_ask_vol",
        "mid", "spread", "microprice", "imbalance_l1",
        "PRICE_EVENT", "TRADEPRICE",
    ]

    for i in range(1, TOP_N + 1):
        float_cols += [
            f"bid_px_{i}", f"bid_vol_{i}",
            f"ask_px_{i}", f"ask_vol_{i}",
        ]

    int_cols = [
        "NO", "TIME", "ACTION", "ORDERNO",
        "VOLUME_EVENT", "ACTIVE_ORDERS",
    ]

    for col in float_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    for col in int_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    if "TRADENO" in out.columns:
        out["TRADENO"] = pd.to_numeric(out["TRADENO"], errors="coerce").astype("Int64")

    if "BUYSELL" in out.columns:
        out["BUYSELL"] = out["BUYSELL"].astype("string")

    return out



if rows:
    out = pd.DataFrame(rows)
    out = normalize_output_df(out)

    tmp_path = tmp_dir / f"part{part_id:06d}.parquet"
    out.to_parquet(tmp_path, index=False, engine="pyarrow")

    tmp_files.append(tmp_path)

    saved_rows += len(out)
    part_id += 1

    del out
    gc.collect()
