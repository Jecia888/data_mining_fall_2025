#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generalized merge & align features with YAML control.

What this script does
- Supports minute/hour/day cadences (derived from YAML `date_range.granularity`)
- For intraday (minute/hour), aligns to Regular Trading Hours (RTH, 09:30–16:00 ET);
  daily mode skips RTH windowing
- Two-sided limited average fill for gaps (limits measured in “bars”)
- Computes robust intraday r_close + QC
- IO paths can use a {cad} template to avoid collisions between cadences

Key fixes & enhancements
  * After reading, timestamps are aligned to the “minute” boundary
    (both minute and hour are floored to minute; avoids 09:30:00.500 or hour bars
    being snapped oddly)
  * RTH right boundary is half-open: 09:30 ≤ t < 16:00, matching a 390-minute grid
  * Hourly cadence uses an “adaptive anchor” (auto-detect :00 or :30) to build the grid
    to avoid reindexing to all-NaN
  * Keeps `rth_end_handling` to optionally include the 15:30 half-hour when using the :30 anchor

Example YAML keys:
merge:
  enabled: true
  input_timestamp_tz: "UTC"
  session: "rth"
  timezone: "America/New_York"
  rth_end_handling: "drop_last_half_hour"   # Handling of 15:30–16:00 for hourly: drop|keep_half_bar|pad_to_full_hour
  fill_limit_bars: 3
  rclose:
    fill_limit_bars: 1
    min_valid_ratio: 0.70
    extra_limit_bars: 2
  inputs:
    custom_indicators_dir: "D:\\...\\output\\custom_indicators_{cad}"
    indicators_dir:        "D:\\...\\output\\indicators_{cad}"
  outputs:
    merged_dir: "D:\\...\\feature_extraction\\output\\intermediate_dataset_{cad}"
    write_qc: true
"""

import os
import glob
import sys
import argparse
from typing import List, Optional
import numpy as np
import pandas as pd
import yaml

try:
    from zoneinfo import ZoneInfo
    NY_TZ = ZoneInfo("America/New_York")
except Exception:
    import pytz
    NY_TZ = pytz.timezone("America/New_York")


# ======================== cadence utilities ======================== #

def normalize_granularity(ts: str) -> str:
    m = {
        "minute": "minute", "min": "minute", "m": "minute",
        "hour": "hour", "hr": "hour", "h": "hour",
        "day": "day", "daily": "day", "d": "day"
    }
    tsn = m.get((ts or "").strip().lower())
    if not tsn:
        raise SystemExit("granularity must be one of: minute|hour|day")
    return tsn

def derive_cadence_label(cfg: dict) -> str:
    g = cfg.get("date_range", {}).get("granularity", "minute")
    return normalize_granularity(g)  # "minute"|"hour"|"day"

def expected_bars_per_rth_day(cad: str) -> int:
    if cad == "minute":
        return 390
    if cad == "hour":
        # With a half-hour anchor (default): 09:30,10:30,11:30,12:30,13:30,14:30 (optionally 15:30)
        # With a top-of-hour anchor: 10:00..15:00
        return 6
    if cad == "day":
        return 1
    raise ValueError(f"Unsupported cadence: {cad}")

def cadence_to_pd_freq(cad: str) -> str:
    # Only for day alignment; minute/hour are manually floored to “min”
    return {"minute": "min", "hour": "H", "day": "D"}[cad]


# ======================== IO / config helpers ======================== #

def load_config(path: str) -> dict:
    if not os.path.exists(path):
        print(f"Config not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}

def fill_path_template(path_or_tpl: Optional[str], cad: str) -> Optional[str]:
    if not path_or_tpl:
        return None
    return path_or_tpl.replace("{cad}", cad)

def resolve_dirs(cfg: dict, cad: str):
    merge_cfg = cfg.get("merge", {}) or {}
    inputs = merge_cfg.get("inputs", {}) or {}
    outputs = merge_cfg.get("outputs", {}) or {}

    custom_dir = fill_path_template(inputs.get("custom_indicators_dir"), cad)
    ind_dir    = fill_path_template(inputs.get("indicators_dir"), cad)

    if not custom_dir or not ind_dir:
        root = cfg.get("output_dir") or "output"
        custom_dir = custom_dir or os.path.join(root, f"custom_indicators_{cad}")
        ind_dir    = ind_dir    or os.path.join(root, f"indicators_{cad}")

    out_dir = fill_path_template(outputs.get("merged_dir"), cad)
    if not out_dir:
        out_dir = os.path.join(
            os.path.dirname(custom_dir),
            f"..{os.sep}..{os.sep}feature_extraction{os.sep}output{os.sep}intermediate_dataset_{cad}"
        )
        out_dir = os.path.abspath(out_dir)

    write_qc = bool(outputs.get("write_qc", True))
    return custom_dir, ind_dir, out_dir, write_qc

def get_merge_params(cfg: dict, cad: str):
    merge_cfg = cfg.get("merge", {}) or {}
    session = merge_cfg.get("session") or ("day" if cad == "day" else "rth")
    tz_name = merge_cfg.get("timezone", "America/New_York")
    input_tz_name = merge_cfg.get("input_timestamp_tz", "UTC")
    rth_end_handling = merge_cfg.get("rth_end_handling", "drop_last_half_hour")
    fill_limit = int(merge_cfg.get("fill_limit_bars", 3))
    rcfg = merge_cfg.get("rclose", {}) or {}
    rclose_fill_limit = int(rcfg.get("fill_limit_bars", 1))
    rclose_min_valid  = float(rcfg.get("min_valid_ratio", 0.70))
    rclose_extra_limit= int(rcfg.get("extra_limit_bars", 2))
    return {
        "session": session,
        "tz_name": tz_name,
        "input_tz_name": input_tz_name,
        "rth_end_handling": rth_end_handling,
        "fill_limit": fill_limit,
        "rclose_fill_limit": rclose_fill_limit,
        "rclose_min_valid": rclose_min_valid,
        "rclose_extra_limit": rclose_extra_limit,
    }


# ======================== reading & time handling ======================== #

def read_csv_to_et_index(path: str, input_tz_name: str, cad: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"'timestamp' column not found in {path}")

    # Normalize to UTC first (avoids mixed-timezone object dtype issues)
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    # Convert to ET, then floor to a minute or day boundary for stable indexing
    ts = ts.dt.tz_convert(NY_TZ)
    ts = ts.dt.floor("min" if cad in ("minute", "hour") else "D")

    df["timestamp"] = ts
    df = (
        df.dropna(subset=["timestamp"])
          .sort_values("timestamp")
          .drop_duplicates("timestamp", keep="last")
          .set_index("timestamp")
    )
    return df

def rth_mask(idx: pd.DatetimeIndex) -> pd.Series:
    """
    Boolean mask for 09:30 ≤ t < 16:00 (half-open right boundary).
    """
    t = idx.tz_convert(NY_TZ)
    start = pd.Timestamp("09:30").time()
    end   = pd.Timestamp("16:00").time()
    return pd.Series([(start <= hh < end) for hh in t.time], index=idx)

def generate_rth_grid_for_days(days: List[pd.Timestamp], cad: str, end_handling: str) -> pd.DatetimeIndex:
    """Generate the intraday baseline grid: minute = 390 points; hour = half-hour anchor (backward compatible)."""
    out = []
    for d in days:
        d = pd.Timestamp(d)
        d = d.tz_localize(NY_TZ) if d.tz is None else d.tz_convert(NY_TZ)
        start = d.replace(hour=9, minute=30, second=0, microsecond=0)
        if cad == "minute":
            rng = pd.date_range(start=start, periods=390, freq="min", tz=NY_TZ)  # 09:30..15:59
            out.append(rng)
        elif cad == "hour":
            # Default half-hour anchor: 09:30..14:30; optionally add 15:30
            hours = [start + pd.Timedelta(hours=i) for i in range(6)]
            if end_handling in ("keep_half_bar", "pad_to_full_hour"):
                hours.append(start + pd.Timedelta(hours=6))  # 15:30
            out.append(pd.DatetimeIndex(hours))
        else:
            raise ValueError("RTH grid only applies to minute/hour cadences.")
    if not out:
        return pd.DatetimeIndex([], tz=NY_TZ)
    return out[0].append(out[1:]) if len(out) > 1 else out[0]

def compute_bar_index_since_open(index: pd.DatetimeIndex) -> pd.Series:
    """Per-ET-day counter 0..N-1 for bars since open."""
    dates = index.tz_convert(NY_TZ).date
    counter, prev, c = [], None, -1
    for d in dates:
        if d != prev:
            c = 0
            prev = d
        else:
            c += 1
        counter.append(c)
    return pd.Series(counter, index=index, dtype="int32")

def add_volume_z_by_bar_index(df: pd.DataFrame) -> pd.Series:
    """Compute a z-score of volume by intraday bar position (minute or hour)."""
    mod = compute_bar_index_since_open(df.index)
    df = df.copy()
    df["_mod"] = mod.values
    grp = df.groupby("_mod")["volume"]
    mean = grp.transform("mean")
    std = grp.transform("std").replace(0, np.nan)
    z = (df["volume"] - mean) / std
    return z.fillna(0.0)


# ======================== fill & r_close (intraday) ======================== #

PRIMARY_PRICE_COLS = ["open", "high", "low", "close", "vwap", "volume", "transactions"]

def two_sided_avg_fill_limited(s: pd.Series, limit: int = 3) -> pd.Series:
    """
    Two-sided limited fill:
    - If both forward-fill and back-fill exist, take their average
    - If only one side exists, use that side
    - Limit controls how many consecutive NaNs can be filled
    """
    if s.dtype.kind not in "biufc":
        return s
    fwd = s.ffill(limit=limit)
    bwd = s.bfill(limit=limit)
    out = s.copy()
    m = s.isna()
    both = m & fwd.notna() & bwd.notna()
    out[both] = (fwd[both] + bwd[both]) / 2.0
    left_only = m & fwd.notna() & bwd.isna()
    right_only = m & bwd.notna() & fwd.isna()
    out[left_only] = fwd[left_only]
    out[right_only] = bwd[right_only]
    return out

def _fill_limited_per_day(s: pd.Series, limit: int) -> pd.Series:
    """Per-day forward/back fill with the same limit on both sides."""
    return s.ffill(limit=limit).bfill(limit=limit)

def _compute_rclose_from_close(close_series: pd.Series) -> pd.Series:
    """Compute log returns of close; set the first bar’s return to 0."""
    rc = np.log(close_series) - np.log(close_series.shift(1))
    if len(rc) > 0:
        rc.iloc[0] = 0.0
    return rc

def _rclose_repair_and_qc(df: pd.DataFrame,
                          rclose_fill_limit: int = 1,
                          rclose_min_valid: float = 0.70,
                          rclose_extra_limit: int = 2):
    """
    Compute r_close per day with a conservative fill strategy:
    - Start with a small fill_limit
    - If valid ratio is below threshold, retry with a larger limit and pick the better outcome
    Returns:
      r_close series aligned to df.index, and a QC dataframe per day.
    """
    close_numeric = pd.to_numeric(df["close"], errors="coerce")
    r_close_out = pd.Series(index=df.index, dtype=float)
    day_groups = df.groupby(pd.to_datetime(df.index.tz_convert(NY_TZ).date))
    qc_rows = []
    for day, gidx in day_groups.groups.items():
        idx = df.index.isin(df.loc[gidx].index)
        g_close = close_numeric[idx].copy()

        g_filled = _fill_limited_per_day(g_close, limit=rclose_fill_limit)
        g_filled = g_filled.mask(g_filled <= 0, np.nan)
        rc1 = _compute_rclose_from_close(g_filled)
        valid1 = np.isfinite(rc1).mean() if len(rc1) else 0.0

        if valid1 < rclose_min_valid and rclose_extra_limit > rclose_fill_limit:
            g_filled2 = _fill_limited_per_day(g_close, limit=rclose_extra_limit)
            g_filled2 = g_filled2.mask(g_filled2 <= 0, np.nan)
            rc2 = _compute_rclose_from_close(g_filled2)
            valid2 = np.isfinite(rc2).mean() if len(rc2) else 0.0
            if valid2 > valid1:
                r_close_out[idx] = rc2
                used_limit = rclose_extra_limit
                final_valid = valid2
            else:
                r_close_out[idx] = rc1
                used_limit = rclose_fill_limit
                final_valid = valid1
        else:
            r_close_out[idx] = rc1
            used_limit = rclose_fill_limit
            final_valid = valid1

        qc_rows.append({
            "day_id": pd.Timestamp(day).date(),
            "rclose_valid_ratio": float(final_valid),
            "rclose_limit_used": int(used_limit),
            "N": int(idx.sum())
        })
    qc_df = pd.DataFrame(qc_rows).sort_values("day_id")
    return r_close_out, qc_df


# ======================== column configs ======================== #

CUSTOM_BASE_COLS = [
    "timestamp", "open", "high", "low", "close", "volume", "vwap", "transactions"
]
CUSTOM_INDICATOR_COLS = [
    "macd", "macd_signal", "macd_hist",
    "bb_mid_20", "bb_up_20", "bb_dn_20", "bb_bw_20",
    "stoch_k_14", "stoch_d_14", "obv"
]
INDICATORS_ONLY_COLS = [
    "sma_20", "sma_50", "ema_12", "ema_26", "rsi_14", "volatility_20", "atr_14"
]


# ======================== core per-ticker merge ======================== #

def _gen_hour_grid_adaptive(days: List[pd.Timestamp], anchor_minute: int, end_handling: str) -> pd.DatetimeIndex:
    """
    Adaptive hourly grid (RTH):
      - anchor_minute == 30 → 09:30..14:30 (optionally append 15:30)
      - anchor_minute == 0  → 10:00..15:00
    """
    out = []
    for d in days:
        d = pd.Timestamp(d)
        d = d.tz_localize(NY_TZ) if d.tz is None else d.tz_convert(NY_TZ)
        if anchor_minute == 0:
            # Top-of-hour: 10:00..15:00 (6 points)
            hours = [d.replace(hour=h, minute=0, second=0, microsecond=0) for h in (10, 11, 12, 13, 14, 15)]
        else:
            # Half-hour anchor: 09:30..14:30 (6 points), optionally add 15:30
            start = d.replace(hour=9, minute=30, second=0, microsecond=0)
            hours = [start + pd.Timedelta(hours=i) for i in range(6)]
            if end_handling in ("keep_half_bar", "pad_to_full_hour"):
                hours.append(start + pd.Timedelta(hours=6))  # 15:30
        out.append(pd.DatetimeIndex(hours, tz=NY_TZ))
    return out[0].append(out[1:]) if len(out) > 1 else (out[0] if out else pd.DatetimeIndex([], tz=NY_TZ))

def process_one_ticker(
    ticker: str,
    custom_path: str,
    indicator_path: str,
    out_dir: str,
    cad: str,
    session: str,
    input_tz_name: str,
    rth_end_handling: str,
    fill_limit: int,
    rclose_fill_limit: int,
    rclose_min_valid: float,
    rclose_extra_limit: int,
    write_qc: bool,
):
    print(f"[{ticker}] reading custom:", custom_path)
    df_custom = read_csv_to_et_index(custom_path, input_tz_name, cad)

    if cad in ("minute", "hour"):
        # Restrict to RTH
        df_custom = df_custom.loc[rth_mask(df_custom.index)]

        # Select columns to keep
        cols_keep = [c for c in CUSTOM_BASE_COLS if c != "timestamp"] + \
                    [c for c in CUSTOM_INDICATOR_COLS if c in df_custom.columns]
        cols_keep = [c for c in cols_keep if c in df_custom.columns]
        df_custom = df_custom.reindex(columns=cols_keep)

        # Build baseline grid per day
        days = pd.to_datetime(df_custom.index.tz_convert(NY_TZ).date).unique()

        # ✅ Adaptive hourly anchor (minute keeps the fixed 09:30..15:59 grid)
        if cad == "hour":
            if len(df_custom):
                # Choose anchor by the mode of minute values; treat 15..45 as :30
                minute_mode = int(pd.Series(df_custom.index.minute).mode().iloc[0])
                anchor = 30 if 15 <= minute_mode <= 45 else 0
            else:
                anchor = 0
            baseline_idx = _gen_hour_grid_adaptive(days, anchor_minute=anchor, end_handling=rth_end_handling)
        else:
            baseline_idx = generate_rth_grid_for_days(days, cad=cad, end_handling=rth_end_handling)

        base = pd.DataFrame(index=baseline_idx)

        # Read indicators and align to baseline
        print(f"[{ticker}] reading indicators:", indicator_path)
        df_ind = read_csv_to_et_index(indicator_path, input_tz_name, cad)
        df_ind = df_ind.loc[rth_mask(df_ind.index)]
        ind_cols = [c for c in INDICATORS_ONLY_COLS if c in df_ind.columns]
        df_ind = df_ind.reindex(columns=ind_cols).reindex(baseline_idx)

        # Align custom side to baseline
        df_base = base.join(df_custom, how="left")

        # Merge both sides
        df = df_base.join(df_ind, how="left")

        # Per-day grouping key
        df["day_id"] = pd.to_datetime(df.index.tz_convert(NY_TZ).date)

        # Pre-fill gap mark on primary price columns
        had_nan_primary = df[PRIMARY_PRICE_COLS].isna().any(axis=1)
        df["is_gap"] = had_nan_primary.astype(np.int8)

        # Per-day, limited two-sided average fill (numeric columns only)
        numeric_cols = [c for c in df.columns if c != "is_gap" and pd.api.types.is_numeric_dtype(df[c])]
        df[numeric_cols] = df.groupby("day_id", group_keys=False)[numeric_cols].apply(
            lambda g: g.apply(lambda s: two_sided_avg_fill_limited(s, limit=fill_limit))
        )

        # Context columns
        df["session"] = "rth"
        df["bar_index_since_open"] = compute_bar_index_since_open(df.index)
        if cad == "minute":
            df["minutes_since_open"] = df["bar_index_since_open"].astype("int32")
        else:
            df.drop(columns=["minutes_since_open"], inplace=True, errors="ignore")

        # Spread to VWAP
        if "vwap" in df.columns and "close" in df.columns:
            df["spread_vwap"] = df["close"] - df["vwap"]
        else:
            df["spread_vwap"] = np.nan

        # Robust r_close (intraday only)
        r_close_series, qc_df = _rclose_repair_and_qc(
            df,
            rclose_fill_limit=rclose_fill_limit,
            rclose_min_valid=rclose_min_valid,
            rclose_extra_limit=rclose_extra_limit
        )
        df["r_close"] = r_close_series

        # Day-mean ATR and normalized spread
        if "atr_14" in df.columns:
            atr_mean = df.groupby("day_id")["atr_14"].transform("mean")
            df["ATR_14_day_mean"] = atr_mean
            df["spread_vwap_norm"] = df["spread_vwap"] / df["ATR_14_day_mean"]
        else:
            df["ATR_14_day_mean"] = np.nan
            df["spread_vwap_norm"] = np.nan

        # Volume z-score by bar-of-day position
        if "volume" in df.columns:
            df["volume_z_by_tod"] = add_volume_z_by_bar_index(df)
        else:
            df["volume_z_by_tod"] = np.nan

    else:
        # cad == "day": no RTH windowing, no intraday r_close
        cols_keep_custom = [c for c in ["open","high","low","close","volume","vwap","transactions"] + CUSTOM_INDICATOR_COLS
                            if c in df_custom.columns]
        df_custom = df_custom.reindex(columns=cols_keep_custom)

        # Normalize index to ET date (tz-aware midnight)
        day_idx = pd.to_datetime(df_custom.index.tz_convert(NY_TZ).date)
        df_custom.index = pd.DatetimeIndex(day_idx, tz=NY_TZ)

        print(f"[{ticker}] reading indicators:", indicator_path)
        df_ind = read_csv_to_et_index(indicator_path, input_tz_name, cad)
        cols_keep_ind = [c for c in INDICATORS_ONLY_COLS if c in df_ind.columns]
        day_idx2 = pd.to_datetime(df_ind.index.tz_convert(NY_TZ).date)
        df_ind.index = pd.DatetimeIndex(day_idx2, tz=NY_TZ)
        df_ind = df_ind.reindex(columns=cols_keep_ind)

        df = df_custom.join(df_ind, how="outer").sort_index()

        df["day_id"] = pd.to_datetime(df.index.tz_convert(NY_TZ).date)
        had_nan_primary = df[PRIMARY_PRICE_COLS].isna().any(axis=1)
        df["is_gap"] = had_nan_primary.astype(np.int8)

        # Daily: no cross-day filling (equivalent to limit=0)
        numeric_cols = [c for c in df.columns if c != "is_gap" and pd.api.types.is_numeric_dtype(df[c])]
        df[numeric_cols] = df[numeric_cols].apply(lambda s: two_sided_avg_fill_limited(s, limit=0))

        df["session"] = "day"
        df["bar_index_since_open"] = 0
        df["spread_vwap"] = (df["close"] - df["vwap"]) if "vwap" in df.columns else np.nan
        df["r_close"] = np.nan
        df["ATR_14_day_mean"] = df["atr_14"] if "atr_14" in df.columns else np.nan
        df["spread_vwap_norm"] = (df["spread_vwap"] / df["ATR_14_day_mean"]) if "atr_14" in df.columns else np.nan
        df["volume_z_by_tod"] = np.nan
        qc_df = pd.DataFrame(columns=["day_id","rclose_valid_ratio","rclose_limit_used","N"])

    # ---- Save results ----
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ticker}.csv")
    df.to_csv(out_path, index_label="timestamp")
    print(f"[{ticker}] done → {out_path} (rows={len(df)})")

    # ---- QC summary ----
    if 'qc_df' in locals() and not qc_df.empty:
        low_days = qc_df.loc[qc_df["rclose_valid_ratio"] < rclose_min_valid]
        ok_ratio = (1.0 - len(low_days) / len(qc_df)) if len(qc_df) else 1.0
        print(f"[{ticker}] r_close QC: days={len(qc_df)}, "
              f"median_valid={qc_df['rclose_valid_ratio'].median():.3f}, "
              f"min_valid={qc_df['rclose_valid_ratio'].min():.3f}, "
              f"days_below_thresh={len(low_days)} "
              f"({ok_ratio:.0%} days meet threshold {rclose_min_valid:.2f})")
        if write_qc:
            qc_dir = os.path.join(out_dir, "_qc")
            os.makedirs(qc_dir, exist_ok=True)
            qc_path = os.path.join(qc_dir, f"{ticker}_rclose_qc.csv")
            qc_df.to_csv(qc_path, index=False)
            print(f"[{ticker}] r_close QC → {qc_path}")
    else:
        print(f"[{ticker}] r_close QC skipped (cadence={cad})")


# ======================== main ======================== #

def main():
    ap = argparse.ArgumentParser(description="Generalized merge & align with YAML control.")
    ap.add_argument("--config", default=r"D:\Data mining project\data_mining_fall_2025\feature_extraction\configs\data_preprocess_config.yaml",
                    help="Path to YAML config.")
    ap.add_argument("--custom_dir", default=None, help="Override custom_indicators dir")
    ap.add_argument("--indicator_dir", default=None, help="Override indicators dir")
    ap.add_argument("--out_dir", default=None, help="Override merged output dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cad = derive_cadence_label(cfg)  # "minute"|"hour"|"day"

    # Resolve IO dirs
    custom_dir, ind_dir, out_dir, write_qc_default = resolve_dirs(cfg, cad)
    if args.custom_dir:    custom_dir = args.custom_dir
    if args.indicator_dir: ind_dir    = args.indicator_dir
    if args.out_dir:       out_dir    = args.out_dir

    # Parameters
    p = get_merge_params(cfg, cad)
    session             = p["session"]
    rth_end_handling    = p["rth_end_handling"]
    fill_limit          = p["fill_limit"]
    rclose_fill_limit   = p["rclose_fill_limit"]
    rclose_min_valid    = p["rclose_min_valid"]
    rclose_extra_limit  = p["rclose_extra_limit"]
    input_tz_name       = p["input_tz_name"]
    write_qc            = write_qc_default

    # Find common tickers
    custom_files = {os.path.splitext(os.path.basename(p))[0]: p
                    for p in glob.glob(os.path.join(custom_dir, "*.csv"))}
    indicator_files = {os.path.splitext(os.path.basename(p))[0]: p
                       for p in glob.glob(os.path.join(ind_dir, "*.csv"))}
    common = sorted(set(custom_files).intersection(indicator_files))
    if not common:
        raise SystemExit(f"No common tickers between:\n  {custom_dir}\n  {ind_dir}")

    # Validate r_close limits
    if rclose_extra_limit < rclose_fill_limit:
        raise SystemExit("merge.rclose.extra_limit_bars must be >= merge.rclose.fill_limit_bars")

    print(f"Cadence: {cad} | Session: {session} | Files: {len(common)} tickers")
    print(f"custom_dir={custom_dir}\nindicator_dir={ind_dir}\nout_dir={out_dir}")

    for tic in common:
        try:
            process_one_ticker(
                ticker=tic,
                custom_path=custom_files[tic],
                indicator_path=indicator_files[tic],
                out_dir=out_dir,
                cad=cad,
                session=session,
                input_tz_name=input_tz_name,
                rth_end_handling=rth_end_handling,
                fill_limit=fill_limit,
                rclose_fill_limit=rclose_fill_limit,
                rclose_min_valid=rclose_min_valid,
                rclose_extra_limit=rclose_extra_limit,
                write_qc=write_qc
            )
        except Exception as e:
            print(f"[{tic:>6}] ERROR: {e}")

if __name__ == "__main__":
    main()
