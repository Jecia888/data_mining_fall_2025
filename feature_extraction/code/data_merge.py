#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge + align minute features from two folders, RTH-only, and preprocess.
**With r_close repair & QC**

Key points:
- Baseline timeline = custom_indicators_minute
- Restrict to RTH (09:30–16:00 America/New_York)
- Join indicators_minute features onto baseline timestamps
- Missing-value imputation = two-sided average within each day, limited length (default 3 min)
- Compute standard preprocessing columns
- **Robust r_close**:
    * Force numeric for close
    * r_close-only light fill per day (limit=1), then log returns (first minute=0)
    * Per-day QC: if r_close valid ratio < min_valid, retry only for those days with extra_limit
    * Never fabricate across long gaps; keep NaN where uncertainty is large
- Output: one CSV per ticker in --out_dir
- Optional QC CSV with per-day r_close valid ratio summary
"""

import os
import glob
import argparse
from typing import List, Tuple
import numpy as np
import pandas as pd

try:
    # Python ≥ 3.9
    from zoneinfo import ZoneInfo
    NY_TZ = ZoneInfo("America/New_York")
except Exception:
    import pytz
    NY_TZ = pytz.timezone("America/New_York")


# ------------------------ column configurations ------------------------ #

# From custom_indicators_minute (keep OHLCV/VWAP/transactions + custom indicators)
CUSTOM_BASE_COLS = [
    "timestamp", "open", "high", "low", "close", "volume", "vwap", "transactions"
]
CUSTOM_INDICATOR_COLS = [
    "macd", "macd_signal", "macd_hist",
    "bb_mid_20", "bb_up_20", "bb_dn_20", "bb_bw_20",
    "stoch_k_14", "stoch_d_14", "obv"
]

# From indicators_minute (take indicator columns only; ignore duplicate OHLCV)
INDICATORS_ONLY_COLS = [
    "sma_20", "sma_50", "ema_12", "ema_26", "rsi_14", "volatility_20", "atr_14"
]

PRIMARY_PRICE_COLS = ["open", "high", "low", "close", "vwap", "volume", "transactions"]


# ------------------------ helper functions ------------------------ #

def read_minutes_csv(path: str) -> pd.DataFrame:
    """Read CSV with 'timestamp' column → tz-aware ET index → sort & drop duplicates."""
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"'timestamp' column not found in {path}")
    ts = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(NY_TZ, nonexistent="shift_forward", ambiguous="NaT")
    else:
        ts = ts.dt.tz_convert(NY_TZ)
    df["timestamp"] = ts
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    df = df.set_index("timestamp")
    return df


def rth_mask(idx: pd.DatetimeIndex) -> pd.Series:
    """Return boolean mask for 09:30 ≤ t <= 16:00 ET."""
    t = idx.tz_convert(NY_TZ)
    return pd.Series(
        [(pd.Timestamp("09:30").time() <= hh <= pd.Timestamp("16:00").time()) for hh in t.time],
        index=idx
    )


def generate_rth_grid_for_days(days: List[pd.Timestamp]) -> pd.DatetimeIndex:
    """Generate minute grid (390 points) for each day in ET, then concatenate."""
    all_idx = []
    for d in days:
        d = pd.Timestamp(d)
        d = d.tz_localize(NY_TZ) if d.tz is None else d.tz_convert(NY_TZ)
        start = d.replace(hour=9, minute=30, second=0, microsecond=0)
        rng = pd.date_range(start=start, periods=390, freq="T", tz=NY_TZ)  # 09:30..15:59
        all_idx.append(rng)
    if not all_idx:
        return pd.DatetimeIndex([], tz=NY_TZ)
    return all_idx[0].append(all_idx[1:]) if len(all_idx) > 1 else all_idx[0]


def two_sided_avg_fill_limited(s: pd.Series, limit: int = 3) -> pd.Series:
    """
    Two-sided average fill with length limit (in minutes).
    - Fill only gaps of length ≤ limit using (ffill + bfill)/2
    - If only one side available within limit, use that side
    - Otherwise keep NaN (left as is_gap for downstream weighting)
    """
    if s.dtype.kind not in "biufc":
        return s  # skip non-numeric
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


def compute_minutes_since_open(index: pd.DatetimeIndex) -> pd.Series:
    """Return 0–389 index within each day."""
    dates = index.tz_convert(NY_TZ).date
    counter = []
    prev = None
    c = -1
    for d in dates:
        if d != prev:
            c = 0
            prev = d
        else:
            c += 1
        counter.append(c)
    return pd.Series(counter, index=index, dtype="int32")


def add_volume_z_by_tod(df: pd.DataFrame) -> pd.Series:
    """Compute z-score of volume by minute-of-day across all days for that ticker."""
    mod = compute_minutes_since_open(df.index)
    df = df.copy()
    df["_mod"] = mod.values
    grp = df.groupby("_mod")["volume"]
    mean = grp.transform("mean")
    std = grp.transform("std").replace(0, np.nan)
    z = (df["volume"] - mean) / std
    return z.fillna(0.0)


# ------------------------ r_close helpers (repair + QC) ------------------------ #

def _fill_limited_per_day(s: pd.Series, limit: int) -> pd.Series:
    """Per-day limited ffill/bfill for a series (no cross-day spill)."""
    return s.ffill(limit=limit).bfill(limit=limit)

def _compute_rclose_from_close(close_series: pd.Series) -> pd.Series:
    """Log returns per day; first minute set to 0."""
    rc = np.log(close_series) - np.log(close_series.shift(1))
    if len(rc) > 0:
        rc.iloc[0] = 0.0
    return rc

def _rclose_repair_and_qc(df: pd.DataFrame,
                          rclose_fill_limit: int = 1,
                          rclose_min_valid: float = 0.70,
                          rclose_extra_limit: int = 2) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Build r_close robustly with:
      - force numeric close
      - per-day limited fill for r_close input
      - mask nonpositive closes
      - QC per day; retry with extra_limit where valid ratio < threshold
    Returns: (r_close_series_aligned_to_df_index, qc_dataframe)
    """
    # Force numeric to avoid hidden strings
    close_numeric = pd.to_numeric(df["close"], errors="coerce")

    # Prepare container aligned to df.index
    r_close_out = pd.Series(index=df.index, dtype=float)

    # Group by ET date
    day_groups = df.groupby(pd.to_datetime(df.index.tz_convert(NY_TZ).date))
    qc_rows = []

    for day, gidx in day_groups.groups.items():
        idx = df.index.isin(df.loc[gidx].index)
        g_close = close_numeric[idx].copy()

        # Stage 1: light fill
        g_filled = _fill_limited_per_day(g_close, limit=rclose_fill_limit)
        g_filled = g_filled.mask(g_filled <= 0, np.nan)  # avoid log(<=0)
        rc1 = _compute_rclose_from_close(g_filled)
        valid1 = np.isfinite(rc1).mean() if len(rc1) else 0.0

        # If low coverage and extra_limit allows, Stage 2 retry only for this day
        if valid1 < rclose_min_valid and rclose_extra_limit > rclose_fill_limit:
            g_filled2 = _fill_limited_per_day(g_close, limit=rclose_extra_limit)
            g_filled2 = g_filled2.mask(g_filled2 <= 0, np.nan)
            rc2 = _compute_rclose_from_close(g_filled2)
            valid2 = np.isfinite(rc2).mean() if len(rc2) else 0.0

            # Choose the better one (higher valid ratio)
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


# ------------------------ main pipeline per ticker ------------------------ #

def process_one_ticker(ticker: str,
                       custom_path: str,
                       indicator_path: str,
                       out_dir: str,
                       fill_limit: int = 3,
                       rclose_fill_limit: int = 1,
                       rclose_min_valid: float = 0.70,
                       rclose_extra_limit: int = 2,
                       write_qc: bool = True):
    print(f"[{ticker}] reading custom:", custom_path)
    df_custom = read_minutes_csv(custom_path)

    # Restrict to RTH-only (Regular Trading Hours)
    df_custom = df_custom.loc[rth_mask(df_custom.index)]

    # Keep only selected base + indicator columns
    cols_keep = [c for c in CUSTOM_BASE_COLS if c != "timestamp"] + [c for c in CUSTOM_INDICATOR_COLS if c in df_custom.columns]
    df_custom = df_custom.reindex(columns=cols_keep)

    # Build baseline minute grid (09:30..15:59) for all ET-days present in custom
    days = pd.to_datetime(df_custom.index.tz_convert(NY_TZ).date).unique()
    baseline_idx = generate_rth_grid_for_days(days)
    base = pd.DataFrame(index=baseline_idx)

    # Align custom data to baseline
    df_base = base.join(df_custom, how="left")

    # Read indicators_minute data, restrict to RTH, align to baseline
    print(f"[{ticker}] reading indicators:", indicator_path)
    df_ind = read_minutes_csv(indicator_path)
    df_ind = df_ind.loc[rth_mask(df_ind.index)]
    ind_cols = [c for c in INDICATORS_ONLY_COLS if c in df_ind.columns]
    df_ind = df_ind.reindex(columns=ind_cols).reindex(baseline_idx)

    # Combine datasets
    df = df_base.join(df_ind, how="left")

    # Add day_id BEFORE filling (for per-day fill grouping)
    df["day_id"] = pd.to_datetime(df.index.tz_convert(NY_TZ).date)

    # Flag rows with NaNs in primary columns BEFORE global filling
    had_nan_primary = df[PRIMARY_PRICE_COLS].isna().any(axis=1)
    df["is_gap"] = had_nan_primary.astype(np.int8)

    # ---- General per-day limited fill for numeric cols (avoid cross-day) ----
    numeric_cols = [c for c in df.columns if c != "is_gap" and pd.api.types.is_numeric_dtype(df[c])]
    df[numeric_cols] = df.groupby("day_id", group_keys=False)[numeric_cols].apply(
        lambda g: g.apply(lambda s: two_sided_avg_fill_limited(s, limit=fill_limit))
    )

    # Context columns
    df["session"] = "rth"
    df["minutes_since_open"] = compute_minutes_since_open(df.index)

    # Derived: spread_vwap
    if "vwap" in df.columns and "close" in df.columns:
        df["spread_vwap"] = df["close"] - df["vwap"]
    else:
        df["spread_vwap"] = np.nan

    # -------- Robust r_close (repair + QC) --------
    r_close_series, qc_df = _rclose_repair_and_qc(
        df,
        rclose_fill_limit=rclose_fill_limit,
        rclose_min_valid=rclose_min_valid,
        rclose_extra_limit=rclose_extra_limit
    )
    df["r_close"] = r_close_series

    # Per-day mean ATR_14 & normalized spread_vwap
    if "atr_14" in df.columns:
        atr_mean = df.groupby("day_id")["atr_14"].transform("mean")
        df["ATR_14_day_mean"] = atr_mean
        df["spread_vwap_norm"] = df["spread_vwap"] / df["ATR_14_day_mean"]
    else:
        df["ATR_14_day_mean"] = np.nan
        df["spread_vwap_norm"] = np.nan

    # Volume z-score by minute-of-day
    if "volume" in df.columns:
        df["volume_z_by_tod"] = add_volume_z_by_tod(df)
    else:
        df["volume_z_by_tod"] = np.nan

    # ---- Save results ----
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ticker}.csv")
    df.to_csv(out_path, index_label="timestamp")
    print(f"[{ticker}] done → {out_path} (rows={len(df)})")

    # ---- QC summary (print & optional csv) ----
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


def main():
    ap = argparse.ArgumentParser(description="Merge & align minute features (RTH-only) + robust r_close.")
    ap.add_argument("--custom_dir",
        default=r"D:\Data mining project\data_mining_fall_2025\data_preprocessing_pipeline\output\custom_indicators_minute",
        help="Path to custom_indicators_minute/")
    ap.add_argument("--indicator_dir",
        default=r"D:\Data mining project\data_mining_fall_2025\data_preprocessing_pipeline\output\indicators_minute",
        help="Path to indicators_minute/")
    ap.add_argument("--out_dir",
        default=r"D:\Data mining project\data_mining_fall_2025\feature_extraction\output\intermediate_dataset",
        help="Output directory for merged CSVs")

    # General fill limit for global numeric columns
    ap.add_argument("--fill_limit", type=int, default=3,
        help="Maximum consecutive minutes to fill using two-sided averaging within a day (general columns).")

    # r_close-specific repair & QC
    ap.add_argument("--rclose_fill_limit", type=int, default=1,
        help="Per-day limited fill used ONLY for r_close input (first pass).")
    ap.add_argument("--rclose_min_valid", type=float, default=0.70,
        help="Minimum per-day valid ratio required for r_close; below this we try extra_limit.")
    ap.add_argument("--rclose_extra_limit", type=int, default=2,
        help="If a day's r_close valid ratio < threshold, retry that day with this higher limit (must be >= rclose_fill_limit).")
    ap.add_argument("--write_qc", action="store_true",
        help="Write per-day r_close QC CSV under <out_dir>/_qc/.")

    args = ap.parse_args()

    # Find common tickers present in BOTH folders
    custom_files = {os.path.splitext(os.path.basename(p))[0]: p
                    for p in glob.glob(os.path.join(args.custom_dir, "*.csv"))}
    indicator_files = {os.path.splitext(os.path.basename(p))[0]: p
                       for p in glob.glob(os.path.join(args.indicator_dir, "*.csv"))}

    common = sorted(set(custom_files).intersection(indicator_files))
    if not common:
        raise SystemExit("No common tickers found between the two folders.")

    print(f"Tickers to process ({len(common)}): {', '.join(common)}")

    # Validate extra limit config
    if args.rclose_extra_limit < args.rclose_fill_limit:
        raise SystemExit("--rclose_extra_limit must be >= --rclose_fill_limit")

    for tic in common:
        try:
            process_one_ticker(
                tic,
                custom_files[tic],
                indicator_files[tic],
                args.out_dir,
                fill_limit=args.fill_limit,
                rclose_fill_limit=args.rclose_fill_limit,
                rclose_min_valid=args.rclose_min_valid,
                rclose_extra_limit=args.rclose_extra_limit,
                write_qc=args.write_qc
            )
        except Exception as e:
            print(f"[{tic}] ERROR: {e}")


if __name__ == "__main__":
    main()
