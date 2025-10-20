#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build daily simple aggregation features from merged minute data (RTH 09:30–15:59),
with coverage-aware gating for window features.

Key changes vs. original:
- For short-window features (ret_open15, ret_last10, reversal_score), if r_close valid coverage
  within the window (or whole day for reversal) is below a threshold, output NaN instead of 0.
- Add *_valid_ratio companion columns to make downstream weighting/filtering easy.
- Optionally gate trend_persistence: if valid r_close count in the day is too low, set NaN.

Default thresholds:
- min_valid_ratio_open15 = 0.50
- min_valid_ratio_last10 = 0.50
- min_valid_ratio_reversal = 0.50  (ratio of valid r_close across the full day)
- min_valid_count_run = 5          (min valid minutes required to trust trend_persistence)
"""

import os
import glob
import argparse
from typing import Tuple
import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
    NY_TZ = ZoneInfo("America/New_York")
except Exception:
    import pytz
    NY_TZ = pytz.timezone("America/New_York")


# ------------------ utils: timestamp parsing ------------------ #

def read_minutes_csv(path: str) -> pd.DataFrame:
    """Read a minutes CSV, parse tz-aware ET timestamp as index."""
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"'timestamp' column not found in {path}")
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(NY_TZ, nonexistent="shift_forward", ambiguous="NaT")
    else:
        ts = ts.dt.tz_convert(NY_TZ)
    df = df.drop(columns=["timestamp"])
    df.index = ts
    df = df[~df.index.isna()].sort_index()
    return df


def et_day_bounds(day: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return (09:30, 16:00) ET timestamps for a given ET date."""
    d = pd.Timestamp(day)
    d = d.tz_localize(NY_TZ) if d.tz is None else d.tz_convert(NY_TZ)
    start = d.replace(hour=9, minute=30, second=0, microsecond=0)
    end = d.replace(hour=16, minute=0, second=0, microsecond=0)
    return start, end


# ------------------ quality metrics ------------------ #

def gap_quality_metrics(day_df: pd.DataFrame) -> Tuple[float, int]:
    """Compute gap ratio and max consecutive gap for a day's minutes."""
    if "is_gap" not in day_df.columns:
        return 0.0, 0
    g = day_df["is_gap"].astype(int).to_numpy()
    gap_ratio = float(np.mean(g)) if len(g) else 0.0
    max_run = 0
    cur = 0
    for v in g:
        if v == 1:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return gap_ratio, int(max_run)


def longest_same_sign_run(r: np.ndarray) -> int:
    """Longest consecutive run length of same sign in r_close. NaN/0 break runs."""
    max_run = 0
    cur = 0
    prev_sign = 0
    for v in r:
        if not np.isfinite(v) or v == 0.0:
            cur = 0
            prev_sign = 0
            continue
        s = 1 if v > 0 else -1
        if s == prev_sign:
            cur += 1
        else:
            cur = 1
            prev_sign = s
        max_run = max(max_run, cur)
    return int(max_run)


def pearson_corr_masked(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Pearson correlation on paired finite values under mask; else NaN."""
    if x.size != y.size or x.size != mask.size:
        return np.nan
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return np.nan
    x = x - np.nanmean(x)
    y = y - np.nanmean(y)
    sx = np.nanstd(x)
    sy = np.nanstd(y)
    if not np.isfinite(sx) or not np.isfinite(sy) or sx == 0 or sy == 0:
        return np.nan
    return float(np.nanmean((x / sx) * (y / sy)))


# ------------------ coverage-aware helpers ------------------ #

def window_sum_with_gating(r: np.ndarray, mask: np.ndarray, min_valid_ratio: float):
    """
    Sum r within mask; return (value, valid_ratio).
    If valid_ratio < threshold or window empty, return (np.nan, valid_ratio).
    """
    idx = mask
    if idx.sum() == 0:
        return np.nan, 0.0
    r_w = r[idx]
    valid_ratio = float(np.isfinite(r_w).mean()) if r_w.size else 0.0
    if r_w.size and valid_ratio >= min_valid_ratio:
        return float(np.nansum(r_w)), valid_ratio
    else:
        return np.nan, valid_ratio


def corr_with_gating(tod: np.ndarray, r: np.ndarray, min_valid_ratio: float):
    """
    Pearson corr between minute-of-day (tod) and cumret computed on valid r only.
    Gate by valid_ratio across the FULL DAY; if below threshold -> NaN.
    """
    valid = np.isfinite(r) & np.isfinite(tod)
    valid_ratio = float(valid.mean()) if valid.size else 0.0
    if valid_ratio < min_valid_ratio:
        return np.nan, valid_ratio
    # cumret uses r with NaN->0, but correlation pairs are masked by valid
    r_fill0 = np.where(np.isfinite(r), r, 0.0)
    cumret = np.cumsum(r_fill0)
    val = pearson_corr_masked(tod, cumret, valid)
    return val, valid_ratio


# ------------------ per-ticker pipeline ------------------ #

def build_agg_for_ticker(ticker: str,
                         in_path: str,
                         out_dir: str,
                         min_valid_ratio_open15: float = 0.50,
                         min_valid_ratio_last10: float = 0.50,
                         min_valid_ratio_reversal: float = 0.50,
                         min_valid_count_run: int = 5,
                         emit_validity_cols: bool = True):
    print(f"[{ticker}] reading:", in_path)
    df = read_minutes_csv(in_path)

    # 只用 RTH
    if "session" in df.columns:
        df = df.loc[df["session"] == "rth"]

    # 必需列检查
    required_hard = ["spread_vwap_norm", "volume_z_by_tod", "volume",
                     "minutes_since_open", "day_id", "is_gap", "close"]
    for c in required_hard:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {in_path}")

    # 若缺 r_close，则现场按日重算（首分钟 NaN）
    if "r_close" not in df.columns:
        df["r_close"] = df.groupby(pd.to_datetime(df.index.tz_convert(NY_TZ).date))["close"].apply(
            lambda s: np.log(s) - np.log(s.shift(1))
        )

    # day_id 统一为 tz-aware ET 日期
    if not pd.api.types.is_datetime64_any_dtype(df["day_id"]):
        df["day_id"] = pd.to_datetime(df["day_id"], errors="coerce")
    if getattr(df["day_id"].dt, "tz", None) is None:
        df["day_id"] = df["day_id"].dt.tz_localize(NY_TZ)

    groups = dict(tuple(df.groupby(df["day_id"].dt.date)))  # 按 ET 日期分组

    rows = []
    for d_date, g in groups.items():
        day = pd.Timestamp(d_date).tz_localize(NY_TZ)
        g = g.sort_index()
        n = len(g)

        # 质量
        gap_ratio, max_run_gap = gap_quality_metrics(g)

        # --- 核心序列 ---
        r = g["r_close"].to_numpy(dtype=float)                  # 保留 NaN
        svn = g["spread_vwap_norm"].to_numpy(dtype=float)       # 可含 NaN
        vz = g["volume_z_by_tod"].to_numpy(dtype=float)         # 可含 NaN
        vol = g["volume"].to_numpy(dtype=float)                 # 可含 NaN
        tod = g["minutes_since_open"].to_numpy(dtype=float)

        # 窗口掩码
        idx_open15 = (tod >= 0) & (tod <= 14)
        idx_last10 = (tod >= 380) & (tod <= 389)   # 15:50–15:59

        # ---- 全日聚合（NaN-safe）----
        ret_oc = float(np.nansum(r)) if not np.isnan(r).all() else np.nan
        r2 = np.square(r)
        rv = float(np.nansum(r2)) if not np.isnan(r2).all() else np.nan

        # ---- 开盘15分钟：覆盖率门控 ----
        ret_open15, open15_valid = window_sum_with_gating(
            r, idx_open15, min_valid_ratio_open15
        )

        # ---- 尾盘10分钟：覆盖率门控 ----
        ret_last10, last10_valid = window_sum_with_gating(
            r, idx_last10, min_valid_ratio_last10
        )

        # ---- trend_persistence：有效样本不足则 NaN ----
        valid_count_day = int(np.isfinite(r).sum())
        if valid_count_day >= min_valid_count_run:
            trend_persistence = longest_same_sign_run(r)
        else:
            trend_persistence = np.nan

        # ---- reversal_score：全日覆盖率门控 ----
        reversal_score, rev_valid = corr_with_gating(
            tod, r, min_valid_ratio_reversal
        )

        # ---- 其余（按原逻辑）----
        mabs_svn = float(np.nanmean(np.abs(svn))) if np.isfinite(np.nanmean(np.abs(svn))) else np.nan
        eod_svn = float(pd.Series(svn).dropna().iloc[-1]) if np.isfinite(svn).any() else np.nan

        vol_total = float(np.nansum(vol)) if np.isfinite(vol).any() else np.nan
        vol_last10 = np.nansum(vol[idx_last10]) if (vol[idx_last10].size and np.isfinite(vol[idx_last10]).any()) else np.nan
        vol_last10_frac = (vol_last10 / vol_total) if (np.isfinite(vol_last10) and np.isfinite(vol_total) and vol_total > 0) else np.nan

        mask_vz = np.isfinite(vz)
        hi2_frac = float(np.mean(vz[mask_vz] > 2)) if mask_vz.any() else np.nan

        # 时间戳
        day_start_ts, day_end_ts = et_day_bounds(day)

        row = {
            "ticker": ticker,
            "day_id": day.date(),            # YYYY-MM-DD (ET)
            "day_start_ts": day_start_ts,    # 09:30 ET tz-aware
            "day_end_ts": day_end_ts,        # 16:00 ET tz-aware
            "feature_ts": day_end_ts,        # 这一天的特征时间戳
            # 质量
            "AGG_gap_ratio": gap_ratio,
            "AGG_max_consecutive_gap": max_run_gap,
            "AGG_N": int(n),
            # 10 features（带覆盖率门控后的）
            "ret_oc": ret_oc,
            "rv": rv,
            "ret_open15": ret_open15,
            "ret_last10": ret_last10,
            "trend_persistence": trend_persistence,
            "reversal_score": reversal_score,
            "mabs_svn": mabs_svn,
            "eod_svn": eod_svn,
            "vol_last10_frac": vol_last10_frac,
            "hi2_frac": hi2_frac,
        }

        if emit_validity_cols:
            row.update({
                "open15_valid_ratio": open15_valid,
                "last10_valid_ratio": last10_valid,
                "reversal_valid_ratio": rev_valid,
                "rclose_valid_count": valid_count_day,
                "rclose_valid_ratio_day": float(np.isfinite(r).mean()) if r.size else np.nan,
            })

        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values(["day_id"])

    # ---- write to <out_dir>/<ticker>/agg/agg_daily.csv ----
    out_ticker_dir = os.path.join(out_dir, ticker, "agg")
    os.makedirs(out_ticker_dir, exist_ok=True)
    out_path = os.path.join(out_ticker_dir, "agg_daily.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[{ticker}] AGG → {out_path} (days={len(out_df)})")


# ------------------ main ------------------ #

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build daily simple aggregation features per ticker (RTH 09:30–15:59), coverage-aware.")
    ap.add_argument("--input_dir",
                    default=r"D:\Data mining project\data_mining_fall_2025\feature_extraction\output\intermediate_dataset",
                    help="Path to merged/preprocessed per-ticker minutes (CSV files).")
    ap.add_argument("--out_dir",
                    default=r"D:\Data mining project\data_mining_fall_2025\feature_extraction\output\features",
                    help="Root output directory. Will create <out_dir>/<ticker>/agg/agg_daily.csv.")

    # Coverage thresholds (can be tuned)
    ap.add_argument("--min_valid_ratio_open15", type=float, default=0.50,
                    help="Min valid ratio of r_close within first 15 minutes to output ret_open15; else NaN.")
    ap.add_argument("--min_valid_ratio_last10", type=float, default=0.50,
                    help="Min valid ratio of r_close within last 10 minutes to output ret_last10; else NaN.")
    ap.add_argument("--min_valid_ratio_reversal", type=float, default=0.50,
                    help="Min valid ratio of r_close across the day to output reversal_score; else NaN.")
    ap.add_argument("--min_valid_count_run", type=int, default=5,
                    help="Min valid r_close count in the day to output trend_persistence; else NaN.")
    ap.add_argument("--emit_validity_cols", action="store_true",
                    help="Also write *_valid_ratio and rclose_valid_* helper columns.")

    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found under --input_dir={args.input_dir}")

    for p in files:
        ticker = os.path.splitext(os.path.basename(p))[0]
        try:
            build_agg_for_ticker(
                ticker, p, args.out_dir,
                min_valid_ratio_open15=args.min_valid_ratio_open15,
                min_valid_ratio_last10=args.min_valid_ratio_last10,
                min_valid_ratio_reversal=args.min_valid_ratio_reversal,
                min_valid_count_run=args.min_valid_count_run,
                emit_validity_cols=args.emit_validity_cols
            )
        except Exception as e:
            print(f"[{ticker}] ERROR: {e}")
