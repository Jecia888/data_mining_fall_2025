#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build daily FFT features from merged minute data.

Input (per-ticker CSV in --input_dir):
  timestamp (tz-aware ET), open, high, low, close, volume, vwap, transactions,
  macd, macd_signal, macd_hist, bb_mid_20, bb_up_20, bb_dn_20, bb_bw_20,
  stoch_k_14, stoch_d_14, obv, sma_20, sma_50, ema_12, ema_26, rsi_14,
  volatility_20, atr_14, is_gap, session, day_id, minutes_since_open,
  spread_vwap, r_close, ATR_14_day_mean, spread_vwap_norm, volume_z_by_tod

Output (per ticker):
  <out_dir>/<TICKER>/fft/fft_daily.csv  (one row per day with FFT features)
"""

import os
import glob
import argparse
from typing import Tuple, Dict
import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
    NY_TZ = ZoneInfo("America/New_York")
except Exception:
    import pytz
    NY_TZ = pytz.timezone("America/New_York")

from scipy.signal import welch


# ------------------ utils: timestamp parsing ------------------ #

def read_minutes_csv(path: str) -> pd.DataFrame:
    """Read a minutes CSV and parse tz-aware ET timestamp as index."""
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"'timestamp' column not found in {path}")
    # More robust: don’t force utc=True; keep -04:00/-05:00 offsets if present, otherwise localize to ET
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


# ------------------ FFT core (robust detrending) ------------------ #

def detrend_linear_safe(x: np.ndarray) -> np.ndarray:
    """
    Robust linear detrending:
    - Fit only on finite values; skip if <3 valid points or very low variance.
    - On any error (including SVD non-convergence), return original series.
    """
    n = x.shape[0]
    if n < 3:
        return x
    t = np.arange(n, dtype=float)
    mask = np.isfinite(x)
    if mask.sum() < 3:
        return x
    xv = x[mask]
    tv = t[mask]
    if np.nanstd(xv) < 1e-12 or (tv.max() - tv.min()) < 1:
        return x
    try:
        a, b = np.polyfit(tv, xv, 1)
    except Exception:
        return x
    return x - (a * t + b)


def welch_psd(x: np.ndarray, fs: float = 1.0, nperseg: int = 128, noverlap: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD with Hann window (auto-adjust for short sequences)."""
    nseg = max(8, min(nperseg, len(x)))
    novl = max(0, min(noverlap, nseg - 1))
    f, Pxx = welch(x, fs=fs, window="hann", nperseg=nseg, noverlap=novl,
                   detrend=False, scaling="spectrum", return_onesided=True)
    return f, Pxx


def band_masks(f: np.ndarray) -> Dict[str, np.ndarray]:
    """Build boolean masks for low/mid/high bands (cycles per minute)."""
    lo = f < (1.0 / 60.0)                          # period > 60 min
    mid = (f >= (1.0 / 60.0)) & (f < (1.0 / 15.0)) # 15–60 min
    hi = f >= (1.0 / 15.0)                         # period < 15 min
    lo = lo & (f > 0)                              # exclude DC
    return {"lo": lo, "mid": mid, "hi": hi}


def spectral_features(f: np.ndarray, P: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    """Compute power ratios, centroid, flatness, peak frequency & ratio."""
    mask_nonzero = f > 0
    f = f[mask_nonzero]
    P = P[mask_nonzero]
    total = np.sum(P)
    if not np.isfinite(total) or total <= eps:
        return {k: np.nan for k in [
            "power_lo","power_mid","power_hi","centroid","flatness","peak_freq","peak_ratio"
        ]}

    masks = band_masks(f)
    power_lo = float(np.sum(P[masks["lo"]]) / total)
    power_mid = float(np.sum(P[masks["mid"]]) / total)
    power_hi = float(np.sum(P[masks["hi"]]) / total)

    centroid = float(np.sum(f * P) / total)
    flatness = float(np.exp(np.mean(np.log(P + eps))) / (np.mean(P) + eps))

    idx = int(np.argmax(P))
    peak_freq = float(f[idx])
    peak_ratio = float(P[idx] / total)

    return {
        "power_lo": power_lo,
        "power_mid": power_mid,
        "power_hi": power_hi,
        "centroid": centroid,
        "flatness": flatness,
        "peak_freq": peak_freq,
        "peak_ratio": peak_ratio,
    }


def prep_series_for_fft(s: pd.Series, mode: str) -> np.ndarray:
    """
    Prepare a time series for FFT:
      - Mean-center (keep NaN intact)
      - Apply robust linear detrending if spread_vwap_norm
      - Replace NaN/Inf with 0
      - Standardize to std=1 (if std==0, return zeros)
    """
    x = s.to_numpy(dtype=float)
    mu = np.nanmean(x)
    if np.isfinite(mu):
        x = x - mu

    if mode == "spread_vwap_norm":
        x = detrend_linear_safe(x)

    x[~np.isfinite(x)] = 0.0

    std = np.std(x)
    if not np.isfinite(std) or std == 0:
        return x * 0.0
    return x / std


def safe_period(freq: float) -> float:
    """Convert frequency to period (minutes); return NaN if invalid (avoid inf)."""
    if freq is None or not np.isfinite(freq) or freq <= 0:
        return np.nan
    return 1.0 / freq


# ------------------ quality metrics ------------------ #

def gap_quality_metrics(day_df: pd.DataFrame) -> Tuple[float, int]:
    """Compute gap ratio and max consecutive gap length for one trading day."""
    if "is_gap" not in day_df.columns:
        return 0.0, 0
    g = day_df["is_gap"].astype(int).to_numpy()
    gap_ratio = float(np.mean(g))
    # longest consecutive ones
    max_run = 0
    cur = 0
    for v in g:
        if v == 1:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return gap_ratio, int(max_run)


# ------------------ per-ticker pipeline ------------------ #

def build_fft_for_ticker(ticker: str, in_path: str, out_dir: str,
                         gap_ratio_thresh: float = 0.10, max_gap_run_thresh: int = 5,
                         do_bb_bw: bool = False):
    print(f"[{ticker}] reading:", in_path)
    df = read_minutes_csv(in_path)

    # Use RTH (regular trading hours) only
    if "session" in df.columns:
        df = df.loc[df["session"] == "rth"]

    # Check required columns
    required_cols = ["close", "spread_vwap_norm", "volume_z_by_tod", "day_id", "is_gap"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {in_path}")

    # Convert day_id → ET date (tz-aware 00:00)
    if not pd.api.types.is_datetime64_any_dtype(df["day_id"]):
        df["day_id"] = pd.to_datetime(df["day_id"], errors="coerce")
    if getattr(df["day_id"].dt, "tz", None) is None:
        df["day_id"] = df["day_id"].dt.tz_localize(NY_TZ)

    groups = dict(tuple(df.groupby(df["day_id"].dt.date)))  # group by ET date

    rows = []
    for d_date, g in groups.items():
        day = pd.Timestamp(d_date).tz_localize(NY_TZ)
        g = g.sort_index()
        n = len(g)

        # Gap quality metrics
        gap_ratio, max_run = gap_quality_metrics(g)
        low_conf = int((gap_ratio > gap_ratio_thresh) or (max_run > max_gap_run_thresh) or (n < 300))

        # ---- Recalculate r_close (intra-day, first minute = 0) ----
        rc = np.log(g["close"]) - np.log(g["close"].shift(1))
        if len(rc) > 0:
            rc.iloc[0] = 0.0

        # Degeneracy protection
        rc_centered = rc.to_numpy(dtype=float) - np.nanmean(rc.to_numpy(dtype=float))
        rc_std = np.nanstd(rc_centered)
        rclose_degenerate = (not np.isfinite(rc_std)) or (rc_std < 1e-8)
        if rclose_degenerate:
            low_conf = 1

        # Prepare input series
        seqs = {
            "spread_vwap_norm": g["spread_vwap_norm"],
            "volume_z_by_tod": g["volume_z_by_tod"],
        }
        if do_bb_bw and ("bb_bw_20" in g.columns):
            seqs["bb_bw_20"] = g["bb_bw_20"]

        feats = {}

        # Compute for steady-state series
        for name, s in seqs.items():
            x = prep_series_for_fft(s, mode=name)
            f, P = welch_psd(x, fs=1.0, nperseg=128, noverlap=64)
            sp = spectral_features(f, P)
            feats.update({
                f"FFT_{name}_power_lo": sp["power_lo"],
                f"FFT_{name}_power_mid": sp["power_mid"],
                f"FFT_{name}_power_hi": sp["power_hi"],
                f"FFT_{name}_centroid": sp["centroid"],                          # cycles/min
                f"FFT_{name}_centroid_period_min": safe_period(sp["centroid"]),   # minutes
                f"FFT_{name}_flatness": sp["flatness"],
                f"FFT_{name}_peak_freq": sp["peak_freq"],                         # cycles/min
                f"FFT_{name}_peak_period_min": safe_period(sp["peak_freq"]),      # minutes
                f"FFT_{name}_peak_ratio": sp["peak_ratio"],
            })

        # Compute for r_close (or all NaN if degenerate)
        if rclose_degenerate:
            feats.update({
                "FFT_r_close_power_lo": np.nan,
                "FFT_r_close_power_mid": np.nan,
                "FFT_r_close_power_hi": np.nan,
                "FFT_r_close_centroid": np.nan,
                "FFT_r_close_centroid_period_min": np.nan,
                "FFT_r_close_flatness": np.nan,
                "FFT_r_close_peak_freq": np.nan,
                "FFT_r_close_peak_period_min": np.nan,
                "FFT_r_close_peak_ratio": np.nan,
            })
        else:
            x = prep_series_for_fft(rc, mode="r_close")
            f, P = welch_psd(x, fs=1.0, nperseg=128, noverlap=64)
            sp = spectral_features(f, P)
            feats.update({
                "FFT_r_close_power_lo": sp["power_lo"],
                "FFT_r_close_power_mid": sp["power_mid"],
                "FFT_r_close_power_hi": sp["power_hi"],
                "FFT_r_close_centroid": sp["centroid"],
                "FFT_r_close_centroid_period_min": safe_period(sp["centroid"]),
                "FFT_r_close_flatness": sp["flatness"],
                "FFT_r_close_peak_freq": sp["peak_freq"],
                "FFT_r_close_peak_period_min": safe_period(sp["peak_freq"]),
                "FFT_r_close_peak_ratio": sp["peak_ratio"],
            })

        # Day timestamps
        day_start_ts, day_end_ts = et_day_bounds(day)
        rows.append({
            "ticker": ticker,
            "day_id": day.date(),                     # YYYY-MM-DD (ET)
            "day_start_ts": day_start_ts,             # 09:30 ET tz-aware
            "day_end_ts": day_end_ts,                 # 16:00 ET tz-aware
            "feature_ts": day_end_ts,                 # feature timestamp for the day
            "FFT_flag_low_conf": low_conf,
            "FFT_gap_ratio": gap_ratio,
            "FFT_max_consecutive_gap": max_run,
            "FFT_N": int(n),
            **feats
        })

    out_df = pd.DataFrame(rows).sort_values(["day_id"])
    # ---- Write to <out_dir>/<ticker>/fft/fft_daily.csv ----
    out_ticker_dir = os.path.join(out_dir, ticker, "fft")
    os.makedirs(out_ticker_dir, exist_ok=True)
    out_path = os.path.join(out_ticker_dir, "fft_daily.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[{ticker}] FFT → {out_path} (days={len(out_df)})")


# ------------------ main ------------------ #

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build daily FFT features per ticker.")
    ap.add_argument("--input_dir", default=r"D:\Data mining project\data_mining_fall_2025\feature_extraction\output\intermediate_dataset",
                    help="Path to merged/preprocessed per-ticker minutes (CSV files).")
    ap.add_argument("--out_dir", default=r"D:\Data mining project\data_mining_fall_2025\feature_extraction\output\features",
                    help="Root output directory. Will create <out_dir>/<ticker>/fft/fft_daily.csv.")
    ap.add_argument("--include_bb_bw", action="store_true",
                    help="Also compute FFT on bb_bw_20 as a volatility rhythm proxy.")
    ap.add_argument("--gap_ratio_thresh", type=float, default=0.10,
                    help="If daily gap ratio > threshold, mark FFT_flag_low_conf=1.")
    ap.add_argument("--max_gap_run_thresh", type=int, default=5,
                    help="If max consecutive gap minutes > threshold, mark low confidence.")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found under --input_dir={args.input_dir}")

    for p in files:
        ticker = os.path.splitext(os.path.basename(p))[0]
        try:
            build_fft_for_ticker(
                ticker, p, args.out_dir,
                gap_ratio_thresh=args.gap_ratio_thresh,
                max_gap_run_thresh=args.max_gap_run_thresh,
                do_bb_bw=args.include_bb_bw
            )
        except Exception as e:
            print(f"[{ticker}] ERROR: {e}")
