#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CWT (full-window mode) — reporting cadence mirrors FFT.

Reads merged per-ticker CSVs, slices to RTH for intraday cadences, builds
reporting windows (day/week/month/quarter/year), concatenates all bars within
each window, runs a Continuous Wavelet Transform (CWT), and writes window-level
features to:
  <features_root>/<TICKER>/cwt_<sampling>/<cadence>.csv

Configure via the YAML `cwt` section.
"""

import os
import sys
import glob
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml

try:
    import pywt
except Exception:
    print("❌ PyWavelets not installed. Run: pip install pywavelets", file=sys.stderr)
    raise

try:
    from zoneinfo import ZoneInfo
    NY = ZoneInfo("America/New_York")
except Exception:
    import pytz
    NY = pytz.timezone("America/New_York")


# ───────────────────────── YAML / cadence helpers ───────────────────────── #

def load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}

def normalize_granularity(ts: str) -> str:
    mapping = {
        "minute": "minute", "min": "minute", "m": "minute",
        "hour": "hour", "hr": "hour", "h": "hour",
        "day": "day", "daily": "day", "d": "day"
    }
    out = mapping.get((ts or "").strip().lower())
    if not out:
        raise SystemExit("date_range.granularity must be one of: minute|hour|day")
    return out

def derive_sampling(cfg: dict) -> str:
    return normalize_granularity(cfg.get("date_range", {}).get("granularity", "minute"))

def allowed_reporting_for(sampling: str) -> List[str]:
    if sampling in ("minute", "hour"):
        return ["day", "week", "month", "quarter", "year"]
    if sampling == "day":
        return ["week", "month", "quarter", "year"]
    return []

def cadence_to_pd_freq(cad: str) -> str:
    return {"day": "D", "week": "W-FRI", "month": "M", "quarter": "Q", "year": "Y"}[cad]

def expected_bars_per_rth_day(sampling: str) -> int:
    if sampling == "minute":
        return 390  # 09:30..15:59 ET
    if sampling == "hour":
        return 6    # 09:30,10:30,11:30,12:30,13:30,14:30
    if sampling == "day":
        return 1
    raise ValueError(f"unsupported sampling {sampling}")

def fill_path_template(p: Optional[str], cad: str) -> Optional[str]:
    return None if not p else p.replace("{cad}", cad)


# ─────────────────────────── IO helpers ─────────────────────────── #

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def list_tickers(merged_dir: str) -> Dict[str, str]:
    """Return {TICKER: path_to_csv} for all per-ticker merged CSVs."""
    return {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(merged_dir, "*.csv"))}

def read_intermediate_csv(path: str, tz: str = "America/New_York") -> pd.DataFrame:
    """Read a merged CSV and return a tz-aware ET-indexed DataFrame."""
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"'timestamp' not in {path}")
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["timestamp"] = ts.dt.tz_convert(NY)
    df = df.set_index("timestamp").sort_index()
    return df


def slice_rth_if_needed(df: pd.DataFrame, sampling: str) -> pd.DataFrame:
    """
    For intraday (minute/hour), keep only RTH rows. If 'session' is present
    (already RTH-filtered), this becomes a no-op.
    """
    if sampling in ("minute", "hour"):
        if "session" in df.columns:
            return df.loc[df["session"] == "rth"]
        t = df.index.tz_convert(NY)
        start = pd.Timestamp("09:30").time()
        end   = pd.Timestamp("16:00").time()
        mask = pd.Series([(start <= hh < end) for hh in t.time], index=df.index)
        return df.loc[mask]
    return df


# ─────────────────────── windowing & quality ─────────────────────── #

def build_windows(df: pd.DataFrame, cadence: str) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]]:
    """
    Group rows into reporting windows by cadence (day/week/month/quarter/year).
    Returns a list of (win_start, win_end, df_window).
    """
    freq = cadence_to_pd_freq(cadence)
    grouped = df.groupby(pd.Grouper(freq=freq, label="right", closed="right"))
    out = []
    for label, sub in grouped:
        if sub.empty:
            continue
        win_end = pd.Timestamp(label).tz_localize(NY) if label.tz is None else label.tz_convert(NY)
        win_start = sub.index.min()
        out.append((win_start, win_end, sub))
    return out

def longest_true_run(mask: np.ndarray) -> int:
    """Length of the longest consecutive True run in a boolean array."""
    if mask.size == 0:
        return 0
    m = mask.astype(int)
    diff = np.diff(m)
    starts = np.where(np.concatenate([[m[0] == 1], diff == 1]))[0]
    ends   = np.where(np.concatenate([diff == -1, [m[-1] == 1]]))[0]
    if starts.size == 0 or ends.size == 0:
        return int(mask.sum()) if mask.any() else 0
    return int((ends - starts + 1).max())

def prep_series_for_cwt(x: np.ndarray) -> np.ndarray:
    """Zero-mean and unit-variance normalize, then replace NaNs/Infs with 0."""
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    if not np.isfinite(mu):
        mu = 0.0
    x = x - mu
    x[~np.isfinite(x)] = 0.0
    s = np.nanstd(x)
    if not np.isfinite(s) or s == 0:
        return np.zeros_like(x)
    return x / s

def coi_mask(scales: np.ndarray, nT: int, k: float = np.sqrt(2.0)) -> np.ndarray:
    """
    Cone-of-influence mask (simple approximation):
    zeros out a half-width proportional to scale near both edges.
    """
    mask = np.ones((len(scales), nT), dtype=bool)
    for i, sc in enumerate(scales):
        hw = int(np.ceil(k * sc))
        if hw > 0:
            hw = min(hw, nT // 2)
            mask[i, :hw] = False
            mask[i, nT - hw:] = False
    return mask


# ───────────────────────── CWT core & features ───────────────────────── #

def build_scales(period_min_bars: float, period_max_bars: float, num_scales: int,
                 wavelet: str, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build wavelet scales from target periods (in bars).
    central_frequency * period / dt → scale.
    """
    periods = np.geomspace(period_min_bars, period_max_bars, num=num_scales)
    cf = pywt.central_frequency(wavelet)
    scales = (cf * periods) / dt
    return scales, periods

def cwt_power(x: np.ndarray, scales: np.ndarray, wavelet: str = "morl", dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute CWT coefficients and return power and implied periods (bars)."""
    coeffs, freqs = pywt.cwt(x, scales, wavelet, sampling_period=dt)
    power = np.abs(coeffs) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        periods = 1.0 / freqs
    return power, periods

def spectral_flatness(P: np.ndarray, eps: float = 1e-12) -> float:
    """Geometric mean / arithmetic mean of positive power values."""
    P = P[P > 0]
    if P.size == 0:
        return np.nan
    return float(np.exp(np.mean(np.log(P + eps))) / (np.mean(P) + eps))

def temporal_compactness_gini(energy_t: np.ndarray) -> float:
    """
    Gini-like measure over time for energy concentration.
    1 means highly concentrated in a small portion of time; 0 means uniform.
    """
    x = np.asarray(energy_t, dtype=float)
    x[x < 0] = 0.0
    s = x.sum()
    if s <= 0 or x.size == 0:
        return np.nan
    x = np.sort(x) / s
    cum = np.cumsum(x)
    return float(1.0 - 2.0 * np.mean(cum))

def band_masks_by_period(periods: np.ndarray, bands_cfg: dict) -> Dict[str, np.ndarray]:
    """Boolean masks for low/mid/high bands based on period (bars) thresholds."""
    lo_min = float(bands_cfg.get("lo_min_period_bars", 60))
    mid_min = float(bands_cfg.get("mid_min_period_bars", 15))
    mid_max = float(bands_cfg.get("mid_max_period_bars", 60))
    hi_max = float(bands_cfg.get("hi_max_period_bars", 15))
    lo  = periods > lo_min
    mid = (periods >= mid_min) & (periods <= mid_max)
    hi  = periods < hi_max
    return {"lo": lo, "mid": mid, "hi": hi}

def cwt_features_for_series(s: pd.Series,
                            wave_name: str,
                            period_min: float,
                            period_max: float,
                            num_scales: int,
                            bands_cfg: dict) -> Dict[str, float]:
    """
    Compute CWT-based features for a single series inside one window:
      - band energy fractions (lo/mid/hi)
      - spectral centroid (period in bars)
      - dominant period and its energy ratio
      - temporal compactness (Gini-like)
      - N (effective bars)
    """
    x_raw = s.to_numpy(dtype=float)
    x = prep_series_for_cwt(x_raw)
    N = x.size
    if N < 16 or np.all(x == 0.0):
        return {}
    scales, _ = build_scales(period_min, period_max, num_scales, wave_name, dt=1.0)
    P, periods = cwt_power(x, scales, wave_name, dt=1.0)
    valid = coi_mask(scales, N, k=np.sqrt(2.0))
    P = np.where(valid, P, 0.0)
    tot = float(P.sum())
    if not np.isfinite(tot) or tot <= 0:
        return {}

    masks = band_masks_by_period(periods, bands_cfg)
    e_lo  = float(P[masks["lo"], :].sum())
    e_mid = float(P[masks["mid"], :].sum())
    e_hi  = float(P[masks["hi"], :].sum())

    pct_lo  = e_lo / tot
    pct_mid = e_mid / tot
    pct_hi  = e_hi / tot

    with np.errstate(divide="ignore", invalid="ignore"):
        freqs = 1.0 / periods
    f_all = np.repeat(freqs[:, None], N, axis=1)
    centroid_f = float((f_all * P).sum() / tot) if tot > 0 else np.nan
    centroid_period = (1.0 / centroid_f) if np.isfinite(centroid_f) and centroid_f > 0 else np.nan

    k = np.unravel_index(int(np.argmax(P)), P.shape)
    dom_period = float(periods[k[0]])
    dom_ratio  = float(P[k] / tot)

    compactness = temporal_compactness_gini(P.sum(axis=0))

    return {
        "CWT_power_lo": pct_lo,
        "CWT_power_mid": pct_mid,
        "CWT_power_hi": pct_hi,
        "CWT_centroid_period_bars": centroid_period,
        "CWT_peak_period_bars": dom_period,
        "CWT_peak_ratio": dom_ratio,
        "CWT_temporal_compactness": compactness,
        "CWT_N": int(N),
    }


# ──────────────────────── per-ticker main ───────────────────────── #

def process_one_ticker(ticker: str,
                       merged_path: str,
                       features_root: str,
                       sampling: str,
                       series_list: List[str],
                       wave_cfg: dict,
                       bands_cfg: dict,
                       qual_cfg: dict,
                       report_cads: List[str]):
    """Compute full-window CWT features for one ticker and write CSVs per cadence."""
    df = read_intermediate_csv(merged_path)
    df = slice_rth_if_needed(df, sampling)

    bars_per_day = expected_bars_per_rth_day(sampling)

    # Choose target series; default to r_close. If absent, derive it from close.
    keep = [c for c in (series_list or ["r_close"]) if c in df.columns]
    if not keep:
        if "close" in df.columns:
            close = pd.to_numeric(df["close"], errors="coerce")
            rc = np.log(close) - np.log(close.shift(1))
            if len(rc) > 0:
                rc.iloc[0] = 0.0
            df["r_close"] = rc
            keep = ["r_close"]
        else:
            print(f"[{ticker}] WARN: no target series found; skipping.")
            return

    # Select target columns first, then add _date (prevents KeyError: '_date').
    df = df[keep].copy()
    df["_date"] = pd.to_datetime(df.index.tz_convert(NY).date)

    # CWT params
    wave_name = (wave_cfg.get("name") or "morl")
    period_min = float(wave_cfg.get("period_min_bars", 5.0))
    period_max = float(wave_cfg.get("period_max_bars", 120.0))
    num_scales = int(wave_cfg.get("num_scales", 36))

    # Quality thresholds for determining if a window is usable
    cover_ratio_thresh = float(qual_cfg.get("cover_ratio_thresh", 0.70))
    gap_ratio_thresh   = float(qual_cfg.get("gap_ratio_thresh", 0.20))
    max_gap_run_thresh = int(qual_cfg.get("max_gap_run_thresh", 30))
    min_bars_ratio     = float(qual_cfg.get("min_bars_ratio", 0.75))

    out_dir = os.path.join(features_root, ticker, f"cwt_{sampling}")
    ensure_dir(out_dir)

    for cad in report_cads:
        rows = []
        for (win_start, win_end, sub) in build_windows(df, cad):
            n_days = sub["_date"].nunique()
            expected = int(n_days * bars_per_day)

            row_meta = {
                "ticker": ticker,
                "window_cadence": cad,
                "win_start": win_start,
                "win_end": win_end,
                "days": int(n_days),
                "expected_bars": expected,
            }

            for col in keep:
                s = pd.to_numeric(sub[col], errors="coerce")
                n_valid = int(np.isfinite(s).sum())
                gap_ratio = float(1.0 - (n_valid / expected)) if expected > 0 else 1.0
                max_gap_run = int(longest_true_run(~np.isfinite(s.values))) if s.size else 0
                min_ratio_ok = (n_valid / expected) >= min_bars_ratio if expected > 0 else False
                gap_ok = (gap_ratio <= gap_ratio_thresh) and (max_gap_run <= max_gap_run_thresh)
                prefix = f"{col}"

                if not (min_ratio_ok and gap_ok) or n_valid < 16:
                    # Not enough quality — write NaNs for spectral features but still record QC.
                    row_meta.update({
                        f"{prefix}_CWT_power_lo": np.nan,
                        f"{prefix}_CWT_power_mid": np.nan,
                        f"{prefix}_CWT_power_hi": np.nan,
                        f"{prefix}_CWT_centroid_period_bars": np.nan,
                        f"{prefix}_CWT_peak_period_bars": np.nan,
                        f"{prefix}_CWT_peak_ratio": np.nan,
                        f"{prefix}_CWT_temporal_compactness": np.nan,
                        f"{prefix}_CWT_N": int(n_valid),
                        f"{prefix}_CWT_gap_ratio": float(gap_ratio),
                        f"{prefix}_CWT_max_gap_run": int(max_gap_run),
                    })
                    continue

                feats = cwt_features_for_series(s, wave_name, period_min, period_max, num_scales, bands_cfg)
                if feats:
                    row_meta.update({
                        f"{prefix}_CWT_power_lo": feats["CWT_power_lo"],
                        f"{prefix}_CWT_power_mid": feats["CWT_power_mid"],
                        f"{prefix}_CWT_power_hi": feats["CWT_power_hi"],
                        f"{prefix}_CWT_centroid_period_bars": feats["CWT_centroid_period_bars"],
                        f"{prefix}_CWT_peak_period_bars": feats["CWT_peak_period_bars"],
                        f"{prefix}_CWT_peak_ratio": feats["CWT_peak_ratio"],
                        f"{prefix}_CWT_temporal_compactness": feats["CWT_temporal_compactness"],
                        f"{prefix}_CWT_N": feats["CWT_N"],
                        f"{prefix}_CWT_gap_ratio": float(gap_ratio),
                        f"{prefix}_CWT_max_gap_run": int(max_gap_run),
                    })
                else:
                    row_meta.update({
                        f"{prefix}_CWT_power_lo": np.nan,
                        f"{prefix}_CWT_power_mid": np.nan,
                        f"{prefix}_CWT_power_hi": np.nan,
                        f"{prefix}_CWT_centroid_period_bars": np.nan,
                        f"{prefix}_CWT_peak_period_bars": np.nan,
                        f"{prefix}_CWT_peak_ratio": np.nan,
                        f"{prefix}_CWT_temporal_compactness": np.nan,
                        f"{prefix}_CWT_N": int(n_valid),
                        f"{prefix}_CWT_gap_ratio": float(gap_ratio),
                        f"{prefix}_CWT_max_gap_run": int(max_gap_run),
                    })

            rows.append(row_meta)

        out_path = os.path.join(out_dir, f"{cad}.csv")
        pd.DataFrame(rows).sort_values(["win_end"]).to_csv(out_path, index=False)
        print(f"[{ticker}] {cad} → {out_path} ({len(rows)} rows)")


# ─────────────────────────────── main ────────────────────────────── #

def main():
    ap = argparse.ArgumentParser(description="Full-window CWT on merged features (FFT-like reporting).")
    ap.add_argument("--config",
                    default=r"D:\Data mining project\data_mining_fall_2025\feature_extraction\configs\data_preprocess_config.yaml",
                    help="Path to YAML config.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cwt_cfg = cfg.get("cwt", {}) or {}
    if not cwt_cfg.get("enabled", True):
        print("CWT disabled in YAML. Exit.")
        return

    sampling = derive_sampling(cfg)

    merged_tpl = (cwt_cfg.get("inputs", {}) or {}).get("merged_dir")
    if not merged_tpl:
        merged_tpl = os.path.join(cfg.get("output_dir", "output"), f"intermediate_dataset_{sampling}")
    merged_dir = fill_path_template(merged_tpl, sampling)

    features_root = (cwt_cfg.get("outputs", {}) or {}).get("features_root") \
                    or os.path.join(cfg.get("output_dir", "output"), "features")

    series_list = list(cwt_cfg.get("series", []) or ["r_close"])
    wave_cfg = cwt_cfg.get("wavelet", {}) or {}
    bands_cfg = cwt_cfg.get("bands", {}) or {}
    qual_cfg  = cwt_cfg.get("quality", {}) or {}
    report_cads = list((cwt_cfg.get("reporting", {}) or {}).get("cadences", []) or ["week"])

    allowed = set(allowed_reporting_for(sampling))
    bad = [c for c in report_cads if c not in allowed]
    if bad:
        raise SystemExit(
            f"cwt.reporting.cadences invalid {bad}; "
            f"allowed for sampling={sampling}: {sorted(allowed)}"
        )

    print(f"Sampling cadence: {sampling}")
    print(f"Merged input: {merged_dir}")
    print(f"Features root: {features_root}")
    print(f"Reporting: {report_cads} (full-window CWT)")

    files = list_tickers(merged_dir)
    if not files:
        raise SystemExit(f"No CSV found under {merged_dir}")

    for tic, path in sorted(files.items()):
        try:
            process_one_ticker(
                ticker=tic,
                merged_path=path,
                features_root=features_root,
                sampling=sampling,
                series_list=series_list,
                wave_cfg=wave_cfg,
                bands_cfg=bands_cfg,
                qual_cfg=qual_cfg,
                report_cads=report_cads
            )
        except Exception as e:
            print(f"[{tic}] ERROR: {e}")

if __name__ == "__main__":
    main()
