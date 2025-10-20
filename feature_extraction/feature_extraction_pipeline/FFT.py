#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FFT (full-window mode)
──────────────────────────────────────────────────────────────────────────────
Given the sampling granularity and reporting cadences from YAML, this script
concatenates ALL bars within a reporting window into one long series and runs
Welch’s method once to produce frequency-domain features for that window
(no per-day FFT aggregation step).

Typical usage:
  - granularity=minute, reporting.cadences=["week"]
      → For each week, concatenate 5 RTH trading days (09:30–16:00 ET) of minute bars.
  - granularity=hour,   reporting.cadences=["month"]
      → For each month, concatenate RTH hourly bars.

Dependencies:
  pip install numpy pandas pyyaml scipy

Run:
  python FFT.py --config "D:\\...\\configs\\data_preprocess_config.yaml"

Output directory layout:
  <features_root>/<TICKER>/fft_<sampling>/<cadence>.csv
    e.g., features/AAPL/fft_minute/week.csv
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
    from zoneinfo import ZoneInfo
    NY = ZoneInfo("America/New_York")
except Exception:
    import pytz
    NY = pytz.timezone("America/New_York")

# SciPy
try:
    from scipy.signal import welch
except Exception as e:
    print("❌ SciPy is required: pip install scipy", file=sys.stderr)
    raise

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
    # Only windows >= sampling are supported (no "hour" windows).
    if sampling in ("minute", "hour"):
        return ["day", "week", "month", "quarter", "year"]
    if sampling == "day":
        return ["week", "month", "quarter", "year"]
    return []

def cadence_to_pd_freq(cad: str) -> str:
    return {"day": "D", "week": "W-FRI", "month": "M", "quarter": "Q", "year": "Y"}[cad]

def expected_bars_per_rth_day(sampling: str) -> int:
    if sampling == "minute":
        return 390  # 09:30..15:59
    if sampling == "hour":
        return 6    # 09:30,10:30,11:30,12:30,13:30,14:30 (default drops 15:30 half-hour)
    if sampling == "day":
        return 1
    raise ValueError(f"unsupported sampling {sampling}")

def fill_path_template(path_or_tpl: Optional[str], cad: str) -> Optional[str]:
    return path_or_tpl.replace("{cad}", cad) if path_or_tpl else path_or_tpl

# ─────────────────────────── IO helpers ─────────────────────────── #

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def read_intermediate_csv(path: str, tz: str = "America/New_York") -> pd.DataFrame:
    """Read a per-ticker merged CSV and return a tz-aware ET-indexed DataFrame."""
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"'timestamp' not in {path}")
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["timestamp"] = ts.dt.tz_convert(NY)
    df = df.set_index("timestamp").sort_index()
    return df

def list_tickers(merged_dir: str) -> Dict[str, str]:
    """Return {TICKER: path_to_csv} for all CSVs in merged_dir."""
    return {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(merged_dir, "*.csv"))}

# ──────────────────────── quality & utils ───────────────────────── #

def longest_true_run(mask: np.ndarray) -> int:
    """Length of the longest consecutive True run in a boolean array."""
    if mask.size == 0:
        return 0
    diff = np.diff(mask.astype(int))
    starts = np.where(np.concatenate([[mask[0]], diff == 1]))[0]
    ends   = np.where(np.concatenate([diff == -1, [mask[-1]]]))[0]
    if len(starts) == 0 or len(ends) == 0:
        # Either no True at all or all True
        return int(mask.sum()) if mask.any() else 0
    run_lengths = ends - starts + 1
    return int(run_lengths.max()) if run_lengths.size else 0

def clamp_welch_params(N: int, nperseg: int, noverlap: int) -> Tuple[int, int]:
    """Clamp Welch parameters by series length N to avoid errors."""
    if N < 4:
        return max(N, 1), 0
    nseg = min(max(nperseg, 1), N)
    nov  = min(max(noverlap, 0), nseg - 1)
    return nseg, nov

def compute_band_edges_from_cfg(bands_cfg: dict) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """
    Convert period (bars) thresholds to frequency (cycles/bar) thresholds:
      Low  : P > lo_min                 → f ∈ (0, 1/lo_min)
      Mid  : mid_min ≤ P ≤ mid_max      → f ∈ [1/mid_max, 1/mid_min]
      High : P < hi_max                 → f ∈ (1/hi_max, 0.5]   (Nyquist = 0.5 for fs = 1 bar^-1)
    Returns: {"lo": (fmin, fmax), "mid": (...), "hi": (...)}; None means open-ended on that side.
    """
    lo_min = float(bands_cfg.get("lo_min_period_bars", 60))
    mid_min = float(bands_cfg.get("mid_min_period_bars", 15))
    mid_max = float(bands_cfg.get("mid_max_period_bars", 60))
    hi_max = float(bands_cfg.get("hi_max_period_bars", 15))
    bands = {
        "lo":  (1.0/1e12, 1.0/lo_min),     # effectively (0, 1/lo_min)
        "mid": (1.0/mid_max, 1.0/mid_min), # [1/mid_max, 1/mid_min]
        "hi":  (1.0/hi_max, 0.5),          # (1/hi_max, Nyquist]
    }
    return bands

def slice_by_session(df: pd.DataFrame, sampling: str) -> pd.DataFrame:
    """
    For intraday (minute/hour), keep only RTH rows. If the merged CSV
    already contains an RTH-only subset with a 'session' column, this is a no-op.
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

# ───────────────────── Welch feature extraction ──────────────────── #

def welch_features(x: np.ndarray,
                   nperseg: int,
                   noverlap: int,
                   bands: Dict[str, Tuple[Optional[float], Optional[float]]]
                   ) -> Dict[str, float]:
    """
    Run Welch on a single 1D series and return:
      - total power
      - band powers (lo/mid/hi) and their fractions
      - dominant frequency & period
      - spectral centroid frequency & period
      - FFT_N (effective sample count)
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return {}
    nseg, nov = clamp_welch_params(x.size, nperseg, noverlap)
    f, Pxx = welch(x - np.nanmean(x), fs=1.0, window="hann",
                   nperseg=nseg, noverlap=nov, detrend="constant",
                   return_onesided=True, scaling="density")
    valid = f > 0
    f = f[valid]
    Pxx = Pxx[valid]
    if f.size == 0:
        return {}

    total = float(np.trapz(Pxx, f))
    out = {"power_total": total}

    def band_power(fmin: Optional[float], fmax: Optional[float]) -> float:
        mask = np.ones_like(f, dtype=bool)
        if fmin is not None:
            mask &= (f >= fmin)
        if fmax is not None:
            mask &= (f <= fmax)
        if not mask.any():
            return 0.0
        return float(np.trapz(Pxx[mask], f[mask]))

    plo = band_power(*bands["lo"])
    pmid = band_power(*bands["mid"])
    phi = band_power(*bands["hi"])

    out.update({
        "power_lo": plo,
        "power_mid": pmid,
        "power_hi": phi,
        "pct_lo": (plo/total) if total > 0 else 0.0,
        "pct_mid": (pmid/total) if total > 0 else 0.0,
        "pct_hi": (phi/total) if total > 0 else 0.0,
    })

    k = int(np.argmax(Pxx))
    dom_f = float(f[k])
    out["dom_freq_cpb"] = dom_f
    out["dom_period_bars"] = (1.0/dom_f) if dom_f > 0 else np.nan

    centroid = float(np.sum(f * Pxx) / np.sum(Pxx)) if np.sum(Pxx) > 0 else 0.0
    out["centroid_freq_cpb"] = centroid
    out["centroid_period_bars"] = (1.0/centroid) if centroid > 0 else np.nan
    out["FFT_N"] = int(x.size)
    return out

# ──────────────────────── window building ───────────────────────── #

def build_windows(df: pd.DataFrame, cadence: str) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]]:
    """
    Split df into windows by the given cadence (day/week/month/quarter/year).
    Returns a list of (win_start, win_end, df_window).
    """
    freq = cadence_to_pd_freq(cadence)
    grouped = df.groupby(pd.Grouper(freq=freq, label="right", closed="right"))
    windows = []
    for label, sub in grouped:
        if sub.empty:
            continue
        win_end = pd.Timestamp(label).tz_localize(NY) if label.tz is None else label.tz_convert(NY)
        win_start = sub.index.min()
        windows.append((win_start, win_end, sub))
    return windows

# ──────────────────────── main per-ticker ───────────────────────── #

def process_one_ticker(ticker: str,
                       merged_path: str,
                       features_root: str,
                       sampling: str,
                       series_list: List[str],
                       welch_cfg: dict,
                       bands_cfg: dict,
                       qual_cfg: dict,
                       report_cads: List[str]):
    """Compute full-window Welch features for one ticker and write CSVs per cadence."""
    df = read_intermediate_csv(merged_path, tz="America/New_York")
    df = slice_by_session(df, sampling)

    keep = [c for c in series_list if c in df.columns]
    if not keep:
        print(f"[{ticker}] WARN: none of the requested series found; skipping.")
        return
    df = df[keep].copy()

    df["_date"] = pd.to_datetime(df.index.tz_convert(NY).date)

    bands = compute_band_edges_from_cfg(bands_cfg)

    nperseg = int(welch_cfg.get("nperseg", 128))
    noverlap = int(welch_cfg.get("noverlap", nperseg // 2))

    gap_ratio_thresh = float(qual_cfg.get("gap_ratio_thresh", 0.10))
    max_gap_run_thresh = int(qual_cfg.get("max_gap_run_thresh", 5))
    min_bars_ratio = float(qual_cfg.get("min_bars_ratio", 0.75))

    out_dir = os.path.join(features_root, ticker, f"fft_{sampling}")
    ensure_dir(out_dir)

    bars_per_day = expected_bars_per_rth_day(sampling)

    for cad in report_cads:
        rows = []
        windows = build_windows(df, cad)
        for (win_start, win_end, sub) in windows:
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
                s = sub[col].astype(float)
                n_valid = int(np.isfinite(s).sum())
                gap_ratio = float(1.0 - (n_valid / expected)) if expected > 0 else 1.0

                nan_mask = ~np.isfinite(s.values)
                max_gap_run = int(longest_true_run(nan_mask)) if nan_mask.size else 0

                min_ratio_ok = (n_valid / expected) >= min_bars_ratio if expected > 0 else False
                gap_ok = (gap_ratio <= gap_ratio_thresh) and (max_gap_run <= max_gap_run_thresh)

                prefix = f"{col}"
                if not (min_ratio_ok and gap_ok) or n_valid < 4:
                    # Not enough data quality — write NaNs for spectral features, still record QC
                    row_meta.update({
                        f"{prefix}_power_total": np.nan,
                        f"{prefix}_power_lo": np.nan,
                        f"{prefix}_power_mid": np.nan,
                        f"{prefix}_power_hi": np.nan,
                        f"{prefix}_pct_lo": np.nan,
                        f"{prefix}_pct_mid": np.nan,
                        f"{prefix}_pct_hi": np.nan,
                        f"{prefix}_dom_freq_cpb": np.nan,
                        f"{prefix}_dom_period_bars": np.nan,
                        f"{prefix}_centroid_freq_cpb": np.nan,
                        f"{prefix}_centroid_period_bars": np.nan,
                        f"{prefix}_FFT_N": int(n_valid),
                        f"{prefix}_gap_ratio": float(gap_ratio),
                        f"{prefix}_max_gap_run": int(max_gap_run),
                    })
                    continue

                feats = welch_features(s.values[np.isfinite(s.values)],
                                       nperseg=nperseg, noverlap=noverlap,
                                       bands=bands)
                if feats:
                    row_meta.update({
                        f"{prefix}_power_total": feats["power_total"],
                        f"{prefix}_power_lo": feats["power_lo"],
                        f"{prefix}_power_mid": feats["power_mid"],
                        f"{prefix}_power_hi": feats["power_hi"],
                        f"{prefix}_pct_lo": feats["pct_lo"],
                        f"{prefix}_pct_mid": feats["pct_mid"],
                        f"{prefix}_pct_hi": feats["pct_hi"],
                        f"{prefix}_dom_freq_cpb": feats["dom_freq_cpb"],
                        f"{prefix}_dom_period_bars": feats["dom_period_bars"],
                        f"{prefix}_centroid_freq_cpb": feats["centroid_freq_cpb"],
                        f"{prefix}_centroid_period_bars": feats["centroid_period_bars"],
                        f"{prefix}_FFT_N": feats["FFT_N"],
                        f"{prefix}_gap_ratio": float(gap_ratio),
                        f"{prefix}_max_gap_run": int(max_gap_run),
                    })
                else:
                    row_meta.update({
                        f"{prefix}_power_total": np.nan,
                        f"{prefix}_power_lo": np.nan,
                        f"{prefix}_power_mid": np.nan,
                        f"{prefix}_power_hi": np.nan,
                        f"{prefix}_pct_lo": np.nan,
                        f"{prefix}_pct_mid": np.nan,
                        f"{prefix}_pct_hi": np.nan,
                        f"{prefix}_dom_freq_cpb": np.nan,
                        f"{prefix}_dom_period_bars": np.nan,
                        f"{prefix}_centroid_freq_cpb": np.nan,
                        f"{prefix}_centroid_period_bars": np.nan,
                        f"{prefix}_FFT_N": int(n_valid),
                        f"{prefix}_gap_ratio": float(gap_ratio),
                        f"{prefix}_max_gap_run": int(max_gap_run),
                    })

            rows.append(row_meta)

        out_path = os.path.join(out_dir, f"{cad}.csv")
        pd.DataFrame(rows).sort_values(["win_end"]).to_csv(out_path, index=False)
        print(f"[{ticker}] {cad} → {out_path} ({len(rows)} rows)")

# ─────────────────────────────── main ────────────────────────────── #

def main():
    ap = argparse.ArgumentParser(description="Full-window FFT on merged intraday/daily features.")
    ap.add_argument("--config", default=r"D:\Data mining project\data_mining_fall_2025\feature_extraction\configs\data_preprocess_config.yaml",
                    help="Path to YAML config.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    fft_cfg = cfg.get("fft", {}) or {}
    if not fft_cfg.get("enabled", True):
        print("FFT disabled in YAML. Exit.")
        return

    sampling = derive_sampling(cfg)  # minute|hour|day

    merged_dir_tpl = (fft_cfg.get("inputs", {}) or {}).get("merged_dir")
    if not merged_dir_tpl:
        merged_dir_tpl = os.path.join(cfg.get("output_dir", "output"), f"intermediate_dataset_{sampling}")
    merged_dir = fill_path_template(merged_dir_tpl, sampling)

    features_root = (fft_cfg.get("outputs", {}) or {}).get("features_root") \
                    or os.path.join(cfg.get("output_dir", "output"), "features")

    series_list = list(fft_cfg.get("series", []) or [])
    if not series_list:
        raise SystemExit("fft.series is empty: configure the column names to FFT in the YAML.")

    welch_cfg = (fft_cfg.get("welch", {}) or {})
    bands_cfg = (fft_cfg.get("bands", {}) or {})
    qual_cfg  = (fft_cfg.get("quality", {}) or {})
    report_cads = list((fft_cfg.get("reporting", {}) or {}).get("cadences", []) or [])
    if not report_cads:
        report_cads = ["week"]

    allowed = set(allowed_reporting_for(sampling))
    bad = [c for c in report_cads if c not in allowed]
    if bad:
        raise SystemExit(f"reporting.cadences has invalid levels {bad} (sampling={sampling}, allowed={sorted(allowed)})")

    print(f"Sampling cadence: {sampling}")
    print(f"Merged input dir: {merged_dir}")
    print(f"Features root: {features_root}")
    print(f"Reporting cadences: {report_cads} | reducer=full_window (no per-day aggregation)")

    files = list_tickers(merged_dir)
    if not files:
        raise SystemExit(f"No CSV files found in {merged_dir}.")

    for tic, path in sorted(files.items()):
        try:
            process_one_ticker(
                ticker=tic,
                merged_path=path,
                features_root=features_root,
                sampling=sampling,
                series_list=series_list,
                welch_cfg=welch_cfg,
                bands_cfg=bands_cfg,
                qual_cfg=qual_cfg,
                report_cads=report_cads
            )
        except Exception as e:
            print(f"[{tic}] ERROR: {e}")

if __name__ == "__main__":
    main()
