#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGG (per-day then aggregate) — FFT/CWT-style integration
────────────────────────────────────────────────────────
- Sampling cadence comes from YAML: date_range.granularity (minute|hour|day)
- Reads merged inputs under merge.outputs.merged_dir (supports {cad})
- Builds per-day intraday features (coverage-aware), then aggregates them
  into reporting windows (day/week/month/quarter/year) using a reducer.

Outputs:
  <features_root>/<TICKER>/agg_<sampling>/<cadence>.csv

YAML (add an `agg` section):
agg:
  enabled: true
  inputs:
    merged_dir: "D:\\...\\intermediate_dataset_{cad}"
  outputs:
    features_root: "D:\\...\\features"
  reporting:
    cadences: ["day","week"]          # default: ["day"]
    reducer: "median"                  # "median" | "mean" | "weighted_mean"
    weighting: "by_valid_bars"         # used when reducer = "weighted_mean"
  quality:
    min_valid_ratio_open15: 0.50
    min_valid_ratio_last10: 0.50
    min_valid_ratio_reversal: 0.50
    min_valid_count_run: 5
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


# ───────────────────────── YAML / cadence helpers ───────────────────────── #

def load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}

def normalize_granularity(ts: str) -> str:
    m = {
        "minute": "minute", "min": "minute", "m": "minute",
        "hour": "hour", "hr": "hour", "h": "hour",
        "day": "day", "daily": "day", "d": "day"
    }
    out = m.get((ts or "").strip().lower())
    if not out:
        raise SystemExit("date_range.granularity must be one of: minute|hour|day")
    return out

def derive_sampling(cfg: dict) -> str:
    return normalize_granularity(cfg.get("date_range", {}).get("granularity", "minute"))

def allowed_reporting_for(sampling: str) -> List[str]:
    if sampling == "minute":
        return ["day", "week", "month", "quarter", "year"]
    if sampling == "hour":
        return ["day", "week", "month", "quarter", "year"]
    if sampling == "day":
        return ["week", "month", "quarter", "year"]
    return []

def cadence_to_pd_freq(cad: str) -> str:
    return {"day": "D", "week": "W-FRI", "month": "M", "quarter": "Q", "year": "Y"}[cad]

def expected_bars_per_rth_day(sampling: str) -> int:
    if sampling == "minute": return 390
    if sampling == "hour":   return 6
    if sampling == "day":    return 1
    raise ValueError(f"unsupported sampling {sampling}")

def fill_path_template(p: Optional[str], cad: str) -> Optional[str]:
    return None if not p else p.replace("{cad}", cad)


# ───────────────────────── IO & time helpers ───────────────────────── #

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def list_tickers(merged_dir: str) -> Dict[str, str]:
    return {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(merged_dir, "*.csv"))}

def read_intermediate_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"'timestamp' not in {path}")

    # Key: normalize to UTC first to avoid mixed-offset object dtype issues.
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    # Then convert to New York time (DST handled automatically).
    ts = ts.dt.tz_convert(NY)

    df["timestamp"] = ts
    df = df.set_index("timestamp").sort_index()
    return df


def slice_rth(df: pd.DataFrame) -> pd.DataFrame:
    if "session" in df.columns:
        return df.loc[df["session"] == "rth"]
    t = df.index.tz_convert(NY)
    start = pd.Timestamp("09:30").time()
    end   = pd.Timestamp("16:00").time()
    mask = pd.Series([(start <= hh < end) for hh in t.time], index=df.index)
    return df.loc[mask]

def et_day_bounds(day: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    d = pd.Timestamp(day)
    d = d.tz_localize(NY) if d.tz is None else d.tz_convert(NY)
    start = d.replace(hour=9, minute=30, second=0, microsecond=0)
    end   = d.replace(hour=16, minute=0, second=0, microsecond=0)
    return start, end

def minutes_since_open_from_index(idx: pd.DatetimeIndex) -> np.ndarray:
    """
    Compute minutes since 09:30 ET from timestamps (works for minute/hour bars).
    Returns int array, unclipped (may skip values for hour bars).
    """
    out = np.empty(len(idx), dtype=np.int32)
    for i, ts in enumerate(idx.tz_convert(NY)):
        d = pd.Timestamp(ts.date()).tz_localize(NY)
        start = d.replace(hour=9, minute=30, second=0, microsecond=0)
        out[i] = int((ts - start).total_seconds() // 60)
    return out


# ───────────────────────── per-day feature helpers ───────────────────────── #

def gap_quality_metrics(day_df: pd.DataFrame) -> Tuple[float, int]:
    if "is_gap" not in day_df.columns:
        return 0.0, 0
    g = day_df["is_gap"].astype(int).to_numpy()
    gap_ratio = float(np.mean(g)) if g.size else 0.0
    max_run, cur = 0, 0
    for v in g:
        if v == 1:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return gap_ratio, int(max_run)

def longest_same_sign_run(r: np.ndarray) -> int:
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
    if x.size != y.size or x.size != mask.size:
        return np.nan
    x = x[mask]; y = y[mask]
    if x.size < 2:
        return np.nan
    x = x - np.nanmean(x); y = y - np.nanmean(y)
    sx = np.nanstd(x); sy = np.nanstd(y)
    if not np.isfinite(sx) or not np.isfinite(sy) or sx == 0 or sy == 0:
        return np.nan
    return float(np.nanmean((x / sx) * (y / sy)))

def window_sum_with_gating(r: np.ndarray, mask: np.ndarray, min_valid_ratio: float):
    if mask.sum() == 0:
        return np.nan, 0.0
    r_w = r[mask]
    valid_ratio = float(np.isfinite(r_w).mean()) if r_w.size else 0.0
    if r_w.size and valid_ratio >= min_valid_ratio:
        return float(np.nansum(r_w)), valid_ratio
    else:
        return np.nan, valid_ratio

def corr_with_gating(tod: np.ndarray, r: np.ndarray, min_valid_ratio: float):
    valid = np.isfinite(r) & np.isfinite(tod)
    valid_ratio = float(valid.mean()) if valid.size else 0.0
    if valid_ratio < min_valid_ratio:
        return np.nan, valid_ratio
    r_fill0 = np.where(np.isfinite(r), r, 0.0)
    cumret = np.cumsum(r_fill0)
    val = pearson_corr_masked(tod, cumret, valid)
    return val, valid_ratio


def build_day_features(g: pd.DataFrame,
                       sampling: str,
                       min_valid_ratio_open15: float,
                       min_valid_ratio_last10: float,
                       min_valid_ratio_reversal: float,
                       min_valid_count_run: int,
                       emit_validity_cols: bool) -> Dict[str, float]:
    """Compute one trading day's features from intraday rows g (already RTH, one ticker)."""
    g = g.sort_index()
    # r_close — build if missing
    if "r_close" not in g.columns:
        close = pd.to_numeric(g["close"], errors="coerce")
        rc = np.log(close) - np.log(close.shift(1))
        if len(rc) > 0:
            rc.iloc[0] = 0.0
        g["r_close"] = rc

    # TOD (minutes since open) — derive if absent
    if "minutes_since_open" in g.columns and pd.api.types.is_numeric_dtype(g["minutes_since_open"]):
        tod = g["minutes_since_open"].to_numpy(dtype=float)
    else:
        tod = minutes_since_open_from_index(g.index).astype(float)

    # Inputs
    r   = g["r_close"].to_numpy(dtype=float)
    svn = g["spread_vwap_norm"].to_numpy(dtype=float) if "spread_vwap_norm" in g.columns else np.full(len(g), np.nan)
    vz  = g["volume_z_by_tod"].to_numpy(dtype=float)  if "volume_z_by_tod" in g.columns else np.full(len(g), np.nan)
    vol = g["volume"].to_numpy(dtype=float)           if "volume" in g.columns else np.full(len(g), np.nan)

    # Basic quality
    gap_ratio, max_gap_run = gap_quality_metrics(g)
    valid_count_day = int(np.isfinite(r).sum())
    valid_ratio_day = float(np.isfinite(r).mean()) if r.size else np.nan

    # Full-day aggregates
    ret_oc = float(np.nansum(r)) if np.isfinite(r).any() else np.nan
    rv     = float(np.nansum(np.square(r))) if np.isfinite(r).any() else np.nan

    # Minute-window masks (only meaningful for minute data; for hour/day they just yield NaN)
    idx_open15 = (tod >= 0) & (tod <= 14)
    idx_last10 = (tod >= 380) & (tod <= 389)

    ret_open15, open15_valid = window_sum_with_gating(r, idx_open15, min_valid_ratio_open15)
    ret_last10, last10_valid = window_sum_with_gating(r, idx_last10, min_valid_ratio_last10)

    # Trend persistence
    if valid_count_day >= min_valid_count_run:
        trend_persistence = longest_same_sign_run(r)
    else:
        trend_persistence = np.nan

    # Reversal score (corr TOD vs cumret) gated by full-day valid ratio
    reversal_score, rev_valid = corr_with_gating(tod, r, min_valid_ratio_reversal)

    # Others
    mabs_svn = float(np.nanmean(np.abs(svn))) if np.isfinite(np.nanmean(np.abs(svn))) else np.nan
    eod_svn  = float(pd.Series(svn).dropna().iloc[-1]) if np.isfinite(svn).any() else np.nan

    vol_total = float(np.nansum(vol)) if np.isfinite(vol).any() else np.nan
    vol_last10 = np.nansum(vol[idx_last10]) if (vol[idx_last10].size and np.isfinite(vol[idx_last10]).any()) else np.nan
    vol_last10_frac = (vol_last10 / vol_total) if (np.isfinite(vol_last10) and np.isfinite(vol_total) and vol_total > 0) else np.nan

    mask_vz = np.isfinite(vz)
    hi2_frac = float(np.mean(vz[mask_vz] > 2)) if mask_vz.any() else np.nan

    row = {
        # quality/meta per-day
        "AGG_gap_ratio": gap_ratio,
        "AGG_max_consecutive_gap": max_gap_run,
        "AGG_N": int(len(g)),
        "valid_bars": valid_count_day,
        "valid_ratio_day": valid_ratio_day,
        # features
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
        })
    return row


# ───────────────────────── windowing & reduction ───────────────────────── #

def build_windows_from_days(daily_df: pd.DataFrame, cadence: str) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]]:
    """
    Group daily table into reporting windows using day_id as the time axis.
    daily_df must contain:
      - 'day_id' (tz-aware ET midnight) and per-day numeric cols.
    Returns list of (win_start, win_end, sub_df_of_days).
    """
    # Use the day_id as index for Grouper
    tmp = daily_df.set_index("day_id").sort_index()
    freq = cadence_to_pd_freq(cadence)
    grouped = tmp.groupby(pd.Grouper(freq=freq, label="right", closed="right"))
    out = []
    for label, sub in grouped:
        if sub.empty:
            continue
        win_end = pd.Timestamp(label).tz_localize(NY) if label.tz is None else label.tz_convert(NY)
        win_start = pd.to_datetime(sub.index.min()).tz_convert(NY)
        out.append((win_start, win_end, sub))
    return out

def reduce_window(sub: pd.DataFrame,
                  reducer: str,
                  weighting: str,
                  bars_per_day: int) -> Dict[str, float]:
    """
    Reduce a bunch of daily rows into one window row.
    Weighted mean uses 'valid_bars' as weights when weighting=='by_valid_bars'.
    """
    out = {}
    n_days = int(sub.shape[0])
    expected_bars = int(n_days * bars_per_day)
    valid_bars = int(np.nansum(sub["valid_bars"].to_numpy(dtype=float))) if "valid_bars" in sub.columns else np.nan
    out.update({
        "days": n_days,
        "expected_bars": expected_bars,
        "valid_bars": valid_bars,
        "cover_ratio": (valid_bars / expected_bars) if (isinstance(valid_bars, (int, float)) and expected_bars > 0) else np.nan
    })

    # numeric feature columns to aggregate (skip obvious meta)
    skip = {"AGG_N","valid_bars","expected_bars","days","cover_ratio"}
    numeric_cols = [c for c in sub.columns if c not in skip and pd.api.types.is_numeric_dtype(sub[c])]

    if reducer == "median":
        agg = sub[numeric_cols].median(numeric_only=True)
    elif reducer == "mean":
        agg = sub[numeric_cols].mean(numeric_only=True)
    elif reducer == "weighted_mean":
        if weighting == "by_valid_bars" and "valid_bars" in sub.columns:
            w = sub["valid_bars"].to_numpy(dtype=float)
            def wmean(v):
                vv = sub[v].to_numpy(dtype=float)
                m = np.isfinite(vv) & np.isfinite(w)
                if not m.any() or np.nansum(w[m]) <= 0:
                    return np.nan
                return float(np.nansum(vv[m] * w[m]) / np.nansum(w[m]))
            agg_vals = {c: wmean(c) for c in numeric_cols}
            agg = pd.Series(agg_vals)
        else:
            agg = sub[numeric_cols].mean(numeric_only=True)
    else:
        raise SystemExit("agg.reporting.reducer must be one of: median|mean|weighted_mean")

    out.update(agg.to_dict())
    return out


# ───────────────────────── per-ticker main ───────────────────────── #

def process_one_ticker(ticker: str,
                       merged_path: str,
                       features_root: str,
                       sampling: str,
                       report_cads: List[str],
                       reducer: str,
                       weighting: str,
                       qual_cfg: dict,
                       emit_validity_cols: bool):
    df = read_intermediate_csv(merged_path)
    df = slice_rth(df)

    # Sanity & required columns (soft — we compute alternatives when missing)
    for c in ["close"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {merged_path}")

    # Derive day_id for grouping
    df["day_id"] = pd.to_datetime(df.index.tz_convert(NY).date).tz_localize(NY)

    # Per-day compute
    bars_per_day = expected_bars_per_rth_day(sampling)
    rows = []
    for d, g in df.groupby(df["day_id"]):
        feats = build_day_features(
            g,
            sampling=sampling,
            min_valid_ratio_open15=float(qual_cfg.get("min_valid_ratio_open15", 0.50)),
            min_valid_ratio_last10=float(qual_cfg.get("min_valid_ratio_last10", 0.50)),
            min_valid_ratio_reversal=float(qual_cfg.get("min_valid_ratio_reversal", 0.50)),
            min_valid_count_run=int(qual_cfg.get("min_valid_count_run", 5)),
            emit_validity_cols=emit_validity_cols
        )
        start_ts, end_ts = et_day_bounds(d)
        rows.append({
            "ticker": ticker,
            "day_id": pd.Timestamp(d).tz_convert(NY),
            "day_start_ts": start_ts,
            "day_end_ts": end_ts,
            "feature_ts": end_ts,
            **feats
        })

    if not rows:
        print(f"[{ticker}] WARN: no daily rows computed; skip.")
        return

    daily = pd.DataFrame(rows).sort_values("day_id")

    # Write daily (always)
    out_dir = os.path.join(features_root, ticker, f"agg_{sampling}")
    ensure_dir(out_dir)
    daily_out = os.path.join(out_dir, "day.csv")
    daily.to_csv(daily_out, index=False)
    print(f"[{ticker}] day → {daily_out} ({len(daily)} rows)")

    # Reduce to other cadences
    for cad in [c for c in report_cads if c != "day"]:
        win_rows = []
        for (win_start, win_end, sub) in build_windows_from_days(daily, cad):
            meta = {
                "ticker": ticker,
                "window_cadence": cad,
                "win_start": win_start,
                "win_end": win_end,
            }
            reduced = reduce_window(sub, reducer=reducer, weighting=weighting, bars_per_day=bars_per_day)
            win_rows.append({**meta, **reduced})
        out_path = os.path.join(out_dir, f"{cad}.csv")
        pd.DataFrame(win_rows).sort_values(["win_end"]).to_csv(out_path, index=False)
        print(f"[{ticker}] {cad} → {out_path} ({len(win_rows)} rows)")


# ─────────────────────────────── main ────────────────────────────── #

def main():
    ap = argparse.ArgumentParser(description="Per-day intraday features → cadence aggregation (FFT/CWT-style).")
    ap.add_argument("--config",
                    default=r"D:\Data mining project\data_mining_fall_2025\feature_extraction\configs\data_preprocess_config.yaml",
                    help="Path to YAML config.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    agg_cfg = cfg.get("agg", {}) or {}
    if not agg_cfg.get("enabled", True):
        print("AGG disabled in YAML. Exit.")
        return

    sampling = derive_sampling(cfg)

    merged_tpl = (agg_cfg.get("inputs", {}) or {}).get("merged_dir")
    if not merged_tpl:
        merged_tpl = os.path.join(cfg.get("output_dir", "output"), f"intermediate_dataset_{sampling}")
    merged_dir = fill_path_template(merged_tpl, sampling)

    features_root = (agg_cfg.get("outputs", {}) or {}).get("features_root") \
                    or os.path.join(cfg.get("output_dir", "output"), "features")

    rep = (agg_cfg.get("reporting", {}) or {})
    report_cads = list(rep.get("cadences", []) or ["day"])
    reducer = (rep.get("reducer") or "median").lower()
    weighting = (rep.get("weighting") or "by_valid_bars").lower()

    allowed = set(allowed_reporting_for(sampling))
    bad = [c for c in report_cads if c not in allowed]
    if bad:
        raise SystemExit(f"agg.reporting.cadences invalid {bad}; allowed for sampling={sampling}: {sorted(allowed)}")

    qual_cfg = agg_cfg.get("quality", {}) or {}
    emit_validity_cols = True  # keep helper columns by default

    print(f"Sampling cadence: {sampling}")
    print(f"Merged input: {merged_dir}")
    print(f"Features root: {features_root}")
    print(f"Reporting: {report_cads} | reducer={reducer}, weighting={weighting}")

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
                report_cads=report_cads,
                reducer=reducer,
                weighting=weighting,
                qual_cfg=qual_cfg,
                emit_validity_cols=emit_validity_cols
            )
        except Exception as e:
            print(f"[{tic}] ERROR: {e}")

if __name__ == "__main__":
    main()
