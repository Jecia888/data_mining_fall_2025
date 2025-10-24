#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assemble daily JSON per ticker (end-of-hour aligned, with auto-alignment for intermediate)
────────────────────────────────────────────────────────────────────
- news:   <news_dir>/<TICKER>.csv with columns [timestamp,url,text] (ET)
- AGG/FFT/CWT hour features: <features_root>/<TICKER>/<agg|fft|cwt>_<sampling>/hour.csv
  * We always align by window END time (e.g., win_end / hour_end_ts / feature_ts).
  * If only day_id + hour_label is present, we treat hour_label as START and +1h.

- intermediate hourly CSV: <inter_dir>/<TICKER>.csv (has 'timestamp' etc.)
  * --inter-align:
        auto            → choose among {no shift, +1h, -1h} by max overlap with AGG/FFT/CWT hours
        treat_as_start  → +1h (start → end)
        treat_as_end    → no shift
        as_is           → no shift

Write JSON ONLY IF:
  - the day has at least one news item, AND
  - per_hour_data has >=1 hour with at least one feature (we only include non-empty hours)
"""

import os
import json
import glob
import argparse
import csv
from typing import Dict, List, Optional, Tuple, Iterable, Set

import pandas as pd
import numpy as np

# ───────────────────────── TZ helper ───────────────────────── #
try:
    from zoneinfo import ZoneInfo
    NY = ZoneInfo("America/New_York")
except Exception:
    import pytz
    NY = pytz.timezone("America/New_York")

# ───────────────────────── Utilities ───────────────────────── #

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def safe_ticker(t: str) -> str:
    return (t or "").replace("/", "_").replace("\\", "_").replace(".", "_").replace(" ", "")

def to_et_series(s: pd.Series) -> pd.Series:
    """Parse anything → UTC → ET; return Series with tz=ET."""
    ts = pd.to_datetime(s, errors="coerce", utc=True)
    return ts.dt.tz_convert(NY)

def label_time_key(ts: pd.Timestamp) -> str:
    """'HH:MM' in ET."""
    return pd.Timestamp(ts).tz_convert(NY).strftime("%H:%M")

def within_same_et_day(ts: pd.Timestamp, day: pd.Timestamp) -> bool:
    return pd.Timestamp(ts).tz_convert(NY).date() == pd.Timestamp(day).tz_convert(NY).date()

def list_tickers_from_news(news_dir: str) -> List[str]:
    paths = glob.glob(os.path.join(news_dir, "*.csv"))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]

def list_tickers_from_features(feat_root: str) -> List[str]:
    out = []
    for p in glob.glob(os.path.join(feat_root, "*")):
        if not os.path.isdir(p):
            continue
        base = os.path.basename(p)
        if glob.glob(os.path.join(p, "agg_*")) or glob.glob(os.path.join(p, "fft_*")) or glob.glob(os.path.join(p, "cwt_*")):
            out.append(base)
    return out

def list_tickers_from_intermediate(inter_dir: str) -> List[str]:
    paths = glob.glob(os.path.join(inter_dir, "*.csv"))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]

# ───────────────────────── Robust NEWS reader ───────────────────────── #

def read_news_per_day(news_csv_path: str) -> Dict[pd.Timestamp, List[Dict[str, str]]]:
    """
    Read <ticker>.csv tolerating raw commas in text via csv.reader.
    Return {ET_midnight: [ {timestamp_et, text, url}, ... ] }
    """
    if not os.path.exists(news_csv_path):
        return {}

    rows: List[Tuple[str, str, str]] = []
    with open(news_csv_path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        header = next(reader, None)

        has_header = False
        if header:
            hdr = [c.strip().lower() for c in header]
            has_header = ("timestamp" in hdr[0]) and ("url" in " ".join(hdr))

        if header and not has_header:
            row = header
            if len(row) >= 2:
                ts = row[0].strip()
                url = row[1].strip()
                text = ",".join(row[2:]).strip() if len(row) > 2 else ""
                rows.append((ts, url, text))

        for row in reader:
            if not row or all((c.strip() == "" for c in row)):
                continue
            ts = row[0].strip() if len(row) >= 1 else ""
            url = row[1].strip() if len(row) >= 2 else ""
            text = ",".join(row[2:]).strip() if len(row) >= 3 else ""
            if ts == "" and url == "" and text == "":
                continue
            rows.append((ts, url, text))

    if not rows:
        return {}

    df = pd.DataFrame(rows, columns=["timestamp", "url", "text"])
    ts = to_et_series(df["timestamp"])
    df["timestamp_et"] = ts
    df = df.dropna(subset=["timestamp_et"])
    df["date_et"] = df["timestamp_et"].dt.normalize()

    out: Dict[pd.Timestamp, List[Dict[str, str]]] = {}
    for d, sub in df.groupby("date_et"):
        items = []
        sub = sub.sort_values("timestamp_et")
        for _, r in sub.iterrows():
            items.append({
                "timestamp_et": pd.Timestamp(r["timestamp_et"]).isoformat(),
                "text": (r.get("text") or ""),
                "url": (r.get("url") or ""),
            })
        out[pd.Timestamp(d).tz_convert(NY)] = items
    return out

# ───────────────────────── END-of-hour index helpers ───────────────────────── #

def end_index_from_df(df: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Convert various time layouts to END-of-hour (ET) index:
      1) hour_end_ts / win_end / end_ts → use directly.
      2) feature_ts → treat as end.
      3) day_id + hour_label ('HH:MM') → treat as start, then +1h.
      4) fallback time-like columns:
         - if name contains 'start' → +1h; else use as-is.
    """
    for c in ["hour_end_ts", "win_end", "end_ts"]:
        if c in df.columns:
            return pd.DatetimeIndex(to_et_series(df[c]))
    if "feature_ts" in df.columns:
        return pd.DatetimeIndex(to_et_series(df["feature_ts"]))
    if "day_id" in df.columns and "hour_label" in df.columns:
        day = to_et_series(df["day_id"]).dt.normalize()
        hm = df["hour_label"].astype(str).str.strip()
        parts = hm.str.extract(r'^\s*(\d{1,2})\s*:\s*(\d{2})\s*$')
        hh = pd.to_numeric(parts[0], errors="coerce").fillna(0).astype(int)
        mm = pd.to_numeric(parts[1], errors="coerce").fillna(0).astype(int)
        idx = day + pd.to_timedelta(hh, unit="h") + pd.to_timedelta(mm, unit="m") + pd.Timedelta(hours=1)
        return pd.DatetimeIndex(idx)
    for c in ["timestamp", "time", "ts", "win_start", "hour_start_ts"]:
        if c in df.columns:
            idx = to_et_series(df[c])
            if "start" in c.lower():
                idx = idx + pd.Timedelta(hours=1)
            return pd.DatetimeIndex(idx)
    raise ValueError("Cannot derive end-of-hour index from dataframe columns.")

# ───────────────────────── Feature readers ───────────────────────── #

def read_hour_csv_generic(path: str, pick_cols: List[str], prefix: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    idx = end_index_from_df(df)
    keep = [c for c in pick_cols if c in df.columns]
    if not keep:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz=NY))
    sub = df[keep].copy()
    sub.index = idx
    sub = sub[~sub.index.isna()].sort_index()
    sub.index = pd.DatetimeIndex(sub.index).tz_convert(NY)
    sub.columns = [f"{prefix}.{c}" for c in sub.columns]
    return sub

def read_hour_fft(path: str, series: List[str], suffixes: List[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    idx = end_index_from_df(df)
    cols = []
    for s in series:
        for suf in suffixes:
            name = f"{s}_{suf}"
            if name in df.columns:
                cols.append(name)
    if not cols:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz=NY))
    sub = df[cols].copy()
    sub.index = idx
    sub = sub[~sub.index.isna()].sort_index()
    sub.index = pd.DatetimeIndex(sub.index).tz_convert(NY)
    sub.columns = [f"fft.{c}" for c in sub.columns]
    return sub

def read_hour_cwt(path: str, series: List[str], suffixes: List[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    idx = end_index_from_df(df)
    cols = []
    for s in series:
        for suf in suffixes:
            name = f"{s}_{suf}"
            if name in df.columns:
                cols.append(name)
    if not cols:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz=NY))
    sub = df[cols].copy()
    sub.index = idx
    sub = sub[~sub.index.isna()].sort_index()
    sub.index = pd.DatetimeIndex(sub.index).tz_convert(NY)
    sub.columns = [f"cwt.{c}" for c in sub.columns]
    return sub

# ───────────────────────── Intermediate reader (auto-align) ───────────────────────── #

def find_intermediate_csv(inter_dir: str, ticker: str) -> Optional[str]:
    p1 = os.path.join(inter_dir, f"{ticker}.csv")
    if os.path.exists(p1):
        return p1
    cand = glob.glob(os.path.join(inter_dir, f"{ticker}*.csv"))
    return cand[0] if cand else None

def _overlap_score(idx: pd.DatetimeIndex, feat_marks: Set[pd.Timestamp]) -> int:
    if not len(idx) or not feat_marks:
        return 0
    # exact hour match
    return sum(1 for t in idx if t in feat_marks)

def _auto_shift_inter_index(raw_idx: pd.DatetimeIndex, feat_marks: Set[pd.Timestamp]) -> Tuple[pd.DatetimeIndex, str]:
    """Try {0h, +1h, -1h}, pick the max overlap; tie-breaker: prefer +1h, then 0h, then -1h."""
    cand = [
        (raw_idx, "as_is"),
        (raw_idx + pd.Timedelta(hours=1), "treat_as_start"),
        (raw_idx - pd.Timedelta(hours=1), "minus_1h"),
    ]
    scored = [( _overlap_score(i, feat_marks), tag, i) for (i, tag) in cand]
    scored.sort(key=lambda x: (x[0], 1 if x[1]=="treat_as_start" else (0 if x[1]=="as_is" else -1)), reverse=True)
    best_score, best_tag, best_idx = scored[0]
    return best_idx, best_tag

def read_hour_intermediate(
    path: str,
    pick_cols: List[str],
    time_col: str,
    align: str,
    feat_marks: Set[pd.Timestamp],
    prefix: str = "inter",
) -> Tuple[pd.DataFrame, str]:
    """
    Read intermediate hourly CSV; align its timestamp to END-of-hour to match AGG/FFT/CWT.
    Returns (df, chosen_align_tag)
    """
    if not os.path.exists(path):
        return pd.DataFrame(), "missing"

    df = pd.read_csv(path, encoding="utf-8-sig")
    if time_col not in df.columns:
        raise ValueError(f"[intermediate] time column '{time_col}' not found in {path}")

    if "session" in df.columns:
        df = df[df["session"].astype(str).str.lower() == "rth"].copy()

    # ────────────────────────────────────────────────────────────────
    # ★★ 这里就是读取 intermediate 的时间，并做对齐的地方 ★★
    # raw_ts 为原始时间（通常表示该小时“开始”）
    raw_ts = to_et_series(df[time_col])

    if align == "treat_as_start":
        # → 把“开始时间”转换为“结束时间”：强制 +1 小时
        aligned_idx = raw_ts + pd.Timedelta(hours=1)
        chosen = "treat_as_start"
    elif align in ("treat_as_end", "as_is"):
        # → 保持不动（已是“结束时间”）
        aligned_idx = raw_ts
        chosen = "treat_as_end"
    elif align == "auto":
        # → 自动挑选 {不移位, +1h, -1h} 与 AGG/FFT/CWT 的小时集合重合最多的方案
        aligned_idx, chosen = _auto_shift_inter_index(pd.DatetimeIndex(raw_ts), feat_marks)
    else:
        aligned_idx = raw_ts
        chosen = "as_is"
    # ────────────────────────────────────────────────────────────────

    keep = [c for c in pick_cols if c in df.columns]
    if not keep:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz=NY)), chosen

    sub = df[keep].copy()
    sub.index = pd.DatetimeIndex(aligned_idx).tz_convert(NY)
    sub = sub[~sub.index.isna()].sort_index()
    sub.columns = [f"{prefix}.{c}" for c in sub.columns]
    return sub, chosen


# ───────────────────────── Assembler ───────────────────────── #

def assemble_one_day_for_ticker(
    ticker: str,
    day_et: pd.Timestamp,
    news_per_day: Dict[pd.Timestamp, List[Dict[str, str]]],
    dfs_sources: List[pd.DataFrame],
) -> Tuple[dict, bool]:
    """
    Only keep hours that actually have features; if none, per_hour_data is {}.
    Return (payload, has_any_feature).
    """
    news_items = news_per_day.get(day_et.normalize(), [])

    hour_marks = set()
    for df in dfs_sources:
        if df is None or df.empty:
            continue
        for ts in df.index:
            if within_same_et_day(ts, day_et):
                hour_marks.add(pd.Timestamp(ts).tz_convert(NY))

    per_hour_data: Dict[str, Dict] = {}
    has_any_feature = False

    for ts in sorted(hour_marks):
        feat = {}
        for df in dfs_sources:
            if df is None or df.empty:
                continue
            if ts in df.index:
                row = df.loc[ts]
                if isinstance(row, pd.Series):
                    items = row.to_dict()
                else:
                    items = row.iloc[-1].to_dict()
                for k, v in items.items():
                    if pd.notna(v):
                        feat[k] = float(v) if isinstance(v, (int, float, np.floating)) else v
        if feat:
            has_any_feature = True
            per_hour_data[label_time_key(ts)] = feat

    payload = {
        "ticker": ticker,
        "date": day_et.strftime("%Y-%m-%d"),
        "news": news_items,
        "per_hour_data": per_hour_data
    }
    return payload, has_any_feature

# ───────────────────────── CLI / Main ───────────────────────── #

def main():
    ap = argparse.ArgumentParser(description="Assemble per-day JSON from news + hour-level features + intermediate (end-of-hour aligned)")
    ap.add_argument("--news-dir", required=True)
    ap.add_argument("--features-root", required=True)
    ap.add_argument("--inter-dir", default="")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--sampling", choices=["minute", "hour", "day"], default="minute")

    ap.add_argument("--tickers", default="")
    ap.add_argument("--date-from", required=True)
    ap.add_argument("--date-to", required=True)

    # kept for backward-compat; we always align by end-of-hour internally
    ap.add_argument("--hour-mode", choices=["top_of_hour", "include_open_half_hour"], default="top_of_hour",
                    help="(deprecated) ignored internally; we always use end-of-hour alignment.")

    # inclusion lists
    ap.add_argument("--agg-cols", default="")
    ap.add_argument("--fft-series", default="")
    ap.add_argument("--fft-suffixes", default="")
    ap.add_argument("--cwt-series", default="")
    ap.add_argument("--cwt-suffixes", default="")

    # intermediate
    ap.add_argument("--inter-cols", default="")
    ap.add_argument("--inter-time-col", default="timestamp")
    ap.add_argument("--inter-align", choices=["auto", "treat_as_start", "treat_as_end", "as_is"], default="auto")

    args = ap.parse_args()

    news_dir = args.news_dir
    features_root = args.features_root
    inter_dir = args.inter_dir.strip()
    out_root = args.out_root
    sampling = args.sampling

    ensure_dir(out_root)

    # tickers
    if args.tickers.strip():
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = set()
        tickers |= set(list_tickers_from_news(news_dir))
        tickers |= set(list_tickers_from_features(features_root))
        if inter_dir:
            tickers |= set(list_tickers_from_intermediate(inter_dir))
        tickers = sorted(tickers)
    if not tickers:
        raise SystemExit("No tickers found. Provide --tickers or check folders.")

    # includes
    agg_cols = [s.strip() for s in args.agg_cols.split(",") if s.strip()]
    fft_series = [s.strip() for s in args.fft_series.split(",") if s.strip()]
    fft_suffixes = [s.strip() for s in args.fft_suffixes.split(",") if s.strip()]
    cwt_series = [s.strip() for s in args.cwt_series.split(",") if s.strip()]
    cwt_suffixes = [s.strip() for s in args.cwt_suffixes.split(",") if s.strip()]
    inter_cols = [s.strip() for s in args.inter_cols.split(",") if s.strip()]

    print(f"[assemble] news_dir={news_dir}")
    print(f"[assemble] features_root={features_root}")
    print(f"[assemble] inter_dir={inter_dir or 'NONE'} (time_col={args.inter_time_col}, align={args.inter_align})")
    print(f"[assemble] out_root={out_root}")
    print(f"[assemble] sampling={sampling}")
    print(f"[assemble] tickers={tickers}")
    print(f"[assemble] include: agg={agg_cols}, fft={(fft_series, fft_suffixes)}, cwt={(cwt_series, cwt_suffixes)}, inter={inter_cols}")
    print(f"[assemble] note: we align ALL sources to end-of-hour keys (e.g., 10:00–11:00 → 11:00).")

    date_from = pd.Timestamp(args.date_from).tz_localize(NY).normalize()
    date_to   = pd.Timestamp(args.date_to).tz_localize(NY).normalize()

    for t in tickers:
        t_safe = safe_ticker(t)

        # --- news ---
        news_csv = os.path.join(news_dir, f"{t}.csv")
        news_map = read_news_per_day(news_csv)

        # --- features: build union of end-of-hour marks first ---
        agg_hour = os.path.join(features_root, t, f"agg_{sampling}", "hour.csv")
        fft_hour = os.path.join(features_root, t, f"fft_{sampling}", "hour.csv")
        cwt_hour = os.path.join(features_root, t, f"cwt_{sampling}", "hour.csv")

        dfs: List[pd.DataFrame] = []
        feat_marks: Set[pd.Timestamp] = set()

        if agg_cols:
            df_agg = read_hour_csv_generic(agg_hour, pick_cols=agg_cols, prefix="agg")
            if not df_agg.empty:
                dfs.append(df_agg)
                feat_marks |= set(pd.DatetimeIndex(df_agg.index).tz_convert(NY))

        if fft_series and fft_suffixes:
            df_fft = read_hour_fft(fft_hour, series=fft_series, suffixes=fft_suffixes)
            if not df_fft.empty:
                dfs.append(df_fft)
                feat_marks |= set(pd.DatetimeIndex(df_fft.index).tz_convert(NY))

        if cwt_series and cwt_suffixes:
            df_cwt = read_hour_cwt(cwt_hour, series=cwt_series, suffixes=cwt_suffixes)
            if not df_cwt.empty:
                dfs.append(df_cwt)
                feat_marks |= set(pd.DatetimeIndex(df_cwt.index).tz_convert(NY))

        # --- intermediate: AUTO align to feature marks (if requested) ---
        if inter_dir and inter_cols:
            inter_csv = find_intermediate_csv(inter_dir, t)
            if inter_csv:
                df_inter, chosen = read_hour_intermediate(
                    inter_csv,
                    pick_cols=inter_cols,
                    time_col=args.inter_time_col,
                    align=args.inter_align,
                    feat_marks=feat_marks,
                    prefix="inter",
                )
                print(f"[{t}] intermediate align → {chosen} (rows={len(df_inter)})")
                if not df_inter.empty:
                    dfs.append(df_inter)
            else:
                print(f"[{t}] WARN: no intermediate CSV found under {inter_dir}")

        out_dir_ticker = os.path.join(out_root, t_safe)
        ensure_dir(out_dir_ticker)

        any_written = False
        cur = date_from
        while cur <= date_to:
            day_key = cur.normalize()
            news_items = news_map.get(day_key, [])
            if not news_items:
                cur += pd.Timedelta(days=1)
                continue

            payload, has_any_feature = assemble_one_day_for_ticker(
                ticker=t,
                day_et=cur,
                news_per_day=news_map,
                dfs_sources=dfs,
            )
            # 严格：per_hour_data 必须非空且 has_any_feature 为真
            if not has_any_feature or not payload["per_hour_data"]:
                print(f"[{t}] {cur.date()} skipped (news present, but no hourly features).")
                cur += pd.Timedelta(days=1)
                continue

            out_path = os.path.join(out_dir_ticker, f"{cur.strftime('%Y-%m-%d')}_{t}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"[{t}] {cur.date()} → {out_path}")
            any_written = True
            cur += pd.Timedelta(days=1)

        if not any_written:
            print(f"[{t}] NOTE: no JSON written (no qualifying days with news+features).")

if __name__ == "__main__":
    main()
