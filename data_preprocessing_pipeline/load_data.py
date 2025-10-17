#!/usr/bin/env python3
"""
fetch_polygon_data.py
──────────────────────────────────────────────────────────────────────────────
Reads /Users/maojecia/Documents/_data_mining/configs/data_preprocess_config.yaml
and downloads selected Polygon.io datasets, saving CSVs under:

  output/{data_type}/{TICKER}.csv

Notes:
- News date filtering must be passed via params={"published_utc.gte": ..., "published_utc.lte": ...}.
- Some endpoints differ by client version; we feature-detect and skip gracefully.
- Trades/Quotes can be very large; when enabled, they’re fetched day-by-day.
"""

import os
import sys
import yaml
import pandas as pd
from datetime import datetime, timedelta, timezone
from polygon import RESTClient
from polygon.exceptions import BadResponse
import inspect

# ---------- helpers ----------------------------------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_df(df: pd.DataFrame, folder: str, filename: str):
    ensure_dir(folder)
    out_path = os.path.join(folder, f"{filename}.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Saved {filename}.csv → {folder}")

def normalize_timespan(ts: str) -> str:
    if not ts:
        return "day"
    m = {
        "daily": "day", "day": "day",
        "weekly": "week", "week": "week",
        "monthly": "month", "month": "month",
        "hourly": "hour", "hour": "hour",
        "minute": "minute",
        "quarter": "quarter",
        "year": "year", "yearly": "year", "annually": "year",
    }
    tsn = m.get(ts.strip().lower())
    if not tsn:
        raise ValueError("Invalid granularity. Use daily|weekly|monthly|day|week|month|hour|minute|quarter|year.")
    return tsn

def agg_row(bar) -> dict:
    return {
        "timestamp": pd.to_datetime(getattr(bar, "timestamp", getattr(bar, "t", None)), unit="ms", errors="coerce"),
        "open": getattr(bar, "open", getattr(bar, "o", None)),
        "high": getattr(bar, "high", getattr(bar, "h", None)),
        "low": getattr(bar, "low", getattr(bar, "l", None)),
        "close": getattr(bar, "close", getattr(bar, "c", None)),
        "volume": getattr(bar, "volume", getattr(bar, "v", None)),
        "vwap": getattr(bar, "vwap", getattr(bar, "vw", None)),
        "transactions": getattr(bar, "transactions", getattr(bar, "n", None)),
    }

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("timestamp").reset_index(drop=True).copy()
    out["ret"] = out["close"].pct_change()
    out["sma_20"] = out["close"].rolling(20, min_periods=1).mean()
    out["sma_50"] = out["close"].rolling(50, min_periods=1).mean()
    out["ema_12"] = out["close"].ewm(span=12, adjust=False).mean()
    out["ema_26"] = out["close"].ewm(span=26, adjust=False).mean()

    delta = out["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, pd.NA)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    out["volatility_20"] = out["ret"].rolling(20, min_periods=2).std()

    if {"high","low","close"}.issubset(out.columns):
        tr = pd.concat([
            (out["high"] - out["low"]),
            (out["high"] - out["close"].shift()).abs(),
            (out["low"] - out["close"].shift()).abs()
        ], axis=1).max(axis=1)
        out["atr_14"] = tr.rolling(14, min_periods=1).mean()

    return out.drop(columns=["ret"], errors="ignore")

def compute_custom_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("timestamp").reset_index(drop=True).copy()

    # MACD (12,26,9)
    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd
    out["macd_signal"] = signal
    out["macd_hist"] = macd - signal

    # Bollinger (20, 2σ)
    sma20 = out["close"].rolling(20, min_periods=20).mean()
    std20 = out["close"].rolling(20, min_periods=20).std()
    out["bb_mid_20"] = sma20
    out["bb_up_20"] = sma20 + 2*std20
    out["bb_dn_20"] = sma20 - 2*std20
    out["bb_bw_20"] = (out["bb_up_20"] - out["bb_dn_20"]) / out["bb_mid_20"]

    # Stochastic (14,3)
    low14 = out["low"].rolling(14, min_periods=14).min()
    high14 = out["high"].rolling(14, min_periods=14).max()
    out["stoch_k_14"] = 100 * (out["close"] - low14) / (high14 - low14)
    out["stoch_d_14"] = out["stoch_k_14"].rolling(3, min_periods=1).mean()

    # OBV
    obv = [0]
    for i in range(1, len(out)):
        if out.loc[i, "close"] > out.loc[i-1, "close"]:
            obv.append(obv[-1] + (out.loc[i, "volume"] or 0))
        elif out.loc[i, "close"] < out.loc[i-1, "close"]:
            obv.append(obv[-1] - (out.loc[i, "volume"] or 0))
        else:
            obv.append(obv[-1])
    out["obv"] = obv

    return out

def utc_start(day_str: str) -> str:
    return datetime.fromisoformat(day_str).replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

def utc_next_day(day_str: str) -> str:
    d = datetime.fromisoformat(day_str).replace(tzinfo=timezone.utc) + timedelta(days=1)
    return d.isoformat().replace("+00:00", "Z")

def iter_days(start_date: str, end_date: str):
    d0 = datetime.fromisoformat(start_date).date()
    d1 = datetime.fromisoformat(end_date).date()
    cur = d0
    while cur <= d1:
        yield cur.isoformat()
        cur = cur + timedelta(days=1)

# ---------- load config ------------------------------------------------------
CFG_PATH = "/Users/maojecia/Documents/_data_mining/configs/data_preprocess_config.yaml"

if not os.path.exists(CFG_PATH):
    print(f"Config not found: {CFG_PATH}", file=sys.stderr)
    sys.exit(1)

with open(CFG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

API_KEY = CONFIG["api_settings"]["api_key"]
client = RESTClient(API_KEY)

output_root = CONFIG.get("output_dir", "output")
ensure_dir(output_root)

timespan  = normalize_timespan(CONFIG["date_range"]["granularity"])
multiplier = int(CONFIG["date_range"].get("multiplier", 1))
date_from = CONFIG["date_range"]["from"]
date_to   = CONFIG["date_range"]["to"]
adjusted  = bool(CONFIG["api_settings"].get("adjusted", True))

fetch   = CONFIG["fetch"]
tickers = CONFIG["stocks"]

# ---------- 1) Aggregates (OHLCV) -------------------------------------------
if fetch.get("aggregates", False):
    folder = os.path.join(output_root, f"aggregates_{timespan}")
    for t in tickers:
        print(f"Fetching {timespan} aggregates for {t} [{date_from} → {date_to}]")
        try:
            aggs_iter = client.list_aggs(
                ticker=t,
                multiplier=multiplier,
                timespan=timespan,
                from_=date_from,
                to=date_to,
                adjusted=adjusted,
            )
            rows = [agg_row(b) for b in aggs_iter]
            if not rows:
                print(f"⚠️ No aggregates for {t}.")
                continue
            df = pd.DataFrame(rows).sort_values("timestamp")
            save_df(df, folder, t)

            want_basic_inds = fetch.get("moving_averages", False) or fetch.get("volatility", False) or fetch.get("momentum", False)
            if want_basic_inds:
                indf = compute_indicators(df)
                save_df(indf, os.path.join(output_root, f"indicators_{timespan}"), t)

            if fetch.get("custom_indicators", False):
                cidf = compute_custom_indicators(df)
                save_df(cidf, os.path.join(output_root, f"custom_indicators_{timespan}"), t)

        except BadResponse as e:
            print(f"⚠️ Aggregates error for {t}: {e}")
        except Exception as e:
            print(f"⚠️ Unexpected error (aggregates) for {t}: {e}")

# ---------- 2) Daily open/close (for `date_to`) -----------------------------
if fetch.get("daily_open_close", False):
    folder = os.path.join(output_root, "daily_open_close")
    for t in tickers:
        try:
            resp = client.get_daily_open_close(t, date_to, adjusted=adjusted)
            df = pd.DataFrame([{
                "symbol": getattr(resp, "symbol", t),
                "date": date_to,
                "open": getattr(resp, "open", None),
                "high": getattr(resp, "high", None),
                "low": getattr(resp, "low", None),
                "close": getattr(resp, "close", None),
                "afterHours": getattr(resp, "afterHours", None),
                "preMarket": getattr(resp, "preMarket", None),
                "volume": getattr(resp, "volume", None),
            }])
            save_df(df, folder, t)
        except Exception as e:
            print(f"⚠️ daily_open_close error for {t}: {e}")

# ---------- 3) Previous close -----------------------------------------------
if fetch.get("previous_close", False):
    folder = os.path.join(output_root, "previous_close")
    for t in tickers:
        try:
            prev = client.get_previous_close(t, adjusted=adjusted)
            results = getattr(prev, "results", None)
            if results is None:
                results = list(prev) if prev is not None else []
            rows = []
            for b in results:
                rows.append({
                    "symbol": t,
                    "timestamp": pd.to_datetime(getattr(b, "timestamp", getattr(b, "t", None)), unit="ms", errors="coerce"),
                    "open": getattr(b, "open", getattr(b, "o", None)),
                    "high": getattr(b, "high", getattr(b, "h", None)),
                    "low": getattr(b, "low", getattr(b, "l", None)),
                    "close": getattr(b, "close", getattr(b, "c", None)),
                    "volume": getattr(b, "volume", getattr(b, "v", None)),
                    "vwap": getattr(b, "vwap", getattr(b, "vw", None)),
                    "transactions": getattr(b, "transactions", getattr(b, "n", None)),
                })
            if rows:
                save_df(pd.DataFrame(rows), folder, t)
        except Exception as e:
            print(f"⚠️ previous_close error for {t}: {e}")

# ---------- 4) Snapshots (compact + version tolerant) -----------------------
def _resp_items(resp):
    return (getattr(resp, "tickers", None)
            or getattr(resp, "results", None)
            or getattr(resp, "__dict__", {}).get("tickers")
            or resp)

def _snapshot_all_try(client, tickers):
    # Try client.get_snapshot_all with common signatures (incl. requiring market_type)
    for owner in (client, getattr(client, "stocks", None), getattr(client, "stocks_client", None)):
        if owner is None or not hasattr(owner, "get_snapshot_all"):
            continue
        fn = owner.get_snapshot_all
        params = list(inspect.signature(fn).parameters.keys())
        try:
            if params == []:
                resp = fn()
            elif params == ['market_type']:
                resp = fn("stocks")
            elif params == ['market_type', 'symbols']:
                resp = fn("stocks", ",".join(tickers))
            elif 'market_type' in params and 'params' in params:
                resp = fn("stocks", params={"ticker.any_of": ",".join(tickers)})
            elif 'params' in params:
                resp = fn(params={"ticker.any_of": ",".join(tickers)})
            else:
                resp = fn()
            items = _resp_items(resp) or []
            out = []
            for it in items:
                d = getattr(it, "__dict__", it)
                t = d.get("ticker") or d.get("T")
                if t and (not tickers or t in tickers):
                    out.append(d)
            if out:
                return out
        except Exception:
            pass
    return None

def _snapshot_single_try(client, ticker):
    # Order: explicit market_type first (covers SDKs that require it)
    for call in (
        lambda: getattr(client, "get_snapshot_ticker", None) and client.get_snapshot_ticker("stocks", ticker),
        lambda: getattr(client, "get_snapshot", None) and client.get_snapshot("stocks", ticker),
        lambda: getattr(client, "get_snapshot", None) and client.get_snapshot(ticker=ticker),
        lambda: getattr(getattr(client, "stocks", None), "get_snapshot", None) and client.stocks.get_snapshot(ticker=ticker),
        lambda: getattr(getattr(client, "stocks_client", None), "get_snapshot", None) and client.stocks_client.get_snapshot(ticker=ticker),
    ):
        try:
            resp = call()
            if resp:
                return getattr(resp, "__dict__", resp)
        except Exception:
            continue
    return None

if fetch.get("snapshots", False):
    folder = os.path.join(output_root, "snapshots")
    ensure_dir(folder)
    wrote_any = False

    items = _snapshot_all_try(client, tickers)
    if items:
        df_all = pd.DataFrame(items)
        save_df(df_all, folder, "snapshot_all_filtered")
        key = "ticker" if "ticker" in df_all.columns else ("T" if "T" in df_all.columns else None)
        if key:
            for tkr, sub in df_all.groupby(key):
                save_df(sub.copy(), folder, str(tkr))
        wrote_any = True

    if not wrote_any:
        for t in tickers:
            payload = _snapshot_single_try(client, t)
            if payload:
                save_df(pd.DataFrame([payload]), folder, t)
                wrote_any = True

    if not wrote_any:
        print("⚠️ snapshots: no compatible method worked on this SDK.")

# ---------- 5) Reference / fundamentals ------------------------------------
if fetch.get("ticker_details", False):
    folder = os.path.join(output_root, "ticker_details")
    for t in tickers:
        try:
            det = client.get_ticker_details(t)
            save_df(pd.DataFrame([det.__dict__]), folder, t)
        except Exception as e:
            print(f"⚠️ ticker_details error for {t}: {e}")

if fetch.get("dividends", False):
    folder = os.path.join(output_root, "dividends")
    for t in tickers:
        try:
            divs = list(client.list_dividends(ticker=t, limit=1000))
            if divs:
                save_df(pd.DataFrame([d.__dict__ for d in divs]), folder, t)
        except Exception as e:
            print(f"⚠️ dividends error for {t}: {e}")

if fetch.get("splits", False):
    folder = os.path.join(output_root, "splits")
    for t in tickers:
        try:
            splits = list(client.list_splits(ticker=t, limit=1000))
            if splits:
                save_df(pd.DataFrame([s.__dict__ for s in splits]), folder, t)
        except Exception as e:
            print(f"⚠️ splits error for {t}: {e}")

if fetch.get("corporate_actions", False):
    folder = os.path.join(output_root, "corporate_actions")
    for t in tickers:
        try:
            if hasattr(client, "list_corporate_actions"):
                rows = [ca.__dict__ for ca in client.list_corporate_actions(ticker=t, limit=1000)]
                if rows:
                    save_df(pd.DataFrame(rows), folder, t)
            else:
                print(f"ℹ️ corporate_actions not exposed by this SDK; skipped {t}.")
        except Exception as e:
            print(f"⚠️ corporate_actions error for {t}: {e}")

if fetch.get("financials", False):
    folder = os.path.join(output_root, "financials")
    for t in tickers:
        try:
            if hasattr(client, "list_stock_financials"):
                fins = list(client.list_stock_financials(ticker=t, limit=100))
            elif hasattr(getattr(client, "vx", {}), "list_stock_financials"):
                fins = list(client.vx.list_stock_financials(ticker=t, limit=100))
            else:
                fins = []
            if fins:
                save_df(pd.DataFrame([f.__dict__ for f in fins]), folder, t)
        except Exception as e:
            print(f"⚠️ financials error for {t}: {e}")

if fetch.get("analyst_ratings", False):
    folder = os.path.join(output_root, "analyst_ratings")
    for t in tickers:
        try:
            rows = []
            if hasattr(client, "list_analyst_ratings"):
                rows = [r.__dict__ for r in client.list_analyst_ratings(ticker=t, limit=1000)]
            elif hasattr(getattr(client, "reference", {}), "list_analyst_ratings"):
                rows = [r.__dict__ for r in client.reference.list_analyst_ratings(ticker=t, limit=1000)]
            if rows:
                save_df(pd.DataFrame(rows), folder, t)
        except Exception as e:
            print(f"⚠️ analyst_ratings error for {t}: {e}")

# ---------- 6) News (timestamp, url, text) ----------------------------------
if fetch.get("news", False):
    folder = os.path.join(output_root, "news")
    for t in tickers:
        try:
            params = {"published_utc.gte": date_from, "published_utc.lte": date_to}
            rows = []
            for item in client.list_ticker_news(ticker=t, limit=1000, params=params):
                ts = getattr(item, "published_utc", None)
                url = getattr(item, "article_url", None) or getattr(item, "url", None)
                parts = [getattr(item, "title", "") or "", getattr(item, "description", "") or "", getattr(item, "summary", "") or ""]
                rows.append({"timestamp": pd.to_datetime(ts) if ts else pd.NaT, "url": url, "text": " ".join(p for p in parts if p).strip()})
            if rows:
                save_df(pd.DataFrame(rows).sort_values("timestamp"), folder, t)
        except Exception as e:
            print(f"⚠️ news error for {t}: {e}")

# ---------- 7) Trades & Quotes (day-by-day) ---------------------------------
if fetch.get("trades", False):
    root = os.path.join(output_root, "trades")
    for t in tickers:
        print(f"Fetching TRADES day-by-day for {t}...")
        for day in iter_days(date_from, date_to):
            try:
                folder = os.path.join(root, day)
                rows = []
                try:
                    for tr in client.list_trades(ticker=t, execution_date=day, limit=50000):
                        rows.append(tr.__dict__)
                except TypeError:
                    params = {"timestamp.gte": utc_start(day), "timestamp.lt": utc_next_day(day)}
                    for tr in client.list_trades(ticker=t, limit=50000, params=params):
                        rows.append(tr.__dict__)
                if rows:
                    save_df(pd.DataFrame(rows), folder, t)
            except Exception as e:
                print(f"⚠️ trades error for {t} {day}: {e}")

if fetch.get("quotes", False):
    root = os.path.join(output_root, "quotes")
    for t in tickers:
        print(f"Fetching QUOTES day-by-day for {t}...")
        for day in iter_days(date_from, date_to):
            try:
                folder = os.path.join(root, day)
                rows = []
                try:
                    for q in client.list_quotes(ticker=t, execution_date=day, limit=50000):
                        rows.append(q.__dict__)
                except TypeError:
                    params = {"timestamp.gte": utc_start(day), "timestamp.lt": utc_next_day(day)}
                    for q in client.list_quotes(ticker=t, limit=50000, params=params):
                        rows.append(q.__dict__)
                if rows:
                    save_df(pd.DataFrame(rows), folder, t)
            except Exception as e:
                print(f"⚠️ quotes error for {t} {day}: {e}")

# ---------- 8) Market status ------------------------------------------------
if fetch.get("market_status", False):
    folder = os.path.join(output_root, "market_status")
    try:
        status = client.get_market_status()
        save_df(pd.DataFrame([status.__dict__]), folder, "market_status")
    except Exception as e:
        print(f"⚠️ market_status error: {e}")

print("\n🎉 All selected datasets saved under 'output/'.")
