#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidate assembled daily JSON into a single 'hourly_output' folder.

- Input roots: one or more folders like:
    D:\...\output_2021\assembled_json
    D:\...\output_2022\assembled_json
    ...
    D:\...\output_2025\assembled_json

  Each root has structure:
    <root>/<SAFE_TICKER>/<YYYY-MM-DD>_<TICKER>.json

- Output:
    <out_dir>/hourly_output/<SAFE_TICKER>/<YYYY-MM-DD>_<TICKER>.json

De-duplication (same TICKER+DATE from multiple roots):
1) Prefer JSON whose per_hour_data contains at least one non-empty hour dict
2) If tie, prefer larger file size
3) If tie, prefer newer mtime
4) Otherwise keep the first seen

By default we **copy** files; you can choose to move.
"""

import os
import re
import json
import glob
import shutil
import argparse
from typing import Dict, Tuple, Optional

# ------- helpers -------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_ticker(t: str) -> str:
    return (t or "").replace("/", "_").replace("\\", "_").replace(".", "_").replace(" ", "")

def parse_date_from_filename(name: str) -> Optional[str]:
    # expect "YYYY-MM-DD_<TICKER>.json"
    m = re.match(r"^(\d{4}-\d{2}-\d{2})_", name)
    return m.group(1) if m else None

def load_json_safely(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def json_has_any_features(obj: dict) -> bool:
    phd = obj.get("per_hour_data", {})
    if not isinstance(phd, dict) or not phd:
        return False
    for v in phd.values():
        if isinstance(v, dict) and len(v) > 0:
            return True
    return False

def choose_better(path_a: str, path_b: str) -> str:
    """Pick the 'better' JSON among duplicates."""
    ja = load_json_safely(path_a) or {}
    jb = load_json_safely(path_b) or {}
    fa = json_has_any_features(ja)
    fb = json_has_any_features(jb)
    if fa and not fb:
        return path_a
    if fb and not fa:
        return path_b

    # tie → larger file size
    sa = os.path.getsize(path_a)
    sb = os.path.getsize(path_b)
    if sa > sb:
        return path_a
    if sb > sa:
        return path_b

    # tie → newer mtime
    ma = os.path.getmtime(path_a)
    mb = os.path.getmtime(path_b)
    return path_a if ma >= mb else path_b

# ------- main -------

def main():
    ap = argparse.ArgumentParser(description="Consolidate daily JSON into hourly_output/<TICKER>/...")
    ap.add_argument(
        "--in-roots",
        nargs="+",
        required=True,
        help="One or more assembled_json roots (space-separated)."
    )
    ap.add_argument(
        "--out-dir",
        default="hourly_output",
        help="Destination folder (will create hourly_output/<TICKER>/...). Default: ./hourly_output"
    )
    ap.add_argument(
        "--mode",
        choices=["copy", "move"],
        default="copy",
        help="Copy or move files into the output structure. Default: copy"
    )
    ap.add_argument(
        "--skip-empty",
        action="store_true",
        help="If set, skip JSON whose per_hour_data has no non-empty hour dict."
    )
    args = ap.parse_args()

    in_roots = [r.rstrip("\\/") for r in args.in_roots]
    out_dir  = args.out_dir
    ensure_dir(out_dir)

    # Map: (safe_ticker, date) -> chosen file path
    chosen: Dict[Tuple[str, str], str] = {}

    # 1) scan all roots
    for root in in_roots:
        # pattern: <root>/<SAFE_TICKER>/*.json
        for ticker_dir in glob.glob(os.path.join(root, "*")):
            if not os.path.isdir(ticker_dir):
                continue
            safe_tkr = os.path.basename(ticker_dir)
            for fpath in glob.glob(os.path.join(ticker_dir, "*.json")):
                fname = os.path.basename(fpath)
                day = parse_date_from_filename(fname)
                if not day:
                    # Fallback: try reading JSON for 'date'
                    obj = load_json_safely(fpath)
                    day = (obj or {}).get("date")
                    if not (isinstance(day, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", day or "")):
                        print(f"[skip] cannot determine date from filename or content: {fpath}")
                        continue

                # Optional skip-empty
                if args.skip_empty:
                    obj = load_json_safely(fpath) or {}
                    if not json_has_any_features(obj):
                        # print(f"[skip-empty] {fpath}")
                        continue

                key = (safe_tkr, day)
                if key not in chosen:
                    chosen[key] = fpath
                else:
                    chosen[key] = choose_better(chosen[key], fpath)

    # 2) write into out_dir/hourly_output/<SAFE_TICKER>/<YYYY-MM-DD>_<TICKER>.json
    copied = 0
    moved  = 0
    ensure_dir(out_dir)

    for (safe_tkr, day), src in sorted(chosen.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        # Prefer the TICKER in JSON content for filename fidelity
        obj = load_json_safely(src) or {}
        ticker_for_name = obj.get("ticker") or safe_tkr
        # Keep the original filename if it already starts with the date, else rebuild
        base = os.path.basename(src)
        if not base.startswith(day + "_"):
            base = f"{day}_{ticker_for_name}.json"

        dst_dir = os.path.join(out_dir, safe_tkr)
        ensure_dir(dst_dir)
        dst = os.path.join(dst_dir, base)

        # If destination exists and is the same file (case-insensitive), skip.
        if os.path.abspath(src) == os.path.abspath(dst):
            continue

        # Make sure we don't collide with an existing different file
        if os.path.exists(dst):
            # If exists, decide if current src is better; if yes, overwrite
            prev = dst
            better = choose_better(prev, src)
            if better == src:
                # overwrite
                try:
                    if args.mode == "copy":
                        shutil.copy2(src, dst)
                        copied += 1
                    else:
                        shutil.move(src, dst)
                        moved += 1
                except Exception as e:
                    print(f"[WARN] overwrite failed: {dst} ← {src} ({e})")
            else:
                # keep existing; optionally remove duplicate if moving across roots
                pass
        else:
            try:
                if args.mode == "copy":
                    shutil.copy2(src, dst)
                    copied += 1
                else:
                    shutil.move(src, dst)
                    moved += 1
            except Exception as e:
                print(f"[WARN] transfer failed: {dst} ← {src} ({e})")

    print(f"[done] tickers={len(set(t for (t, _) in chosen.keys()))}, days={len(chosen)}")
    if args.mode == "copy":
        print(f"[copy] files={copied} → {out_dir}")
    else:
        print(f"[move] files={moved} → {out_dir}")

if __name__ == "__main__":
    main()
