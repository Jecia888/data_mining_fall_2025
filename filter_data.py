#!/usr/bin/env python3
import os
import json
import argparse

# Fields to drop from each per_hour_data[time] dict
DROP_KEYS = {
    "agg.reversal_score",
    "cwt.r_close_CWT_power_hi",
    "fft.bb_bw_20_pct_mid",
    "fft.r_close_pct_hi",
    "fft.spread_vwap_norm_pct_hi",
    "fft.volume_z_by_tod_pct_mid",
    "inter.close",
    "inter.ema_12",
    "inter.ema_26",
    "inter.low",
    "inter.macd",
    "inter.macd_hist",
    "inter.macd_signal",
    "inter.open",
    "inter.r_close",
    "inter.sma_20",
    "inter.sma_50",
    "inter.stoch_d_14",
    "inter.stoch_k_14",
}

def process_file(in_path: str, out_path: str):
    """Load one JSON, drop features under per_hour_data, save to out_path."""
    with open(in_path, "r") as f:
        data = json.load(f)

    per_hour = data.get("per_hour_data", {})
    # per_hour is a dict: time_str -> feature dict
    for _, feat_dict in per_hour.items():
        if not isinstance(feat_dict, dict):
            continue
        for key in DROP_KEYS:
            feat_dict.pop(key, None)  # safely remove if present

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    ap = argparse.ArgumentParser(
        description="Drop selected features from per_hour_data for all JSONs in a root dir."
    )
    ap.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Root directory containing per-ticker subfolders with JSON files "
             "(e.g. /scratch/.../hourly_output_test)",
    )
    args = ap.parse_args()

    in_root = os.path.abspath(args.input_root)
    out_root = in_root.rstrip("/\\") + "_drop_feature"

    print(f"Input root : {in_root}")
    print(f"Output root: {out_root}")

    for dirpath, dirnames, filenames in os.walk(in_root):
        # Mirror directory structure under out_root
        rel_dir = os.path.relpath(dirpath, in_root)
        out_dir = os.path.join(out_root, rel_dir)

        for fname in filenames:
            if not fname.endswith(".json"):
                continue
            in_path = os.path.join(dirpath, fname)
            out_path = os.path.join(out_dir, fname)
            try:
                process_file(in_path, out_path)
                print(f"Processed {in_path} -> {out_path}")
            except Exception as e:
                print(f"[WARN] Failed to process {in_path}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
