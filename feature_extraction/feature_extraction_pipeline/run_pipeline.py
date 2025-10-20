#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run the full feature-extraction pipeline:

1) load_data.py        (uses CFG_PATH inside the script, no args)
2) merge_data.py       --config <.../configs/data_preprocess_config.yaml>
3) AGG.py              --config <.../configs/data_preprocess_config.yaml>
4) FFT.py              --config <.../configs/data_preprocess_config.yaml>
5) CWT.py              --config <.../configs/data_preprocess_config.yaml>

Choose which to run via --steps (e.g., --steps merge,agg,fft,cwt).
"""

import sys
import subprocess
from pathlib import Path
import argparse
import yaml
from typing import List, Optional

# ---------- small utilities ----------

def here() -> Path:
    return Path(__file__).resolve().parent

def proj_root() -> Path:
    # feature_extraction_pipeline/  -> project root (one level up)
    return here().parent

def run(cmd: List[str], cwd: Optional[Path] = None):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)

def read_yaml(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def pick_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

def assert_dir_nonempty(path: Path, pattern: str = "*.csv", label: str = ""):
    files = list(path.glob(pattern))
    if not files:
        raise SystemExit(f"[ASSERT] No files matched {pattern} in {path} {label}".strip())
    print(f"✔ {label or path.name}: {len(files)} file(s) found.")

# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Run data pipeline (load → merge → AGG → FFT → CWT).")
    parser.add_argument(
        "--steps",
        default="load,merge,agg,fft,cwt",
        help="Comma-separated subset of steps to run: load,merge,agg,fft,cwt (default: all)"
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use (default: current interpreter)"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config (default: <project>/configs/data_preprocess_config.yaml)"
    )
    parser.add_argument(
        "--skip_checks",
        action="store_true",
        help="Skip sanity checks between steps."
    )
    args = parser.parse_args()

    steps = [s.strip().lower() for s in args.steps.split(",") if s.strip()]
    valid = {"load", "merge", "agg", "fft", "cwt"}
    if any(s not in valid for s in steps):
        raise SystemExit(f"--steps must be a subset of {sorted(valid)}")

    project = proj_root()

    # Paths
    config_path = Path(args.config) if args.config else project / "configs" / "data_preprocess_config.yaml"
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    cfg = read_yaml(config_path)
    gran = (cfg.get("date_range", {}) or {}).get("granularity", "minute").lower()
    if gran not in ("minute", "hour", "day"):
        raise SystemExit("date_range.granularity must be minute|hour|day")

    # Step scripts — prefer the copies inside feature_extraction_pipeline
    load_candidates = [
        here() / "load_data.py",                                   # preferred if present here
        project / "data_preprocessing_pipeline" / "load_data.py",  # fallback
    ]
    load_script = pick_first_existing(load_candidates)
    if load_script is None:
        raise SystemExit("Could not find load_data.py in expected locations.")

    merge_script = here() / "merge_data.py"
    if not merge_script.exists():
        raise SystemExit(f"merge_data.py not found at {merge_script}")

    agg_script = here() / "AGG.py"
    if "agg" in steps and not agg_script.exists():
        raise SystemExit(f"AGG.py not found at {agg_script}")

    fft_script = here() / "FFT.py"
    if "fft" in steps and not fft_script.exists():
        raise SystemExit(f"FFT.py not found at {fft_script}")

    cwt_script = here() / "CWT.py"
    if "cwt" in steps and not cwt_script.exists():
        raise SystemExit(f"CWT.py not found at {cwt_script}")

    print("=== Pipeline settings ===")
    print(f"Project root:      {project}")
    print(f"Runner dir:        {here()}")
    print(f"Config:            {config_path}")
    print(f"Granularity:       {gran}")
    print(f"Python exe:        {args.python}")
    print(f"Steps:             {steps}")
    print("=========================")

    # dynamic numbering
    total = len(steps)
    def banner(i, name): print(f"\n[{i}/{total}] Running {name} …")

    for i, step in enumerate(steps, 1):
        if step == "load":
            banner(i, "load_data.py")
            # NOTE: your load_data.py reads CFG_PATH internally; no need to pass args.
            run([args.python, str(load_script)], cwd=load_script.parent)

            if not args.skip_checks:
                out_root = Path(cfg.get("output_dir") or (project / "data_preprocessing_pipeline" / "output"))
                custom_dir = out_root / f"custom_indicators_{gran}"
                indi_dir   = out_root / f"indicators_{gran}"
                assert_dir_nonempty(custom_dir, "*.csv", label=f"custom_indicators_{gran}")
                assert_dir_nonempty(indi_dir, "*.csv", label=f"indicators_{gran}")

        elif step == "merge":
            banner(i, "merge_data.py")
            run([args.python, str(merge_script), "--config", str(config_path)], cwd=merge_script.parent)

            if not args.skip_checks:
                merged_dir_tpl = ((cfg.get("merge", {}) or {}).get("outputs", {}) or {}).get(
                    "merged_dir",
                    str(project / "feature_extraction" / "output" / "intermediate_dataset_{cad}")
                )
                merged_dir = Path(merged_dir_tpl.replace("{cad}", gran))
                assert_dir_nonempty(merged_dir, "*.csv", label=f"merged ({gran})")

        elif step == "agg":
            banner(i, "AGG.py")
            run([args.python, str(agg_script), "--config", str(config_path)], cwd=agg_script.parent)

            if not args.skip_checks:
                features_root = Path(((cfg.get("agg", {}) or {}).get("outputs", {}) or {}).get(
                    "features_root",
                    str(project / "feature_extraction" / "output" / "features")
                ))
                agg_dirs = list(features_root.glob(f"*/agg_{gran}"))
                if not agg_dirs:
                    raise SystemExit(f"[ASSERT] No folders like */agg_{gran} under {features_root}")
                # verify at least one expected cadence CSV exists
                cadences = list((((cfg.get("agg", {}) or {}).get("reporting", {}) or {}).get("cadences", [])) or ["day"])
                expected_files = [f"{c}.csv" for c in cadences]
                any_csv = False
                for d in agg_dirs:
                    for fname in expected_files:
                        if (d / fname).exists():
                            any_csv = True
                            break
                    if any_csv:
                        break
                if not any_csv:
                    raise SystemExit(f"[ASSERT] No expected AGG CSVs ({', '.join(expected_files)}) found in {features_root}/<TICKER>/agg_{gran}")
                print(f"✔ AGG outputs OK under {features_root} (granularity={gran}, cadences={cadences})")

        elif step == "fft":
            banner(i, "FFT.py")
            run([args.python, str(fft_script), "--config", str(config_path)], cwd=fft_script.parent)

            if not args.skip_checks:
                features_root = Path(((cfg.get("fft", {}) or {}).get("outputs", {}) or {}).get(
                    "features_root",
                    str(project / "feature_extraction" / "output" / "features")
                ))
                fft_dirs = list(features_root.glob(f"*/fft_{gran}"))
                if not fft_dirs:
                    raise SystemExit(f"[ASSERT] No folders like */fft_{gran} under {features_root}")
                any_csv = any(list(d.glob("*.csv")) for d in fft_dirs)
                if not any_csv:
                    raise SystemExit(f"[ASSERT] No CSVs found in {features_root}/<TICKER>/fft_{gran}")
                print(f"✔ FFT outputs OK under {features_root} (granularity={gran})")

        elif step == "cwt":
            banner(i, "CWT.py")
            run([args.python, str(cwt_script), "--config", str(config_path)], cwd=cwt_script.parent)

            if not args.skip_checks:
                features_root = Path(((cfg.get("cwt", {}) or {}).get("outputs", {}) or {}).get(
                    "features_root",
                    str(project / "feature_extraction" / "output" / "features")
                ))
                cwt_dirs = list(features_root.glob(f"*/cwt_{gran}"))
                if not cwt_dirs:
                    raise SystemExit(f"[ASSERT] No folders like */cwt_{gran} under {features_root}")
                any_csv = any(list(d.glob("*.csv")) for d in cwt_dirs)
                if not any_csv:
                    raise SystemExit(f"[ASSERT] No CSVs found in {features_root}/<TICKER>/cwt_{gran}")
                print(f"✔ CWT outputs OK under {features_root} (granularity={gran})")

        else:
            raise SystemExit(f"Unknown step: {step}")

    print("\n🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()
