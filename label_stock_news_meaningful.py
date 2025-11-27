#!/usr/bin/env python3
import os
import json
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple
import torch
from vllm import LLM, SamplingParams


PROMPT_TEMPLATE = """You are evaluating whether a piece of text is meaningful stock commentary.

Meaningful stock commentary:
- Says something specific about the stock, company, business conditions, market movement, or key events.
- Is not just boilerplate, spam, or generic marketing content.

Given the following text, output ONLY a single character:
1 if the text is meaningful stock commentary.
0 if the text is NOT meaningful stock commentary.

Text:
"{text}"

Answer (0 or 1 only):
"""


def build_prompts(
    root_dir: str,
) -> Tuple[List[str], List[Tuple[str, int]], Dict[str, int]]:
    """
    Walk root_dir, collect all news texts and build prompts.

    Returns:
        prompts: list of prompts for vLLM
        meta: list of (file_path, news_index) for each prompt
        news_counts: mapping file_path -> number of news items
    """
    prompts: List[str] = []
    meta: List[Tuple[str, int]] = []
    news_counts: Dict[str, int] = {}

    for ticker in sorted(os.listdir(root_dir)):
        ticker_dir = os.path.join(root_dir, ticker)
        if not os.path.isdir(ticker_dir):
            continue

        for fname in sorted(os.listdir(ticker_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(ticker_dir, fname)
            with open(fpath, "r") as f:
                try:
                    data = json.load(f)
                except Exception as e:
                    print(f"[WARN] Failed to load {fpath}: {e}")
                    continue

            news_list = data.get("news", [])
            news_counts[fpath] = len(news_list)

            for idx, item in enumerate(news_list):
                text = item.get("text", "").strip()
                # If there's no text at all, still send it through model (or treat as not meaningful)
                prompt = PROMPT_TEMPLATE.format(text=text.replace('"', '\\"'))
                prompts.append(prompt)
                meta.append((fpath, idx))

    return prompts, meta, news_counts


def run_vllm(
    model_name: str,
    prompts: List[str],
    batch_size: int = 8,
) -> List[int]:
    """
    Run vLLM on the prompts, returning a list of integer labels (0 or 1).
    """
    print(f"Loading model: {model_name}")
    llm = LLM(
        model=model_name, 
        tensor_parallel_size=2, 
        gpu_memory_utilization=0.9, 
        dtype=torch.bfloat16
    )

    sampling_params = SamplingParams(
        max_tokens=2,
        temperature=0.0,
        top_p=1.0,
        n=1,
    )

    labels: List[int] = []
    # vLLM can take the full list, but we'll chunk manually if desired
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        outputs = llm.generate(batch, sampling_params)
        for out in outputs:
            text = out.outputs[0].text.strip()
            # Parse first character as 0 or 1, default to 0 if something weird
            label_char = text[0] if text else "0"
            label = 1 if label_char == "1" else 0
            labels.append(label)

    assert len(labels) == len(prompts)
    return labels


def main(args):
    root_dir = args.root_dir
    out_report = args.non_meaningful_json
    meaningful_root = args.meaningful_output_dir

    os.makedirs(meaningful_root, exist_ok=True)

    # 1. Build prompts
    print(f"Scanning root dir: {root_dir}")
    prompts, meta, news_counts = build_prompts(root_dir)
    print(f"Total news items found: {len(prompts)}")

    if not prompts:
        print("No news items found; exiting.")
        return

    # 2. Run vLLM to get labels
    labels = run_vllm(args.model_name, prompts, batch_size=args.batch_size)

    # 3. Aggregate labels per file
    file_labels: Dict[str, List[int]] = defaultdict(list)
    for (fpath, _idx), lbl in zip(meta, labels):
        file_labels[fpath].append(lbl)

    # 4. Build JSON report of files with ANY not-meaningful news
    non_meaningful_summary = []
    for fpath, lbls in file_labels.items():
        non_mean_idxs = [i for i, l in enumerate(lbls) if l == 0]
        if non_mean_idxs:
            # store relative path to make it easier to read
            rel_path = os.path.relpath(fpath, root_dir)
            non_meaningful_summary.append(
                {
                    "file": rel_path,
                    "non_meaningful_indices": non_mean_idxs,
                    "num_news": len(lbls),
                }
            )

    with open(out_report, "w") as f:
        json.dump(non_meaningful_summary, f, indent=2)
    print(f"Wrote non-meaningful report to: {out_report}")

    # 5. Create mirrored folder with ONLY meaningful news items
    kept_files = 0
    kept_news = 0

    for fpath, lbls in file_labels.items():
        # Reload the original data
        with open(fpath, "r") as f:
            data = json.load(f)

        news_list = data.get("news", [])
        meaningful_news = [
            item for item, lbl in zip(news_list, lbls) if lbl == 1
        ]

        if not meaningful_news:
            # Skip files where nothing is meaningful
            continue

        data["news"] = meaningful_news

        # Mirror directory structure under meaningful_root
        rel_path = os.path.relpath(fpath, root_dir)
        out_fpath = os.path.join(meaningful_root, rel_path)
        os.makedirs(os.path.dirname(out_fpath), exist_ok=True)

        with open(out_fpath, "w") as f:
            json.dump(data, f, indent=2)

        kept_files += 1
        kept_news += len(meaningful_news)

    print(f"Created meaningful-only folder at: {meaningful_root}")
    print(f"Kept {kept_files} files with {kept_news} meaningful news items total.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label news texts as meaningful / not meaningful using vLLM."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/scratch/dkhasha1/bzhang90/data_mining_fall_2025/hourly_output_test",
        help="Root directory containing per-ticker subfolders with JSON files.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HF model name or local path to use with vLLM.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for vLLM generation.",
    )
    parser.add_argument(
        "--non_meaningful_json",
        type=str,
        default="non_meaningful_files_report.json",
        help="Output JSON path listing files with non-meaningful texts.",
    )
    parser.add_argument(
        "--meaningful_output_dir",
        type=str,
        default="hourly_output_test_meaningful",
        help="Root dir for filtered copies containing only meaningful news.",
    )

    args = parser.parse_args()
    main(args)
