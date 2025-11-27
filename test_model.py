#!/usr/bin/env python3
import os
import json
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)


def build_prompt(ticker, date, per_hour_data):
    per_hour_str = json.dumps(per_hour_data, indent=2, sort_keys=True)
    prompt = (
        f"You are a financial market analyst.\n"
        f"Write a concise market commentary for {ticker} on {date} "
        f"based on the intraday technical indicators below.\n\n"
        f"<INTRADAY_DATA>\n"
        f"{per_hour_str}\n"
        f"</INTRADAY_DATA>\n\n"
        f"Commentary:\n"
    )
    return prompt


def load_tokenizer_and_gen_config(model_path: str,
                                  tokenizer_path: str = None,
                                  base_model_name: str = None):
    """
    Try to load tokenizer (and optionally generation config) in a robust way:

    1. If tokenizer_path is provided: try that first.
    2. Else, try model_path.
    3. If that fails and base_model_name is provided: load from base_model_name.
    """

    # 1. Decide where to load tokenizer from
    tried_paths = []

    if tokenizer_path is not None:
        tried_paths.append(tokenizer_path)
    tried_paths.append(model_path)

    # de-duplicate while preserving order
    seen = set()
    candidate_paths = []
    for p in tried_paths:
        if p not in seen and p is not None:
            seen.add(p)
            candidate_paths.append(p)

    tokenizer = None
    last_error = None

    # Try local paths / checkpoint dirs
    for path in candidate_paths:
        try:
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            print(f"[INFO] Loaded tokenizer from: {path}")
            break
        except Exception as e:
            print(f"[WARN] Failed to load tokenizer from {path}: {e}")
            last_error = e

    # Fallback: base model repo (e.g., 'Qwen/Qwen3-0.6B')
    if tokenizer is None and base_model_name is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            print(f"[INFO] Loaded tokenizer from base model: {base_model_name}")
        except Exception as e:
            print(f"[ERROR] Failed to load tokenizer from base model {base_model_name}: {e}")
            if last_error is not None:
                print("[DEBUG] Previous tokenizer load error:", last_error)
            raise RuntimeError("Could not load tokenizer from any source.")

    if tokenizer is None:
        raise RuntimeError(
            "Could not load tokenizer from model_path, tokenizer_path, or base_model_name. "
            "Please check your arguments."
        )

    # Make sure we have pad/eos tokens
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try to load generation config in a similarly forgiving way
    gen_config = None
    for path in candidate_paths:
        try:
            gen_config = GenerationConfig.from_pretrained(path)
            print(f"[INFO] Loaded GenerationConfig from: {path}")
            break
        except Exception:
            continue

    if gen_config is None and base_model_name is not None:
        try:
            gen_config = GenerationConfig.from_pretrained(base_model_name)
            print(f"[INFO] Loaded GenerationConfig from base model: {base_model_name}")
        except Exception:
            pass  # Not fatal; we can still pass kwargs directly to model.generate

    return tokenizer, gen_config


def main(args):
    # -----------------------------
    # Load tokenizer + (optionally) gen config
    # -----------------------------
    tokenizer, gen_config = load_tokenizer_and_gen_config(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        base_model_name=args.base_model_name,
    )

    # -----------------------------
    # Load finetuned model weights
    # -----------------------------
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    # Ensure pad_token_id is set on config to avoid warnings/issues
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # -----------------------------
    # Load sample JSON
    # -----------------------------
    with open(args.json_file, "r") as f:
        data = json.load(f)

    ticker = data["ticker"]
    date = data["date"]
    per_hour_data = data["per_hour_data"]

    prompt = build_prompt(ticker, date, per_hour_data)

    # -----------------------------
    # Tokenize
    # -----------------------------
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # -----------------------------
    # Prepare generate kwargs
    # -----------------------------
    generate_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # If we successfully loaded a GenerationConfig, let it provide defaults,
    # but override with our explicit args.
    if gen_config is not None:
        generate_kwargs = {**gen_config.to_dict(), **generate_kwargs}

    # -----------------------------
    # Generate output
    # -----------------------------
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generate_kwargs,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Try to strip the prompt part so we only print new content
    if decoded.startswith(prompt):
        generated_only = decoded[len(prompt):].strip()
    else:
        # fallback, if tokenization messed with spaces slightly
        generated_only = decoded

    print("==== GENERATED COMMENTARY ====")
    print(generated_only)
    print("==============================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to finetuned checkpoint directory (e.g., .../checkpoints or .../checkpoint-5000)",
    )
    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
        help="Path to JSON input file",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional path to directory containing tokenizer files "
             "(if different from model_path, e.g., final output_dir).",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Base model name used for finetuning; used as a fallback to load tokenizer/config.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="Max total input length for tokenization.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max number of new tokens to generate.",
    )

    args = parser.parse_args()
    main(args)
