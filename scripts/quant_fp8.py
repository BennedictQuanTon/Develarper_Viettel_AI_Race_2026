#!/usr/bin/env python3
"""FP8 W8A8 offline quant scaffold (run on AMD MI300X after BTC announces the model).

Dry-run locally:
  python scripts/quant_fp8.py --model Qwen/Qwen2.5-0.5B-Instruct --out /tmp/fp8 --dry-run

Real run (AMD Session A):
  python scripts/quant_fp8.py --model <BTC_MODEL_ID> --out /artifacts/LLM-FP8

Requires: transformers, llmcompressor (install on the cloud image).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_recipe():
    from llmcompressor.modifiers.quantization import QuantizationModifier

    # Keep lm_head in higher precision for distribution stability (Accuracy Gate).
    return QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],
    )


def run_quant(model_id: str, out_dir: Path, dry_run: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[quant] model={model_id}")
    print(f"[quant] out={out_dir}")
    print("[quant] scheme=FP8_DYNAMIC  ignore=['lm_head']")

    if dry_run:
        print("[quant] dry-run OK — skipping download / oneshot")
        (out_dir / "DRY_RUN.txt").write_text(
            f"Would quantize {model_id} -> {out_dir} with FP8_DYNAMIC, ignore lm_head\n"
        )
        return

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            "Missing transformers. Install on AMD session before running without --dry-run."
        ) from e

    try:
        from llmcompressor import oneshot
    except ImportError:
        try:
            from llmcompressor.transformers import oneshot  # older API
        except ImportError as e:
            raise SystemExit(
                "Missing llmcompressor. Install on AMD session before running without --dry-run."
            ) from e

    print("[quant] loading model (needs large GPU RAM)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    recipe = build_recipe()

    print("[quant] running oneshot...")
    oneshot(model=model, recipe=recipe)

    print("[quant] saving...")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("[quant] done")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF id or local path (BTC model)")
    ap.add_argument("--out", type=Path, required=True, help="Output directory for FP8 weights")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate CLI / write placeholder; do not load model",
    )
    args = ap.parse_args()

    if not args.dry_run and args.model.startswith("<"):
        print("Refusing to run: replace --model with the real BTC model id.", file=sys.stderr)
        raise SystemExit(2)

    run_quant(args.model, args.out, args.dry_run)


if __name__ == "__main__":
    main()
