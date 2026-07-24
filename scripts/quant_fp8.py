#!/usr/bin/env python3
"""Offline FP8 (llm-compressor FP8_DYNAMIC) for LFM2.5 → vLLM compressed-tensors.

Official path (vLLM docs): static per-channel weights + dynamic per-token activations.
Faster than runtime --quantization=fp8 on BF16 checkpoint.

Usage:
  # dry-run (no GPU)
  python3 scripts/quant_fp8.py --dry-run

  # real (needs GPU + pip install llmcompressor transformers)
  python3 scripts/quant_fp8.py \\
    --model model_weights/LFM2.5-1.2B-Instruct \\
    --out model_weights/LFM2.5-1.2B-Instruct-FP8

Then build: bash scripts/build_push_p6_combo.sh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_MODEL = Path("model_weights/LFM2.5-1.2B-Instruct")
DEFAULT_OUT = Path("model_weights/LFM2.5-1.2B-Instruct-FP8")


def build_recipe():
    from llmcompressor.modifiers.quantization import QuantizationModifier

    # Keep lm_head higher precision for Accuracy Gate stability.
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
    print("[quant] note: LFM2 hybrid conv/Mamba non-Linear stays native dtype")

    if dry_run:
        print("[quant] dry-run OK — skipping load / oneshot")
        (out_dir / "DRY_RUN.txt").write_text(
            f"Would quantize {model_id} -> {out_dir} with FP8_DYNAMIC, ignore lm_head\n"
        )
        return

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit("Missing transformers. pip install transformers") from e

    try:
        from llmcompressor import oneshot
    except ImportError:
        try:
            from llmcompressor.transformers import oneshot
        except ImportError as e:
            raise SystemExit("Missing llmcompressor. pip install llmcompressor") from e

    print("[quant] loading model (GPU recommended)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    recipe = build_recipe()

    print("[quant] running oneshot...")
    oneshot(model=model, recipe=recipe)

    print("[quant] saving compressed-tensors checkpoint...")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    # Preserve chat template if present next to original
    src = Path(model_id)
    for name in ("chat_template.jinja", "generation_config.json"):
        p = src / name
        if p.is_file():
            (out_dir / name).write_bytes(p.read_bytes())
    print("[quant] done →", out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="HF id or local path")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output dir for FP8 weights")
    ap.add_argument("--dry-run", action="store_true", help="Validate only; do not load model")
    args = ap.parse_args()

    if not args.dry_run:
        local = Path(args.model)
        if local.is_dir() and not (local / "config.json").is_file():
            print(f"Missing config.json under {local}", file=sys.stderr)
            raise SystemExit(2)

    run_quant(args.model, args.out, args.dry_run)


if __name__ == "__main__":
    main()
