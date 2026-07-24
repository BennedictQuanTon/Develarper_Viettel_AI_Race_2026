#!/usr/bin/env python3
"""Offline FP8 (llm-compressor FP8_DYNAMIC) for LFM2.5 → vLLM compressed-tensors.

Usage:
  # Mac / no CUDA (CPU, slower):
  .venv-p5/bin/python scripts/quant_fp8.py --device cpu

  # CUDA GPU:
  python scripts/quant_fp8.py

Then:
  bash scripts/build_push_p5_sf8.sh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_MODEL = Path("model_weights/LFM2.5-1.2B-Instruct")
DEFAULT_OUT = Path("model_weights/LFM2.5-1.2B-Instruct-FP8")


def build_recipe():
    from llmcompressor.modifiers.quantization import QuantizationModifier

    return QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],
    )


def run_quant(model_id: str, out_dir: Path, dry_run: bool, device: str = "auto") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[quant] model={model_id}")
    print(f"[quant] out={out_dir}")
    print("[quant] scheme=FP8_DYNAMIC  ignore=['lm_head']")

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

    import torch

    if device == "cpu" or (device == "auto" and not torch.cuda.is_available()):
        device_map, dtype = "cpu", torch.float32
        print("[quant] WARN: using CPU (no CUDA). Slow but OK for 1.2B on Mac.")
        # Apple MPS breaks llm-compressor; force CPU-only device map for dispatch.
        try:
            torch.backends.mps.is_available = lambda: False  # type: ignore[method-assign]
        except Exception:
            pass
        try:
            import compressed_tensors.offload.dispatch as _disp

            def _cpu_memory():
                # Report large host RAM so dispatch places modules on CPU.
                return {torch.device("cpu"): int(64 * 1024**3)}

            _disp.get_device_memory = _cpu_memory  # type: ignore[attr-defined]
        except Exception as e:
            print(f"[quant] WARN: could not patch device memory: {e}")
    else:
        device_map, dtype = "auto", "auto"

    print(f"[quant] loading model device_map={device_map}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map=device_map, torch_dtype=dtype, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    recipe = build_recipe()

    print("[quant] running oneshot...")
    oneshot(model=model, recipe=recipe)

    print("[quant] saving compressed-tensors checkpoint...")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    src = Path(model_id)
    for name in ("chat_template.jinja", "generation_config.json"):
        p = src / name
        if p.is_file():
            (out_dir / name).write_bytes(p.read_bytes())
    dry = out_dir / "DRY_RUN.txt"
    if dry.is_file():
        dry.unlink()
    print("[quant] done →", out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    if not args.dry_run:
        local = Path(args.model)
        if local.is_dir() and not (local / "config.json").is_file():
            print(f"Missing config.json under {local}", file=sys.stderr)
            raise SystemExit(2)

    run_quant(args.model, args.out, args.dry_run, device=args.device)


if __name__ == "__main__":
    main()
