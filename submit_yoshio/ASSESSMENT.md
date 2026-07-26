# Yoshio Branch — Optimization Assessment
**Viettel AI Race 2026 · Challenge 3 · Team Develarper**
**Model:** `LiquidAI/LFM2.5-1.2B-Instruct` · **Slice:** MiG H200 18GB · **Engine:** vLLM only
**Image:** `asterios2707/develarper-agent:latest`

---

## Scoring Model (Quick Ref)

```
Score = 100 × ERS    where ERS = 0.5·s_ttft + 0.5·s_tpot

s_ttft = clamp((400 - TTFT_ms) / 390, 0, 1)^2      ceiling=400ms, floor=10ms
s_tpot = clamp((10  - TPOT_ms) / 9,   0, 1)^2      ceiling=10ms,  floor=1ms
```

> **TPOT is the dominant axis.** γ=2 means every 1ms improvement near the ceiling yields exponentially more points than TTFT gains.

| TPOT (ms) | s_tpot | ERS (est., TTFT~80ms) | Score |
|---|---|---|---|
| 6.0 (P0 baseline) | 0.20 | ~0.44 | ~44 |
| 3.2 (current est.) | 0.56 | ~0.62 | **~62** |
| 2.5 | 0.70 | ~0.69 | ~69 |
| 2.0 | 0.79 | ~0.73 | ~73 |

---

## Workload (BTC Fixed)

| Field | Value |
|---|---|
| Conversations × turns | 70 × 6 = **420 requests** |
| Shared system prefix | 1,000 tokens |
| Per-conv prefix | 1,000 tokens |
| New user tokens/turn | 150 |
| Output tokens/turn | 300 (pinned) |
| **Peak sequence length** | **4,700 tokens** (turn 6 end) |
| Arrival | Poisson, seed 42 |

---

## Submission History (Yoshio Branch)

| Date | Score | Config | Key diff vs prev |
|---|---|---|---|
| 22/07 07:30 | 46.79 | original | baseline (broken: no tp, maxlen=32768, mem=0.88) |
| 22/07 11:41 | 50.64 | early fixes | +tensor-parallel-size=1, +maxlen fix |
| 23/07 07:37 | 49.95 | bf16_safe | +mamba-cache-mode=align, +block-size=16, mem=0.96 |
| 23/07 07:57 | **Failed** | CUDA | +mamba-backend=CUDA → missing kernel in image |
| 23/07 08:07 | **Failed** | speculative | +n-gram speculative → SSM hybrid incompatible |
| 23/07 08:17 | 49.91 | mamba_optlev | +optimization-level=3 → no gain (warmup eats health timeout) |
| 24/07 07:20 | **55.66** | fp8_safe | **+quantization=fp8** → +5.7 pts |
| 24/07 07:45 | **61.66** | fp8_flash | **+mamba-backend=flashinfer, mbt=512, block-size=32** → +6.0 pts |

> QuanTon scored 61.18 (24/07 00:05) on separate branch. Their block-size=32 vs our 16 = ~0.5pt difference. Validated and adopted.

---

## Key Findings

### What Moved the Needle

| Finding | Impact | Notes |
|---|---|---|
| `--max-model-len: 32768 → 8192` | **+major** | Expanded concurrent seqs from 18 → 75 on 18GB. Biggest fix. |
| `--quantization=fp8` | **+5.7 pts** | H200 native FP8 tensor cores. Reduces per-token matmul cost → TPOT↓ |
| `--mamba-backend=flashinfer` | **+3-4 pts (est.)** | FlashInfer SSM kernels for the 63% convolutional layers. |
| `--max-num-batched-tokens: 2048 → 512` | **+2-3 pts (est.)** | Decode interleaves between every micro-chunk → TPOT variance↓ |
| `--block-size=32 (not 16)` | **+0.5 pts** | SSM state alignment with mamba-cache-mode=align benefits from 32. |
| `--mamba-cache-mode=align` | correctness | Required: prevents SSM state corruption across block boundaries with prefix caching. |

### What Failed / Flopped

| Flag | Result | Reason |
|---|---|---|
| `--mamba-backend=CUDA` | **exit 2** | `causal-conv1d` CUDA package not in vllm/vllm-openai:v0.25.1 |
| `--speculative-config` (n-gram) | **exit 2** | `Lfm2ForCausalLM` hybrid KV cache groups incompatible with speculative decoding |
| `--optimization-level=3` | **~0 gain** | CUDA graph warmup (~30-60s) hits BTC health timeout before benchmark |
| `--block-size=16` | **-0.5 pts** | Worse than 32 due to SSM block alignment requirements |

---

## LFM2.5 Architecture — Why It Matters

```
LFM2.5-1.2B-Instruct (16 layers total):
  ├── 10 × LIV Convolutional blocks  (63%)  → O(n), ZERO token-dependent KV cache
  └──  6 × GQA Attention blocks      (37%)  → standard paged KV cache

vLLM class: Lfm2ForCausalLM
SSM backend  (63%): --mamba-backend=flashinfer  ← validated best
Attn backend (37%): FLASHINFER (via VLLM_ATTENTION_BACKEND env in Dockerfile)
SSM alignment:      --mamba-cache-mode=align   ← required with prefix caching
```

> KV cache pressure is ~37% of what a full transformer needs at same param count.
> Peak KV token demand: 70 convs × 4700 tokens × 37% attn layers = dramatically less than pure transformer.

---

## Current Best Config (Score: 61.66)

```yaml
command:
  - --model=/model
  - --served-model-name=LFM2.5-1.2B-Instruct
  - --host=0.0.0.0
  - --port=8000
  - --quantization=fp8              # weight matmul speedup on H200
  - --dtype=bfloat16
  - --kv-cache-dtype=fp8            # KV cache memory halved
  - --tensor-parallel-size=1
  - --enable-prefix-caching
  - --enable-chunked-prefill
  - --mamba-cache-mode=align        # SSM state correctness with prefix cache
  - --mamba-backend=flashinfer      # FlashInfer SSM kernels (validated)
  - --max-num-batched-tokens=512    # aggressive decode interleaving
  - --block-size=32                 # SSM state alignment (empirically superior)
  - --gpu-memory-utilization=0.96
  - --max-model-len=5120            # covers peak 4700 with 9% margin (untested)
  - --max-num-seqs=256
```

---

## Next Ablation Queue (1 variable per submission)

| # | Ablation | Change | Expected ΔERS | Risk |
|---|---|---|---|---|
| **A** | maxlen=5120 scoring | current root compose | +1-2 pts | 420-token margin. Watch fail_count. |
| **B** | `mbt=256` | 512 → 256 | +2-3 pts | TTFT may reach ~150-200ms, still safe |
| **C** | A+B compound | mbt=256 + maxlen=5120 | +3-4 pts | After A and B individually scored |
| **D** | `gpu-mem=0.97` | 0.96 → 0.97 | +0.5 pts | OOM risk under peak 70-conv load |
| **E** | `--optimization-level=3` | Add flag to compose | +1-2 pts | **No rebuild needed** — compose-only flag. 30-60s startup warmup at runtime; only viable if BTC health timeout > 90s. Worth re-testing now image is cached on BTC infra. |

---

## Anti-Cheat Checklist (before every submit)

- [ ] Image `asterios2707/develarper-agent:latest` is **public**
- [ ] Entrypoint: `python3 -m vllm.entrypoints.openai.api_server`
- [ ] `--model=/model`, `--served-model-name=LFM2.5-1.2B-Instruct` present and unchanged
- [ ] No outbound network calls at runtime
- [ ] `/model` baked into image (no HF download on start)
- [ ] ≥600s since last submission
