# Yoshio Branch — Optimization Assessment
**Viettel AI Race 2026 · Challenge 3 · Team Develarper**
**Model:** `LiquidAI/LFM2.5-1.2B-Instruct` · **Slice:** MiG H200 18GB · **Engine:** vLLM only
**Image:** `asterios2707/develarper-agent:latest` (vLLM v0.25.1)
**Last updated:** 2026-07-29 19:11

---

## Scoring Model (Quick Ref)

```
Score = 100 × ERS    where ERS = 0.5·s_ttft + 0.5·s_tpot

s_ttft = clamp((400 - TTFT_ms) / 390, 0, 1)^2      ceiling=400ms, floor=10ms
s_tpot = clamp((10  - TPOT_ms) / 9,   0, 1)^2      ceiling=10ms,  floor=1ms
```

> **TPOT is the dominant axis.** γ=2 means every 1ms improvement near the ceiling yields exponentially more points.

| TPOT (ms) | s_tpot | ERS (est., TTFT~80ms) | Score |
|---|---|---|---|
| 6.0 (P0 baseline) | 0.20 | ~0.44 | ~44 |
| 4.0 (current floor) | 0.44 | ~0.62 | **~62** |
| 3.5 | 0.57 | ~0.69 | ~69 |
| 3.0 | 0.69 | ~0.75 | ~75 |

> **TBT = 4ms is a confirmed hard floor** across all three branches (Yoshio, QuanTon, Tuong). Root cause: 3 vCPU MiG scheduler overhead. No serving flag breaks it.

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

## Two-Track Strategy (as of 2026-07-29)

| Track | Goal | `--quantization=fp8` | Benchmark | Status |
|---|---|---|---|---|
| **Performance track** | Maximize ERS for leaderboard | ✅ Present | **61.66** (fp8_flash) | Branch best |
| **Accuracy-gate track** | GPQA candidate — zero Δ risk | ❌ Absent | **50.64** (early fixes) | In progress: 49.38 |

> The final 5 submissions on the last day must include a **no-quant BF16 candidate** for the accuracy gate (f(Δ) = 1 guaranteed). The root `docker-compose.yml` is currently being used to develop this track.
> **No-quant optimization direction is the OPPOSITE of fp8 track:** BF16 GPU is slower → need higher `mbt` and higher `seqs` for better GPU utilization.

---

## Full Submission History (Yoshio Branch)

| Date | Score | Archive File | Key flags vs prev | Verdict |
|---|---|---|---|---|
| 22/07 07:30 | 46.79 | *(original)* | baseline: no tp, maxlen=32768, mem=0.88 | Broken baseline |
| 22/07 11:41 | 50.64 | *(early fixes)* | +tp=1, mem fix | Partial fix |
| 23/07 07:37 | 49.95 | [bf16_safe](file:///d:/Dev/Projects/Viettel%20AI%20Race/Develarper_Viettel_AI_Race_2026/submit_yoshio/docker_compose_bf16_safe.yaml) | +mamba-cache-mode=align, +block-size=16, mem=0.96 | ✅ stable |
| 23/07 07:57 | **Failed** | [CUDA](file:///d:/Dev/Projects/Viettel%20AI%20Race/Develarper_Viettel_AI_Race_2026/submit_yoshio/docker_compose_CUDA.yaml) | +mamba-backend=CUDA | exit 2: missing kernel |
| 23/07 08:07 | **Failed** | [speculative](file:///d:/Dev/Projects/Viettel%20AI%20Race/Develarper_Viettel_AI_Race_2026/submit_yoshio/docker_compose_bf16_speculative.yaml) | +n-gram speculative | exit 2: SSM hybrid incompatible |
| 23/07 08:17 | 49.91 | [mamba_optlev](file:///d:/Dev/Projects/Viettel%20AI%20Race/Develarper_Viettel_AI_Race_2026/submit_yoshio/docker_compose_bf16_mamba_optlev.yaml) | +optimization-level=3 | ~0 gain; warmup hits health timeout |
| 24/07 07:20 | **55.66** | [fp8_safe](file:///d:/Dev/Projects/Viettel%20AI%20Race/Develarper_Viettel_AI_Race_2026/submit_yoshio/docker_compose_fp8_safe.yaml) | **+quantization=fp8** | +5.7 pts — major win |
| 24/07 07:45 | **61.66** 🏆 | [fp8_flash](file:///d:/Dev/Projects/Viettel%20AI%20Race/Develarper_Viettel_AI_Race_2026/submit_yoshio/docker_compose_fp8_flash.yaml) | **+mamba-backend=flashinfer, mbt=512, block-size=32** | +6.0 pts — branch BEST |
| 24/07 ~later | 61.08 | [fp8_model_greed](file:///d:/Dev/Projects/Viettel%20AI%20Race/Develarper_Viettel_AI_Race_2026/submit_yoshio/docker_compose_fp8_model_greed.yaml) | mbt=512, **maxlen=5120**, mem=0.97 | −0.58: maxlen reduction + mem↑ = regression |
| 24/07 ~later | 60.34 | [fp8_flash_opt](file:///d:/Dev/Projects/Viettel%20AI%20Race/Develarper_Viettel_AI_Race_2026/submit_yoshio/docker_compose_fp8_flash_opt.yaml) | +optimization-level=3, mem=0.97 | −1.32: O3 dead end confirmed again |
| 25/07 ~later | 57.94 | [1.1](file:///d:/Dev/Projects/Viettel%20AI%20Race/Develarper_Viettel_AI_Race_2026/submit_yoshio/docker_compose_1.1.yaml) | **mbt=768, seqs=128**, mem=0.97, +O3 | −3.72: O3 compound caused regression; mbt/seqs tested but polluted |
| ~later | 54.30 | [fp8_bat_model_greed](file:///d:/Dev/Projects/Viettel%20AI%20Race/Develarper_Viettel_AI_Race_2026/submit_yoshio/docker_compose_fp8_bat_model_greed.yaml) | mbt=256, maxlen=5120, mem=0.97 | −7.36: mbt=256 + maxlen compound — severe regression |
| ~later | **Failed** | [fp8_float16](file:///d:/Dev/Projects/Viettel%20AI%20Race/Develarper_Viettel_AI_Race_2026/submit_yoshio/docker_compose_fp8_float16.yaml) | --dtype=float16 | Pod failed to start: LFM2.5 incompatible with float16+fp8 |
| 29/07 | 46.73 | [bf16_flash](file:///d:/Dev/Projects/Viettel%20AI%20Race/Develarper_Viettel_AI_Race_2026/submit_yoshio/docker_compose_bf16_flash.yaml) | **no quant**, kv-fp8, mbt=512, seqs=256 | 🎯 accuracy-gate track; mbt=512 too aggressive for BF16 GPU |
| 29/07 | **49.38** | [bf16_flashv2](file:///d:/Dev/Projects/Viettel%20AI%20Race/Develarper_Viettel_AI_Race_2026/submit_yoshio/docker_compose_bf16_flashv2.yaml) | no quant, kv-fp8, mbt=1024, seqs=128 | 🎯 accuracy-gate; +2.65 vs v1 |
| 29/07 | 47.49 | [bf16_flashv3](file:///d:/Dev/Projects/Viettel%20AI%20Race/Develarper_Viettel_AI_Race_2026/submit_yoshio/docker_compose_bf16_flashv3.yaml) | no quant, kv-fp8, **mbt=2048, seqs=256**, chunked-prefill ON | 🎯 accuracy-gate; REGRESSION vs v2 — mbt↑ hurt, not helped |
| 29/07 | 47.93 | [bf16_noprefill](file:///d:/Dev/Projects/Viettel%20AI%20Race/Develarper_Viettel_AI_Race_2026/submit_yoshio/docker_compose_bf16_noprefill.yaml) | no quant, kv-fp8, mbt=2048, seqs=256, **no chunked-prefill** | 🎯 accuracy-gate; marginal +0.44 vs v3 but still well below 50.64 |
| 29/07 | **49.67** | [bf16_flashv4](file:///d:/Dev/Projects/Viettel%20AI%20Race/Develarper_Viettel_AI_Race_2026/submit_yoshio/docker_compose_bf16_flashv4.yaml) | no quant, kv-fp8, mbt=1024, **seqs=256**, chunked-prefill ON | 🎯 accuracy-gate **NEW BEST**; +0.29 vs v2 — 0.97 below 50.64 |
| *(unscored)* | — | [fp8_batch_greed](file:///d:/Dev/Projects/Viettel%20AI%20Race/Develarper_Viettel_AI_Race_2026/submit_yoshio/docker_compose_fp8_batch_greed.yaml) | mbt=256, mem=0.97 | archived only |

> **Performance track best: 61.66** (fp8_flash, 24/07 07:45)
> **Accuracy-gate track best: 49.67** (bf16_flashv4, 29/07) — 0.97 below 50.64 benchmark
> **Team best: 62.01** (QuanTon p7-oneshot, vLLM 0.26.0)

---

## Key Findings

### What Moved the Needle

| Finding | Impact | Notes |
|---|---|---|
| `--max-model-len: 32768 → 8192` | **+major** | Expanded concurrent seqs from 18 → 75. Biggest architectural fix. |
| `--quantization=fp8` | **+5.7 pts** | H200 native FP8 tensor cores. Core TPOT driver. Do not remove. |
| `--mamba-backend=flashinfer` | **+3-4 pts (est.)** | FlashInfer SSM kernels for the 63% convolutional LFM2.5 layers. |
| `--max-num-batched-tokens: 2048 → 512` | **+2-3 pts (est.)** | Decode interleaves between micro-chunks → TPOT variance↓ |
| `--block-size=32` (not 16) | **+0.5 pts** | SSM state alignment with mamba-cache-mode=align benefits from 32. |
| `--mamba-cache-mode=align` | correctness | Required: prevents SSM state corruption with prefix caching. |

### Confirmed Dead Ends (Do Not Retry)

| Flag / Approach | Result | Reason |
|---|---|---|
| `--mamba-backend=CUDA` | **exit 2** | `causal-conv1d` not in image |
| `--speculative-config` n-gram | **exit 2** | LFM2.5 SSM hybrid incompatible |
| `--optimization-level=3` | **60.34 / regression** | Warmup hits health timeout; confirmed dead on both branches |
| `--dtype=float16` | **Pod failed** | LFM2.5 incompatible with float16 + fp8 quantization |
| `mbt=256` | **54.30** | Severe regression; over-chunked prefill destroys TTFT |
| `mbt=768 + O3 compound` | **57.94** | O3 polluted the mbt=768 signal |
| `maxlen=5120 + mem=0.97` | **61.08** | Marginal regression vs fp8_flash |
| `--gpu-memory-utilization=0.97` | **regression** | Appears in all regressions; 0.96 is the ceiling |
| `--quantization=fp8` removed (performance track) | **46.73** | **Intentional for accuracy-gate track only.** 14+ pts gap is acceptable trade for f(Δ)=1 guarantee. |
| `mbt=512` on BF16 (no-quant) | **46.73** | Too aggressive — BF16 GPU can't process chunks fast enough. BF16 needs higher mbt. |
| `mbt=1024 + seqs=128` on BF16 | **49.38** | Good result. mbt=1024 confirmed sweet spot. |
| `mbt=2048 + seqs=256 + chunked-prefill` on BF16 | **47.49** | WORSE than 1024 — mbt peak is 1024 not 2048. |
| `mbt=2048 + seqs=256, no chunked-prefill` on BF16 | **47.93** | Marginal +0.44 vs v3. No-prefill alone not sufficient. |
| `mbt=1024 + seqs=256` on BF16 | **49.67** | **New no-quant best.** seqs=256 > seqs=128 at mbt=1024. |
| `mbt=256 + maxlen=5120 compound` | **54.30** | Double-greedy destroyed both axes |
| OMP env pinning | *(QuanTon Z3-ENV: 58.34)* | Dead |
| CUDA graphs / compilation-config | *(QuanTon p9: regressed)* | Dead |

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
float16 dtype:      INCOMPATIBLE with fp8 quantization on LFM2.5 — causes pod failure
```

---

## Current Best Config — fp8_flash (Score: 61.66)

```yaml
command:
  - --model=/model
  - --served-model-name=LFM2.5-1.2B-Instruct
  - --host=0.0.0.0
  - --port=8000
  - --quantization=fp8              # DO NOT REMOVE — 14+ pts at stake
  - --dtype=bfloat16                # DO NOT change to float16 — pod failure
  - --kv-cache-dtype=fp8
  - --tensor-parallel-size=1
  - --enable-prefix-caching
  - --enable-chunked-prefill
  - --mamba-cache-mode=align
  - --mamba-backend=flashinfer
  - --max-num-batched-tokens=512
  - --block-size=32
  - --gpu-memory-utilization=0.96   # DO NOT go above — all 0.97 submissions regressed
  - --max-model-len=8192
  - --max-num-seqs=256
```

---

## Root Compose (Current — Unscored)

The current `docker-compose.yml` has regressed from the best: `--quantization=fp8` is **missing**, `mbt=2048` (was 512). This matches `bf16_flash` config which scored **46.73**. Do not submit the root compose as-is.

---

## Remaining Untested Paths

### Performance Track (fp8)

| # | Ablation | Change | Expected | Risk | Status |
|---|---|---|---|---|---|
| A | `mbt=768 + seqs=128` **clean** (no O3) | Mirror QuanTon p7 exactly | ~62.0 | Low | **UNTESTED CLEANLY** |
| B | `seqs=70` | Exact conversation count | fail↓, +0.3-0.5 pts | Low | Untested |
| C | vLLM 0.26.0 base image | Rebuild on 0.26.0 | ~+0.5 pts | Medium | Untested |
| D | Scheduler source patch | `vllm/core/scheduler.py` yield reduction | TBT 4→3.5ms, +2-4 pts | High | Untested |

### Accuracy-Gate Track (bf16, no weight quant)

> **Finding update (29/07 19:11):** mbt=1024 is confirmed peak. seqs=256 > seqs=128 at this mbt (+0.29). Parameter space is now well-mapped.
> BF16 no-quant optimization law: **mbt=1024, seqs=256, chunked-prefill ON** = current ceiling at 49.67.

| # | Ablation | Change | Expected | Risk | Status |
|---|---|---|---|---|---|
| A | `mbt=2048 + seqs=256` | Higher GPU utilization | ~51-53 | Low | ✅ **47.49** — DEAD |
| B | No chunked-prefill | Fewer scheduler rounds | Unknown | Low | ✅ **47.93** — DEAD |
| C | `mbt=1024 + seqs=256` | Confirmed best combo | **49.67** | Low | ✅ **CURRENT BEST** |
| D | `mbt=768 + seqs=256` | Lower mbt probe | ~49-50? | Low | **Next to try** |
| E | Accept 49.67 as CLI ceiling | No further ablation | — | None | Likely fallback |

> **Benchmark to beat: 50.64.** Gap remaining: **0.97 pts.**
> If D also regresses, 49.67 is likely the CLI ceiling for BF16 on vLLM 0.25.1. The 50.64 benchmark may reflect pre-optimization BTC server state rather than a reproducible config advantage.

---

## Anti-Cheat Checklist (before every submit)

**Performance track:**
- [x] `--quantization=fp8` present
- [x] `--dtype=bfloat16` (NOT float16 — pod failure)
- [x] `--gpu-memory-utilization=0.96` (NOT 0.97+)

**Accuracy-gate track:**
- [x] `--quantization=fp8` **ABSENT** (required for f(Δ)=1)
- [x] `--kv-cache-dtype=fp8` present (memory savings, no accuracy impact)
- [x] `--dtype=bfloat16`
- [x] `--gpu-memory-utilization=0.96`

**Both tracks:**
- [x] Image `asterios2707/develarper-agent:latest` is **public**
- [x] Entrypoint: `python3 -m vllm.entrypoints.openai.api_server`
- [x] `--model=/model`, `--served-model-name=LFM2.5-1.2B-Instruct` present
- [x] No outbound network calls at runtime
- [x] ≥600s since last submission
