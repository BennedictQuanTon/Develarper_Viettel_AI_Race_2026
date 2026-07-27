# Implementation Plan — Path to ERS 65+
**Viettel AI Race 2026 · Challenge 3 · Yoshio Branch**
*Date: 2026-07-27 · Based on: cross-branch diff + Issues #14 + #15*

---

## Problem Statement

Team ceiling is **62.01** (QuanTon `p7-oneshot`). TBT = 4ms is locked across all three branches after exhaustive CLI ablation. The root cause is the **3 vCPU MiG scheduler floor** — no serving flag reduces it further.

**ERS gap to 65:**
```
Current:  ERS ≈ 0.620  (TBT=4ms, TTFT≈48ms, fail=5)
Target:   ERS ≥ 0.650

Δ needed: +0.030 ERS = +3.0 pts

Levers:
  fail 5 → 0:       +1.2 pts   (recoverable via seqs tuning)
  TBT 4 → 3.5ms:    +2.0 pts   (requires source patch)
  TTFT 48 → 35ms:   +0.5 pts   (minor, already near ceiling value)
```

---

## Confirmed Dead Ends — Do Not Revisit

> [!CAUTION]
> The following have been tested across QuanTon (vLLM 0.26.0), Yoshio (0.25.1), and Tuong (patched). Results are final.

| Flag / Approach | Tested by | Result |
|---|---|---|
| `--optimization-level=3` | QuanTon x1b, Yoshio mamba_optlev, fp8_flash_opt | ≤ 60.34 — DEAD |
| `--compilation-config` CUDA graphs | QuanTon p9-decode | Regressed — DEAD |
| `--mamba-backend=CUDA` | Yoshio | exit 2 (missing kernel) — DEAD |
| `--speculative-config` n-gram | Yoshio, QuanTon x1 | exit 2 / fail — DEAD |
| `--max-num-partial-prefills≠1` | Tuong T4 | exit 1 NotImplementedError — DEAD |
| `--performance-mode=*` | QuanTon y1 | Regressed — DEAD |
| `--gpu-memory-utilization=0.97+` | Yoshio fp8_flash_opt, QuanTon p9 | Regressed — DEAD |
| OMP/TOKENIZERS env pinning | QuanTon Z3-ENV → 58.34 | DEAD |
| `mbt=256` | Yoshio bat_model_greed → 54.3 | DEAD |
| `mbt=512` vs `768` | QuanTon p7-mbt512 → 61.05 | 768 is superior |
| `--max-num-seqs=256` | Both branches | 128 is superior (p7 proven) |

---

## Phase 1 — Compose-Only Parameter Convergence

**Goal:** Close the 0.35pt gap between Yoshio (61.66) and QuanTon (62.01).
**No image rebuild. No new flags.**

### 1.1 Adopt Validated p7 Parameters

The only two parameters separating Yoshio fp8_flash (61.66) from QuanTon p7-oneshot (62.01):

```diff
# docker-compose.yml
- --max-num-batched-tokens=512
+ --max-num-batched-tokens=768

- --max-num-seqs=256
+ --max-num-seqs=128
```

**Rationale:** QuanTon's ablation proves `mbt=768` is the BTC-specific sweet spot — large enough to avoid excessive prefill fragmentation, small enough for meaningful decode interleaving. `seqs=128` reduces CPU scheduler overhead at peak (70 active conversations) by keeping the ready-queue smaller.

**Expected:** ~62.0 (parity with p7-oneshot on same image base).

#### [MODIFY] [docker-compose.yml](file:///d:/Dev/Projects/Viettel%20AI%20Race/Develarper_Viettel_AI_Race_2026/docker-compose.yml)
- `--max-num-batched-tokens`: `512` → `768`
- `--max-num-seqs`: `256` → `128`

#### [NEW] [docker_compose_p7_mirror.yaml](file:///d:/Dev/Projects/Viettel%20AI%20Race/Develarper_Viettel_AI_Race_2026/submit_yoshio/docker_compose_p7_mirror.yaml)
Archive of this submission for rollback.

---

### 1.2 Ablate `--max-num-seqs=70` (if 1.1 scores ≥ 62.0)

QuanTon used `seqs=128` but never tested `seqs=70` (exact conversation count). Fewer queued sequences = less scheduler contention at peak.

```diff
- --max-num-seqs=128
+ --max-num-seqs=70
```

**Expected:** +0.3-0.5 pts from reduced fail count (fail: 5 → 0-2).
**Risk:** If peak load briefly exceeds 70 active seqs, requests are rejected outright → fail count may increase. Monitor fail_count in result.
**Abort condition:** If fail_count > 5, revert to `seqs=128`.

---

### 1.3 Ablate `--dtype=float16` (only untested flag across all branches)

No branch has tested FP16 vs BF16. On H200 SM90, FP16 has marginally better hardware transpose support for small-batch matmuls (1.2B model dimensions are small).

```diff
- --dtype=bfloat16
+ --dtype=float16
```

**Expected:** +0.1-0.2 pts.
**Risk:** Minimal. FP16 + FP8 quantization is a stable combination on Hopper.
**Abort condition:** If post-online GPQA Δ > 0.08 on smoke test.

---

## Phase 2 — vLLM 0.26.0 Image Rebuild

**Goal:** Match QuanTon's base and unlock potential 0.26.0 scheduler improvements.

QuanTon confirmed 62.01 on vLLM **0.26.0** (`p7-oneshot`). Yoshio runs **0.25.1**. If Phase 1 scores < 62.0, the version delta is the cause.

### 2.1 Rebuild Base Image

```dockerfile
# Dockerfile change — single line
FROM vllm/vllm-openai:v0.26.0   # was: vllm/vllm-openai:v0.25.1 (or latest)
```

**Verify:** `--mamba-backend=flashinfer` and `--mamba-cache-mode=align` still work on 0.26.0 (QuanTon's p7 uses both — confirmed compatible).

**New image tag:** `asterios2707/develarper-agent:v0.26-p1`

#### [MODIFY] [Dockerfile](file:///d:/Dev/Projects/Viettel%20AI%20Race/Develarper_Viettel_AI_Race_2026/Dockerfile)
- Base image: update to `vllm/vllm-openai:v0.26.0`

**Expected:** +0 to +0.5 pts (version improvement is speculative; QuanTon's score is partly image-dependent).

---

## Phase 3 — Breaking TBT < 4ms (Source Patch)

**Goal:** Push TBT from 4ms to ≤ 3.5ms → +2.0 pts → reach 65+.

This is the only path that definitively breaks the CPU scheduler floor. All CLI paths are exhausted.

### 3.1 Profile the Scheduler Bottleneck

The TBT=4ms floor breaks down as:
```
~1-2ms: GPU kernel (FP8 decode matmul + FlashInfer SSM)
~2-3ms: CPU vLLM scheduler loop (schedule_running + output processing)
```

On a 3-vCPU MiG slice, the scheduler's async event loop and output processor share CPU time with the tokenizer and API server.

### 3.2 Scheduler Yield Patch

Target file: `vllm/core/scheduler.py`

The scheduler's `_schedule_running()` loop processes decode steps sequentially. The key optimization is reducing the number of Python-level iterations before yielding back to the GPU:

```python
# vllm/core/scheduler.py — _schedule_running()
# Current: processes all ready sequences before yielding
# Patch target: add early-exit when num_running_tokens >= decode_target

# Also target: vllm/engine/output_processor/single_step.py
# Output processor runs on CPU path between GPU decode steps
# Patch: defer non-critical output processing to batch end
```

**Implementation approach:**
1. Checkout Tuong's image framework (`tuong-opt-v1`) — he already has the patch infrastructure (`ENABLE_DEVELARPER_OPT`, `DEVELARPER_PAYLOADS`)
2. Add a `p02_scheduler_yield` payload targeting the above
3. Build new image: `asterios2707/develarper-agent:v0.26-p2`

**Expected:** TBT 4ms → 3.0-3.5ms → ERS +2.0 pts.
**Risk:** High — scheduler changes can cause starvation bugs under Poisson load. Requires local smoke test with ERS sim before portal submit.

---

## Submission Sequence

```
Phase 1.1 → mbt=768 + seqs=128           → expect ~62.0  (compose-only)
Phase 1.2 → seqs=70                       → expect ~62.3  (compose-only)
Phase 1.3 → dtype=float16                 → expect ~62.5  (compose-only)
Phase 2.1 → rebuild vLLM 0.26.0 base     → expect ~62.5  (new image)
Phase 3.2 → scheduler yield patch         → expect ~65+   (new image + patch)
```

---

## Verification Plan

### Automated (before each portal submit)

```bash
# 1. YAML parse check
docker compose -f docker-compose.yml config --quiet

# 2. Verify no dead-end flags present
grep -E "optimization-level|speculative|partial-prefill|performance-mode|CUDA" docker-compose.yml
# → must return empty

# 3. ERS sim projection (local)
python scripts/ers_sim.py --tbt 4 --ttft 48 --fail 5
```

### Portal Submit Criteria

- Submit if: projected ERS (from config diff) > current best (62.01)
- Abort if: fail_count increases from baseline (5) on same TTFT range
- Rollback to: QuanTon `p7-oneshot` or Yoshio `fp8_flash` as floor

### Accuracy Gate Safety

Keep `fp8_flash` (Yoshio, BF16 weights, online FP8) as a GPQA-safe candidate. Phase 3 source patch must be smoke-tested against GPQA before being designated as GPQA shortlist entry.

---

## Open Questions

1. **QuanTon's `p7-oneshot` image (`longquanton/develarper-lfm25:p7-oneshot`)** — is this based on vLLM 0.26.0 confirmed? If yes, does rebuilding `asterios2707` on 0.26.0 replicate exact behavior?

2. **fail=5 root cause** — are failures OOM (capacity), timeout (TTFT > ceiling), or decode errors? This determines whether `seqs=70` fixes them or exposes a different issue.

3. **Tuong's patch infrastructure** — is `DEVELARPER_PAYLOADS` hookable for Phase 3 without a full fork, or does it require integration into Tuong's image build system?
