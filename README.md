# Viettel AI Race 2026 — Challenge 3  
## Technical Report · Team Develarper

**Round:** Phase 1 (Sơ loại) · **Task:** LLM Inference Optimization  
**Model:** [`LiquidAI/LFM2.5-1.2B-Instruct`](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)  
**Eval hardware:** 1× MiG H200 (**18GB VRAM**, 3 CPU, 8GB RAM) · Ubuntu 24.04 · Driver 590.x / CUDA 13.x  
**Engine (required):** **vLLM only**  
**Submission:** public Docker Hub image + Portal file **`docker-compose.yml`**

```text
Score = 100 × ERS × f(Δ)
ERS: TTFT 10–400 ms · TPOT 1–10 ms · γ=2 · w=0.5
GPQA baseline (ref): ~0.4 · Δ ≤ 0.10 → f(Δ)=1
```

| Submission | Hub image | Score / ERS | Status |
|---|---|---|---|
| **P0** (2026-07-21 13:31) | `longquanton/develarper-lfm25:p0` | **49.81** | **Champion** · GPQA candidate · root compose rollback |
| **S1S2** (maxlen 8192 + bt 2048) | same `p0` | **48.45** | **Rejected** (TBT vẫn 6ms, fail vẫn 7) |
| **B1** (next) | same `p0` | pending | P0 + `--quantization=fp8` — xem PLAN.md |

Docs: [PLAN.md](PLAN.md) · [CONTEXT.md](CONTEXT.md) · [SUBMIT.md](SUBMIT.md) · [PROBLEM_VN.md](PROBLEM_VN.md)

---

## 1. Problem summary

| Workload field (BTC) | Value |
|---|---|
| Conversations × turns | **70 × 6 = 420** requests |
| Shared system prefix | **1000** tokens |
| Per-conversation prefix | **1000** tokens |
| New user tokens / turn | **150** |
| Output tokens / turn (pinned) | **300** |
| Arrival | Poisson, seed 42 |
| Peak context (est. turn 6) | **~4400** tokens |

Online = **ERS only**. After online: choose ≤5 submissions → BTC hậu kiểm → GPQA full → final Score.

---

## 2. Official P0 results (Portal)

| Metric | Value | Reading |
|---|---|---|
| Score | **49.81** | ≈ 100 × ERS × 1 |
| ERS | **~0.498** | |
| failed_count | **7 / 420** | Each fail → request score 0 |
| ttft_p50 / p95 | **48 / 84 ms** | Comfortable vs ceiling 400 |
| tbt_median (TPOT proxy) | **6 ms** | **Main bottleneck** (ceiling 10, floor 1, γ=2) |
| accuracy_drop | 0 | Online display; GPQA later |

**ERS sensitivity (why we chase TPOT):** with TTFT≈48 ms fixed, TPOT 6→3 ms lifts estimated ERS ~0.50→~0.70. TTFT improvements yield far less.

---

## 3. What we optimized

### S1S2 (rejected — Score 48.45 < P0 49.81)

Tried `max-model-len=8192` + chunked + `max-num-batched-tokens=2048`. **TBT stayed 6ms, fails stayed 7**, TTFT slightly worse → scheduler/KV compound is **not** the lever. Archive only: `submit/docker-compose.s1s2_oneshot.yml`.

### Next: B1 FP8 (1 flag on P0 baseline)

| Knob | Value |
|---|---|
| Base | Identical to P0 |
| Add | `--quantization=fp8` |
| File | `submit/docker-compose.b1_fp8.yml` |

Rationale: same TBT on two BF16 configs ⇒ need **compute** path (Hopper FP8), not smaller batch token budget. Keep P0 for Accuracy Gate if FP8 wins ERS but hurts Δ.

Full plan: [PLAN.md](PLAN.md) § ★ SAU S1S2.

---

## 4. Deliverables

| Deliverable | Detail |
|---|---|
| Competition image | `longquanton/develarper-lfm25:p0` — public, **linux/amd64**, vLLM **0.23.0**, weights at `/model` |
| Portal compose (current) | [`docker-compose.yml`](docker-compose.yml) = **S1S2** |
| Ablation library | [`submit/`](submit/) — p0 archive, S1S2, mem90, chunked-only, fp8, kv-fp8 |
| Metrics log | [`eval/ablation_sheet.md`](eval/ablation_sheet.md) |
| Tooling | `Makefile`, download / preflight / smoke / `ers_sim` / `set_image` |

---

## 5. Stack

| Component | Choice | Why |
|---|---|---|
| Framework | **vLLM only** | Rules |
| Base image | `vllm/vllm-openai:v0.23.0` | LFM2.5 needs ≥0.23; sample `v0.22.1` may not load `Lfm2ForCausalLM` |
| Model path | `/model` (baked) | Offline runtime — no HF at serve |
| Entrypoint | `python3 -m vllm.entrypoints.openai.api_server` | BTC form (not `vllm-server`) |
| Build | Docker Buildx `linux/amd64` | Match MiG |

---

## 6. Architecture

```text
Portal upload docker-compose.yml
        │
        ▼
BTC pulls longquanton/develarper-lfm25:p0
        │
        ▼
MiG H200 18GB · vLLM OpenAI :8000 · /model
  · prefix cache · chunked prefill · maxlen 8192 · bt 2048
        │
        ▼
ERS (TTFT + TPOT) → leaderboard
(post-online) ≤5 picks → GPQA → Score
```

**Current serve command (S1S2):**

```text
python3 -m vllm.entrypoints.openai.api_server
  --model=/model
  --served-model-name=LFM2.5-1.2B-Instruct
  --host=0.0.0.0 --port=8000
  --max-model-len=8192
  --gpu-memory-utilization=0.95
  --tensor-parallel-size=1
  --enable-prefix-caching
  --enable-chunked-prefill
  --max-num-batched-tokens=2048
```

---

## 7. Optimization roadmap (post-S1S2)

Priority: **failed→0** then **TPOT→1–3 ms**, keep ≥1 BF16 digest for GPQA.

| Next (when daily quota resets) | Action |
|---|---|
| S1S2 ERS↑ & fail≤7 | Keep as BF16 best; try **online FP8** (`submit/docker-compose.b1_fp8.yml`) |
| Fail↑ / score↓ (maxlen?) | Soften to `max-model-len=12288` or restore P0 flags |
| TPOT still ≥5 ms | Image `p1` + `--optimization-level 3`, then n-gram speculative |
| Always | Do **not** overwrite Hub tag `p0`; use `p1`/`p2` for new images |

Full ranked levers: [PLAN.md](PLAN.md).

---

## 8. Work log

1. Parsed BTC brief (fixed model, MiG 18GB, vLLM-only, ERS params).
2. Scaffolded Dockerfile / compose / scripts / Makefile / eval sheets.
3. Downloaded Instruct weights; baked into Hub image `p0` (amd64, public).
4. Fixed Portal issues (filename must be `docker-compose.yml`; dropped invalid `--disable-log-requests` on v0.23).
5. **P0 scored 49.81** — diagnosed TPOT + fails as bottlenecks.
6. Researched compliant high-leverage knobs; updated PLAN for one remaining submit.
7. Shipped **S1S2** compose (maxlen 8192 + chunked + bt 2048); awaiting Portal metrics.

---

## 9. Checks & tests

| ID | Check | Env | Result |
|---|---|---|---|
| T1 | `scripts/preflight.sh` | Mac | PASS |
| T2 | Mock OpenAI smoke | CPU | PASS |
| T3 | `ers_sim.py` + BTC meta | Local ranking only | PASS (not official) |
| T4–T6 | Download / amd64 build / Hub public | Local + Hub | PASS |
| T7 | Compose vs BTC locked fields | Review | PASS (entrypoint + model/host/port) |
| T8 | Live MiG serve | BTC | **PASS** (P0) |
| T9 | Official ERS P0 | Portal | **49.81** |
| T10 | Official ERS S1S2 | Portal | **Pending** |
| T11 | GPQA / Δ | Post-online | Pending |

---

## 10. Reproduce / ship

```bash
make download-model
make build IMAGE_REPO=longquanton/develarper-lfm25 TAG=p0 VLLM_IMAGE=vllm/vllm-openai:v0.23.0
docker login && docker push longquanton/develarper-lfm25:p0

# Portal: upload root docker-compose.yml (current = S1S2)
# Limits: ≤5 submits/day · ≥600s between submits
```

New image tags only when flags need bake (`p1`+). Ablation sheet: [`eval/ablation_sheet.md`](eval/ablation_sheet.md).

---

## 11. Repository layout

```text
Dockerfile
docker-compose.yml              # Portal current = S1S2
submit/
  docker-compose.p0_baseline.yml
  docker-compose.s1s2_oneshot.yml
  docker-compose.a*.yml / b*.yml
eval/ablation_sheet.md
eval/traces/btc_workload_meta.json
scripts/  configs/  Makefile
CONTEXT.md  PLAN.md  SUBMIT.md  PROBLEM*.md
model_weights/                  # gitignored
```

---

## 12. Compliance (anti-cheat)

| Rule | Our stance |
|---|---|
| Honest GPU serving | Real vLLM decode on BTC MiG |
| No pre-bake / dual-path | Single OpenAI path |
| No outbound network at runtime | Weights in `/model` |
| No illicit tokenizer/weight tampering | Stock Instruct checkpoint |
| No image swap after submit | Keep `p0` immutable; new work → new tags |
| Locked compose fields | Entrypoint + `--model` / served name / host / port unchanged |

Tuning `max-model-len`, chunked prefill, batched-tokens = allowed engine optimization (same class as sample prefix/mem flags).

---

## 13. Status snapshot

| Item | Status |
|---|---|
| Hub image `p0` (BF16, amd64, public) | Ready · **do not overwrite** |
| P0 official Score | **49.81** |
| Portal compose S1S2 | Ready · metrics **pending** |
| Next ablations (FP8 / `-O3` / …) | After S1S2 readout + new daily quota |
| Accuracy Gate shortlist | Keep P0 BF16 (+ best ERS if Δ-safe) |

*Last updated: 2026-07-21 · after P0 score + S1S2 one-shot ship*
