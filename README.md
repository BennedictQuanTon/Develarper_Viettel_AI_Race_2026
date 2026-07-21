# Viettel AI Race 2026 — Challenge 3  
## Technical Report · Team Develarper

**Round:** Phase 1 (Sơ loại) · **Task:** LLM Inference Optimization  
**Model:** [`LiquidAI/LFM2.5-1.2B-Instruct`](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)  
**Eval hardware:** 1× MiG H200 (**18GB VRAM**, 3 CPU, 8GB RAM) · Ubuntu 24.04 · Driver 590.x / CUDA 13.x  
**Engine (required):** **vLLM only**  
**Submission artifact:** public Docker Hub image + Portal `docker-compose.yml`

```text
Score = 100 × ERS × f(Δ)
ERS params: TTFT 10–400 ms · TPOT 1–10 ms · γ=2 · w=0.5
GPQA baseline (ref): ~0.4 · Δ ≤ 0.10 → f(Δ)=1
```

---

## 1. Problem summary

Optimize serving of a fixed Instruct model under a **multi-turn production-like trace**:

| Workload field (BTC) | Value |
|---|---|
| Conversations × turns | **70 × 6 = 420** requests |
| Shared system prefix | **1000** tokens |
| Per-conversation prefix | **1000** tokens |
| New user tokens / turn | **150** |
| Output tokens / turn (pinned) | **300** |
| Arrival | Poisson, seed 42 |

Online leaderboard = **ERS only**. Accuracy Gate (GPQA) runs **after** the online round on ≤5 chosen submissions.

Full brief: [PROBLEM_VN.md](PROBLEM_VN.md) · [PROBLEM.md](PROBLEM.md)

---

## 2. What we built (deliverables)

| Deliverable | Detail |
|---|---|
| Competition image | `longquanton/develarper-lfm25:p0` (public Hub, **linux/amd64**) |
| Compose for Portal | [`docker-compose.submit.yml`](docker-compose.submit.yml) |
| Ablation variants | [`submit/`](submit/) — mem90 / chunked / fp8 / kv-fp8 |
| Weights bake | Model copied to `/model` at image build (runtime **offline**) |
| Local tooling | download, preflight, smoke, ERS simulator, set_image |
| Docs | CONTEXT (strategy), PLAN (ops), SUBMIT (ship), this README |

---

## 3. Stack & versions used

| Component | Choice | Why |
|---|---|---|
| Serving framework | **vLLM** | Required by rules |
| Base image | `vllm/vllm-openai:v0.23.0` | LFM2.5 needs **≥0.23** (`Lfm2ForCausalLM`); BTC sample `v0.22.1` is risky |
| Model | LFM2.5-1.2B-Instruct (~2.2GB BF16) | Official BTC model |
| Build platform | `linux/amd64` via Docker Buildx | Match MiG H200 (not Mac arm64) |
| Quant (P0) | None (BF16/default weights) | Stable first ERS; FP8 = later ablation |
| Prefix cache | **ON** | Shared 1000-token system prefix across 70 conversations |
| Memory util (P0) | `0.95` | Match BTC sample; 18GB slice |
| max-model-len (P0) | `32768` | Model context; may tune if KV pressure |

---

## 4. Architecture

```text
┌─────────────────┐     pull public image      ┌──────────────────────────┐
│  BTC Portal     │ ─────────────────────────► │  MiG H200 18GB           │
│  + compose.yml  │                            │  docker compose up       │
└─────────────────┘                            │  vLLM OpenAI API :8000   │
                                               │  weights at /model       │
                                               │  prefix cache + batching │
                                               └────────────┬─────────────┘
                                                            │ ERS (TTFT/TPOT)
                                                            ▼
                                                     Leaderboard
```

**P0 serve path (inside container):**

```text
python3 -m vllm.entrypoints.openai.api_server
  --model=/model
  --served-model-name=LFM2.5-1.2B-Instruct
  --host=0.0.0.0 --port=8000
  --max-model-len=32768
  --gpu-memory-utilization=0.95
  --tensor-parallel-size=1
  --enable-prefix-caching
```

Entrypoint form matches BTC sample (do **not** replace with `vllm serve` CLI if Portal expects this form).

---

## 5. Optimization strategy (priority order)

1. **Reliability** — no OOM / timeout / 0-token → each failure is `S_request = 0`
2. **Prefix caching** — exploit `shared_system_prefix_tokens=1000` + multi-turn history
3. **TPOT under load** — ceiling only **10 ms**; 300 output tokens × 420 requests
4. **TTFT** — floor 10 ms / ceiling 400 ms; γ=2 → steep penalty away from floor
5. **P1 ablations** (after P0 ERS exists): mem-util, chunked prefill, online FP8, KV-FP8

Strategy rationale: [CONTEXT.md](CONTEXT.md) · Day plan: [PLAN.md](PLAN.md)

---

## 6. What we did (work log)

1. Parsed updated problem (model fixed, MiG 18GB, ERS params published, vLLM-only).
2. Scaffolded repo: Dockerfile, compose, scripts, Makefile, eval sheets.
3. Downloaded Instruct weights locally (`model_weights/`, gitignored).
4. Built `develarper-lfm25:p0` on **vLLM 0.23.0** with `/model` baked; verified `config.json` / `Lfm2ForCausalLM`.
5. Pushed public image: **`longquanton/develarper-lfm25:p0`**.
6. Wired Portal compose to that tag; prepared ablation composes.
7. Ingested BTC workload meta → `eval/traces/btc_workload_meta.json`.
8. Ran local preflight + mock API smoke + ERS ranking simulator.

---

## 7. Checks & tests

| ID | Check / test | Environment | Result | Meaning |
|---|---|---|---|---|
| T1 | `scripts/preflight.sh` | Mac local | **PASS** | Weights path, compose form, scripts compile |
| T2 | Mock OpenAI streaming (`smoke_openai.py`) | CPU mock server | **PASS** | API contract / streaming client OK |
| T3 | `ers_sim.py` + BTC meta + official F/C/γ/w | Local ranking | **PASS** | Prefix ON helps vs OFF (`ERS_hat` ~0.38 vs ~0.22) — **not** official score |
| T4 | HF download LFM2.5-1.2B-Instruct | Local | **PASS** | ~2.2GB safetensors + tokenizer |
| T5 | Docker build bake `/model` | Buildx amd64 | **PASS** | Image contains model files |
| T6 | Hub push + public repo | Docker Hub | **PASS** | Tag `p0`, `is_private=false`, amd64 |
| T7 | Compose vs BTC sample fields | Review | **PASS** | entrypoint / model path / name / port / prefix / GPU |
| T8 | Live vLLM on MiG H200 | BTC infra | **Pending** | Measured only after Portal submit |
| T9 | Official ERS | Portal | **Pending** | Online leaderboard |
| T10 | GPQA / Δ | Post-online | **Pending** | Accuracy Gate |

**Important:** local `ERS_hat` is a **ranking signal only**. Official ERS is computed by BTC from real TTFT / TPOT_mean on the MiG slice.

---

## 8. How to reproduce / ship

```bash
# Build (needs Docker + downloaded weights)
make download-model
make build IMAGE_REPO=longquanton/develarper-lfm25 TAG=p0 VLLM_IMAGE=vllm/vllm-openai:v0.23.0

# Push
docker login
docker push longquanton/develarper-lfm25:p0
./scripts/set_image.sh longquanton/develarper-lfm25:p0

# Portal: upload docker-compose.submit.yml (Docker Compose file)
```

Ablation order (1 change per submit, 5/day, 600s gap): see [`submit/README.md`](submit/README.md) and [`eval/ablation_sheet.md`](eval/ablation_sheet.md).

More ops detail: [SUBMIT.md](SUBMIT.md)

---

## 9. Repository layout

```text
Dockerfile                     # bake /model on vLLM ≥0.23
docker-compose.submit.yml      # Portal P0 compose
docker-compose.yml             # local mock/gpu profiles
Makefile
configs/                       # ERS params + P0/P1 env sketches
scripts/                       # download, preflight, smoke, ers_sim, set_image
submit/                        # ablation composes
eval/traces/                   # BTC workload meta + examples
model_weights/                 # local only (gitignored)
CONTEXT.md  PLAN.md  PROBLEM*.md  SUBMIT.md
```

---

## 10. Team & compliance

- **Team:** Develarper (2× AI + 1× DevOps)
- **Anti-cheat:** offline runtime (no outbound fetch at serve), single behavior path, no pre-bake / dual-path
- **Hub image (P0):** https://hub.docker.com/r/longquanton/develarper-lfm25  
- **Tag:** `p0`

---

## 11. Status snapshot

| Item | Status |
|---|---|
| Code + docs + compose | Ready |
| Public Hub image amd64 | Ready (`longquanton/develarper-lfm25:p0`) |
| Portal compose file | Ready (`docker-compose.submit.yml`) |
| Official ERS / GPQA | After BTC evaluation |

*Last updated: 2026-07-21 · Phase 1 submission package*
