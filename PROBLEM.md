# [VIETTEL AI RACE 2026] Challenge 3 — LLM Inference Optimization

**Problem update (incl. 18/07/2026):** fixed model, MiG H200 18GB slice, vLLM-only, published ERS parameters.

---

## 1. Task & Infrastructure

### Task

Deploy and optimize an **LLM inference server** for **LFM2.5-1.2B-Instruct** on a production-like **multi-turn** workload trace.

- **Online round:** maximize **ERS** (latency score). Accuracy Gate is **not** run per submission.
- **After online:** team selects up to **5** submissions → validity audit → full **GPQA Diamond** → final `Score = 100 × ERS × f(Δ)`.

### Trace fields

| Field | Meaning |
|---|---|
| `num_conversations` | Independent conversations running concurrently |
| `user_turns_per_conversation` | User turns per conversation |
| `total_request` | Total requests |
| `shared_system_prefix_tokens` | Shared system prefix across conversations |
| `per_conversation_prefix_tokens` | Per-conversation context (added to turn-1 input) |
| `new_user_tokens_per_turn` | New user prompt tokens each turn (turn 1 also includes both prefixes) |
| `output_tokens_per_turn_pinned` | Pinned output tokens per turn |
| `arrival` | Request arrival pattern |

### Evaluation environment

BTC auto-provisions **1 MiG instance** and benchmarks the team endpoint:

| Item | Value |
|---|---|
| Hardware | **1× MiG H200 — 18GB VRAM**, 3 CPU cores, 8GB RAM |
| OS / Driver | Ubuntu 24.04 LTS, NVIDIA driver 590.x (CUDA 13.x) |
| Model | `LiquidAI/LFM2.5-1.2B-Instruct` |
| Weights | https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct |

---

## 2. Scoring

### 2.1 ERS

```text
ERS = (1/N) × Σ S_request,i    ∈ [0, 1]
```

Where N is the total number of requests.

**Per-request score:**

```text
S_request =
  0                                           if error / timeout / 0 tokens
  w · s_ttft + (1 - w) · s_tpot               if success
```

**Latency components:**

```text
s_ttft = [ clamp( (C_ttft - TTFT)       / (C_ttft - F_ttft), 0, 1 ) ]^γ
s_tpot = [ clamp( (C_tpot - TPOT_mean)  / (C_tpot - F_tpot), 0, 1 ) ]^γ
```

| Symbol | Meaning | Value |
|---|---|---|
| F_ttft | TTFT floor | **10 ms** |
| C_ttft | TTFT ceiling | **400 ms** |
| F_tpot | TPOT floor | **1 ms** |
| C_tpot | TPOT ceiling | **10 ms** |
| γ | Exponent | **2** |
| w | TTFT weight | **0.5** |

### 2.2 Accuracy Gate (post-online)

Default BF16 GPQA baseline reference: **0.4**.

```text
Δ = Accuracy_baseline - Accuracy_submission
```

```text
f(Δ) =
  1.0                              if Δ ≤ 0.10
  1.0 - (Δ - 0.10) / 0.06          if 0.10 < Δ < 0.16
  0.0                              if Δ ≥ 0.16
```

### 2.3 Overall score

```text
Score = 100 × ERS × f(Δ)
```

Team score = best valid submission after audit + GPQA.

---

## 3. Allowed optimizations

**Required framework:** **vLLM only**.

Allowed directions: online quantization; PagedAttention; KV FP8/INT8; prefix / semantic caching; CPU/NVMe offload; continuous batching; speculative decoding; memory-aware scheduling; custom CUDA/Triton; FlashAttention/FlashInfer; CUDA Graphs.

---

## 4. Submission

1. Build a public **Docker Hub** image  
2. Submit **docker-compose.yml** on the portal  
3. BTC pulls image → MiG H200 → healthcheck → **ERS** benchmark  
4. Leaderboard by ERS; GPQA only after online ends  

**Baseline image:** `vllm/vllm-openai:v0.22.1`

Sample compose (keep this entrypoint form):

```yaml
services:
  model:
    image: vllm/vllm-openai:v0.22.1
    entrypoint:
      - python3
      - -m
      - vllm.entrypoints.openai.api_server
    command:
      - --model=/model
      - --served-model-name=LFM2.5-1.2B-Instruct
      - --host=0.0.0.0
      - --port=8000
      - --max-model-len=32768
      - --gpu-memory-utilization=0.95
      - --tensor-parallel-size=1
      - --enable-prefix-caching
    ports:
      - "8000:8000"
    shm_size: "2g"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## 5. Anti-cheating

Prohibited: pre-bake/hardcode; dual-path; metric gaming; outbound network calls; unauthorized tokenizer/weight tampering; swapping images after submit.

---

## 6. Tie-break & appeals

Close scores (≤1–3 pts): lower Δ → lower p95 TTFT → higher generation speed → earlier submit.

Appeals within **24 hours** of notification email / phase announcement.
