# [VIETTEL AI RACE 2026] Challenge 3 — LLM Inference Optimization

**Cập nhật đề (theo công bố BTC, incl. 18/07/2026):** model cố định, MiG H200 18GB, chỉ vLLM, tham số ERS công bố.

---

## 1. Nhiệm vụ & Hạ tầng

### Nhiệm vụ

Triển khai và tối ưu một **LLM inference server** cho mô hình **LFM2.5-1.2B-Instruct**, xử lý workload trace **multi-turn** mô phỏng traffic production.

- **Vòng online:** tối đa hoá **ERS** (điểm độ trễ). Không chạy Accuracy Gate mỗi lần nộp.
- **Sau vòng online:** đội chọn tối đa **5** submissions → BTC hậu kiểm hợp lệ → chạy **GPQA Diamond** full → chốt `Score = 100 × ERS × f(Δ)`.

### Trace workload (các trường mô tả)

| Trường | Ý nghĩa |
|---|---|
| `num_conversations` | Số hội thoại độc lập chạy đồng thời |
| `user_turns_per_conversation` | Số lượt hỏi user mỗi hội thoại |
| `total_request` | Tổng số request |
| `shared_system_prefix_tokens` | System prefix **giống nhau** trên các hội thoại |
| `per_conversation_prefix_tokens` | Ngữ cảnh riêng từng hội thoại (bổ sung input turn 1) |
| `new_user_tokens_per_turn` | Token prompt user mỗi turn (turn 1 có thêm 2 khối prefix) |
| `output_tokens_per_turn_pinned` | Số token output mỗi turn (pinned) |
| `arrival` | Nhịp đến của các request |

### Hạ tầng đánh giá

Benchmark chạy tự động trên hệ thống BTC. Đội serve endpoint trên **1 instance MiG**; BTC benchmark trực tiếp vào endpoint:

| Thành phần | Giá trị |
|---|---|
| Hardware | **1× MiG H200 — 18GB VRAM**, 3 Core CPU, 8GB RAM |
| OS / Driver | Ubuntu 24.04 LTS, NVIDIA driver 590.x (CUDA 13.x) |
| Model | `LiquidAI/LFM2.5-1.2B-Instruct` |
| Weights | https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct |

---

## 2. Cách tính điểm

### 2.1 ERS (Effective Request Score)

```text
ERS = (1/N) × Σ S_request,i    ∈ [0, 1]
```

Với N = tổng số request.

**Điểm từng request:**

```text
S_request =
  0                                           nếu lỗi / timeout / trả về 0 token
  w · s_ttft + (1 - w) · s_tpot               nếu xử lý thành công
```

**Thành phần độ trễ:**

```text
s_ttft = [ clamp( (C_ttft - TTFT)       / (C_ttft - F_ttft), 0, 1 ) ]^γ
s_tpot = [ clamp( (C_tpot - TPOT_mean)  / (C_tpot - F_tpot), 0, 1 ) ]^γ
```

**Tham số công bố:**

| Ký hiệu | Ý nghĩa | Giá trị |
|---|---|---|
| F_ttft | Floor TTFT | **10 ms** |
| C_ttft | Ceiling TTFT | **400 ms** |
| F_tpot | Floor TPOT | **1 ms** |
| C_tpot | Ceiling TPOT | **10 ms** |
| γ | Hệ số lũy thừa | **2** |
| w | Trọng số TTFT | **0.5** |

### 2.2 Accuracy Gate (sau vòng online)

Baseline BF16 GPQA mặc định tham chiếu: **0.4** (theo đề).

```text
Δ = Accuracy_baseline - Accuracy_submission
```

```text
f(Δ) =
  1.0                              nếu Δ ≤ 0.10
  1.0 - (Δ - 0.10) / 0.06          nếu 0.10 < Δ < 0.16
  0.0                              nếu Δ ≥ 0.16
```

### 2.3 Điểm tổng

```text
Score = 100 × ERS × f(Δ)
```

Điểm đội = Score tốt nhất trong các submission còn hợp lệ sau hậu kiểm + GPQA.

---

## 3. Không gian tối ưu

**Bắt buộc:** chỉ dùng serving framework **vLLM**.

Hướng được phép:

- **Quantization:** Online Quantization
- **KV Cache & Memory:** PagedAttention; KV quant FP8/INT8; Prefix / Semantic caching; Offload CPU/NVMe
- **Serving & Scheduling:** Continuous batching; Speculative decoding; Memory-aware scheduling
- **System & Runtime:** Custom CUDA/Triton; FlashAttention / FlashInfer; CUDA Graphs; memory layout

---

## 4. Nộp bài & Tài nguyên

1. Đóng gói **Docker Image** (public trên Docker Hub)
2. Nộp **docker-compose.yml** trên Portal (khai báo image + lệnh thực thi)
3. BTC pull image → dựng trên MiG H200 → healthcheck → benchmark **ERS**
4. Leaderboard theo ERS; sau online mới GPQA

**Baseline image tham chiếu:**  
`vllm/vllm-openai:v0.22.1`  
https://hub.docker.com/layers/vllm/vllm-openai/v0.22.1/images/sha256-55c9bcee9fc66644b139fddae8a7a03e4c0c8a25ab5c64b0ce614554a8abf5d5

**Mẫu docker-compose.yml (giữ entrypoint đúng form BTC):**

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

Cấm: pre-bake / hardcode; dual-path; gaming metric; gọi mạng ngoài; sửa tokenizer/weights trái phép; tráo image sau nộp.

Online chỉ ERS; sau vòng: chọn ≤5 bài → hậu kiểm → GPQA full (`lm_eval` / `bench-gpqa-diamond.sh`).

---

## 6. Tie-break & khiếu nại

Thứ tự khi sát điểm (≤1–3 điểm): (1) Δ thấp hơn (2) p95 TTFT thấp hơn (3) tốc độ sinh cao hơn (4) nộp sớm hơn.

Khiếu nại trong **24 giờ** sau email thông báo / công bố Phase.
