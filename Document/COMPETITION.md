# Viettel AI Race 2026 — Competition Reference

**Vòng 1 – Sơ loại** · 02/07/2026 → 30/07/2026 · Challenge 3: LLM Inference Optimization

> Tài liệu gộp từ đề BTC, quy tắc chấm điểm, workload, nộp bài và checklist vận hành.  
> Chiến lược kỹ thuật đội: [TECHNICAL_STRATEGY.md](TECHNICAL_STRATEGY.md)

---

## 1. Nhiệm vụ

Triển khai và tối ưu một **LLM inference server** cho model **LiquidAI/LFM2.5-1.2B-Instruct**, xử lý workload trace **multi-turn** mô phỏng production traffic.

| Giai đoạn | Mục tiêu |
|---|---|
| **Online** | Tối đa hóa **ERS** (độ trễ). Không chạy Accuracy Gate mỗi lần nộp. |
| **Sau online** | Đội chọn tối đa **5** submissions → BTC hậu kiểm → **GPQA Diamond** full → chốt điểm |

**Model weights:** https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct

```text
Score = 100 × ERS × f(Δ)
```

---

## 2. Hạ tầng đánh giá

Benchmark chạy tự động trên hệ thống BTC. Đội serve endpoint trên **1 instance MiG**; BTC benchmark trực tiếp vào endpoint.

| Thành phần | Giá trị |
|---|---|
| GPU | 1× **MiG H200** |
| VRAM | **18 GB** |
| CPU | **3 Core** |
| RAM | **8 GB** |
| OS / Driver | Ubuntu 24.04 LTS, NVIDIA driver 590.x (CUDA 13.x) |
| Framework | **vLLM only** (bắt buộc) |
| Giới hạn nộp | **5 lần/ngày** |
| Timeout | **600 giây** (startup + warmup + healthcheck) |

> Toàn bộ benchmark chạy tự động trên hệ thống BTC. Thí sinh chỉ cần serve endpoint trên instance MiG được cấp phát.

---

## 3. Workload trace

### 3.1 Các trường mô tả

| Trường | Ý nghĩa |
|---|---|
| `num_conversations` | Số hội thoại độc lập chạy đồng thời |
| `user_turns_per_conversation` | Số lượt hỏi user mỗi hội thoại |
| `total_request` | Tổng số request (= conversations × turns) |
| `shared_system_prefix_tokens` | System prefix **giống nhau** trên các hội thoại |
| `per_conversation_prefix_tokens` | Ngữ cảnh riêng từng hội thoại (bổ sung input turn 1) |
| `new_user_tokens_per_turn` | Token prompt user mỗi turn (turn 1 có thêm 2 khối prefix) |
| `output_tokens_per_turn_pinned` | Số token output mỗi turn (pinned) |
| `arrival` | Nhịp đến của các request |

### 3.2 Giá trị workload (grading spec)

```json
{
  "num_conversations": 70,
  "user_turns_per_conversation": 6,
  "total_requests": 420,
  "shared_system_prefix_tokens": 1000,
  "per_conversation_prefix_tokens": 1000,
  "new_user_tokens_per_turn": 150,
  "output_tokens_per_turn_pinned": 300,
  "arrival": "Poisson, seed 42"
}
```

### 3.3 Context growth (peak turn 6 ≈ 4700 tokens)

| Turn | Input tokens | Output | Context sau turn |
|---|---|---|---|
| 1 | 1000 + 1000 + 150 = **2150** | 300 | 2450 |
| 2 | 2450 + 150 = **2600** | 300 | 2900 |
| 3 | 2900 + 150 = **3050** | 300 | 3350 |
| 4 | 3350 + 150 = **3500** | 300 | 3800 |
| 5 | 3800 + 150 = **3950** | 300 | 4250 |
| 6 | 4250 + 150 = **4400** | 300 | **4700 (PEAK)** |

**Insight:** Turn 1 nặng hơn vì cộng shared + per-conversation prefix. `--enable-prefix-caching` có lợi rõ rệt. `--max-model-len=32768` là lãng phí VRAM; peak chỉ ~4700.

---

## 4. Cách tính điểm

### 4.1 ERS — Online score

$$\text{ERS} = \frac{1}{N} \sum_{i=1}^{N} S_{\text{request},i} \in [0, 1]$$

Với $N = 420$ requests.

**Điểm từng request:**

$$S_{\text{request}} = \begin{cases} 0 & \text{lỗi / timeout / 0 token} \\ w \cdot s_{\text{ttft}} + (1-w) \cdot s_{\text{tpot}} & \text{thành công} \end{cases}$$

**Thành phần độ trễ:**

$$s_{\text{ttft}} = \left[\text{clamp}\left(\frac{C_{\text{ttft}} - \text{TTFT}}{C_{\text{ttft}} - F_{\text{ttft}}}, 0, 1\right)\right]^\gamma$$

$$s_{\text{tpot}} = \left[\text{clamp}\left(\frac{C_{\text{tpot}} - \text{TPOT}_{\text{mean}}}{C_{\text{tpot}} - F_{\text{tpot}}}, 0, 1\right)\right]^\gamma$$

| Ký hiệu | Ý nghĩa | Giá trị |
|---|---|---|
| $F_{\text{ttft}}$ | Floor TTFT | **10 ms** |
| $C_{\text{ttft}}$ | Ceiling TTFT | **400 ms** |
| $F_{\text{tpot}}$ | Floor TPOT | **1 ms** |
| $C_{\text{tpot}}$ | Ceiling TPOT | **10 ms** |
| $\gamma$ | Hệ số lũy thừa | **2** |
| $w$ | Trọng số TTFT | **0.5** |

**Giải thích thực tế:** $\gamma=2$ → cải thiện gần floor cho điểm nhiều hơn gần ceiling. TTFT < 10ms → $s_{\text{ttft}}=1$; TTFT > 400ms → 0. TPOT tương tự trong [1ms, 10ms].

| TPOT (ms) | $s_{\text{tpot}}$ | Đóng góp ERS (×0.5) |
|---|---|---|
| 1.0 (floor) | 1.000 | +0.500 |
| 2.0 | 0.790 | +0.395 |
| 4.0 | 0.444 | +0.222 |
| 6.0 | 0.198 | +0.099 |
| 10.0 (ceiling) | 0.000 | 0 |

### 4.2 Accuracy Gate — Post-online

Baseline BF16 GPQA tham chiếu: **~0.4**.

$$\Delta = \frac{\text{Accuracy}_{\text{baseline}} - \text{Accuracy}_{\text{submission}}}{\text{Accuracy}_{\text{baseline}}}$$

$$f(\Delta) = \begin{cases} 1.0 & \Delta \leq 0.10 \\ 1.0 - \dfrac{\Delta - 0.10}{0.06} & 0.10 < \Delta < 0.16 \\ 0.0 & \Delta \geq 0.16 \end{cases}$$

$$\text{Score}_{\text{final}} = 100 \times \text{ERS} \times f(\Delta)$$

**Ngưỡng an toàn:** Giữ $\Delta \leq 0.10$ để không bị phạt. $\Delta \geq 0.16$ → điểm = 0.

---

## 5. Không gian tối ưu (vLLM only)

**Bắt buộc:** chỉ dùng **vLLM**. Không TGI, SGLang, custom server.

| Nhóm | Hướng được phép |
|---|---|
| **Quantization** | Online Quantization (INT8, FP8, INT4…) — qua vLLM, không sửa weights trực tiếp |
| **KV Cache & Memory** | PagedAttention; KV FP8/INT8; Prefix / Semantic caching; CPU/NVMe offload |
| **Serving & Scheduling** | Continuous batching; Speculative decoding; Memory-aware scheduling |
| **System & Runtime** | Custom CUDA/Triton; FlashAttention / FlashInfer; CUDA Graphs; memory layout |

### Trade-off chính

```
Aggressive Quantization (tăng tốc) ←→ Accuracy drop ≤ 10% (ngưỡng an toàn)
```

Với 18GB VRAM + model 1.2B (~2.4GB BF16 / ~0.6GB FP8): không cần offload CPU; prefix caching rất hiệu quả; `tensor-parallel-size=1` (không multi-GPU).

---

## 6. Quy trình nộp bài

```
Develop & Optimize
       ↓
Package → Docker Image (weights baked tại /model)
       ↓
Push → Docker Hub (Public)
       ↓
Submit → docker-compose.yml lên Portal BTC
       ↓
BTC auto-pull → Deploy MiG H200 → Healthcheck → Benchmark ERS
       ↓
Leaderboard (ERS)
       ↓
[Sau online] Chọn ≤5 submissions → Hậu kiểm → GPQA → Score cuối
```

### 6.1 Docker baseline (BTC mẫu)

**Image tham chiếu:** `vllm/vllm-openai:v0.22.1`  
Digest: `sha256:55c9bcee9fc66644b139fddae8a7a03e4c0c8a25ab5c64b0ce614554a8abf5d5`

> **Lưu ý đội:** LFM2.5 cần vLLM **≥ 0.23.0**. Tag 0.22.1 có thể không load `Lfm2ForCausalLM`. Đội dùng v0.25.1 sau khi verify.

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

**Không được đổi:** entrypoint, `--model`, `--served-model-name`, `--host`, `--port`.

### 6.2 Checklist nộp bài (vận hành)

```bash
# 0) Đăng nhập Hub
docker login

# 1) Tải model
bash scripts/workflow.sh download-model

# 2) Build linux/amd64
bash scripts/workflow.sh build-tuong

# 3) Preflight
bash scripts/workflow.sh preflight

# 4) Push public
bash scripts/workflow.sh push-tuong

# 5) Copy compose để upload Portal
bash scripts/workflow.sh submit-compose
```

**Anti-cheat (bắt buộc):**

- [ ] Image **public** trên Docker Hub
- [ ] Weights baked tại `/model` — không download lúc runtime
- [ ] Không gọi mạng ngoài khi container chạy
- [ ] Entrypoint: `python3 -m vllm.entrypoints.openai.api_server`
- [ ] Prefix caching ON (trừ khi A/B chứng minh hại)
- [ ] Không dual-path / hardcode / tráo image sau nộp
- [ ] Boolean flags **không** gán `=true`

**ERS sim nhanh (không phải điểm thật):**

```bash
bash scripts/workflow.sh ers-sim
```

Chi tiết submit hiện tại: [../submit/BTC_SUBMISSION.md](../submit/BTC_SUBMISSION.md)

---

## 7. Quy định chống gian lận

| Vi phạm | Mô tả |
|---|---|
| Pre-bake / Hardcode | Cache cứng kết quả, không inference thật |
| Dual-path / Gaming | Phát hiện benchmark để cư xử khác |
| External network | Gọi API ngoài khi serve |
| Tokenizer/Weight tampering | Can thiệp trái phép weights/tokenizer |
| Image swapping | Tráo image sau khi submit |

BTC bảo lưu quyền hủy kết quả. Probe long-context: server phải trả đủ token, không truncate.

---

## 8. Tie-break & khiếu nại

**Khi điểm sát nhau (≤ 1–3 điểm), thứ tự ưu tiên:**

1. $\Delta$ thấp hơn
2. **p95 TTFT** thấp hơn
3. Tốc độ sinh văn bản (token/s) cao hơn
4. Nộp sớm hơn

**Hậu kiểm:** BTC có thể chấm lại và lấy điểm trung vị nhiều lần chạy.

**Khiếu nại:** Trong **24 giờ** sau email thông báo / công bố Phase.

---

## 9. Tóm tắt nhanh

| Mục | Giá trị |
|---|---|
| Loại bài nộp | Docker Compose |
| Hạ tầng chấm | MiG H200 18GB, 3 CPU |
| Framework | vLLM only |
| Giới hạn | 5 lần/ngày, timeout 600s |
| Submissions post-online | Tối đa 5 |
| Metrics target | TTFT floor 10ms · TPOT floor 1ms · $\Delta \leq 0.10$ |
