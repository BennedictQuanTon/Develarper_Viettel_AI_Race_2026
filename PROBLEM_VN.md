# [VIETTEL AI RACE 2026] Challenge 3 — LLM Inference Optimization

**Cập nhật đề chính thức:** Model cố định `LiquidAI/LFM2.5-1.2B-Instruct`, 1 instance MiG H200 18GB, chỉ dùng vLLM, tham số ERS công bố.

---

## 1. Nhiệm vụ & Hạ tầng

### Nhiệm vụ (Cập nhật ngày 18/07/2026)

Triển khai và tối ưu một **LLM inference server** cho mô hình **LFM2.5-1.2B-Instruct** xử lý một workload trace **multi-turn** mô phỏng traffic production. Trong vòng online, mục tiêu là tối đa hoá **ERS** (điểm độ trễ). Accuracy Gate chỉ chạy sau khi kết thúc vòng online, trên tối đa 5 submissions đội tự chọn.

### Mô tả các giá trị trong file trace mô tả workload

| Trường | Ý nghĩa |
|---|---|
| `num_conversations` | Số hội thoại độc lập chạy đồng thời |
| `user_turns_per_conversation` | Số lượt hỏi của user trên mỗi hội thoại |
| `total_request` | Tổng số request |
| `shared_system_prefix_tokens` | System prefix, giống nhau trên các hội thoại |
| `per_conversation_prefix_tokens` | Ngữ cảnh riêng cho từng hội thoại (bổ sung input cho turn 1 của từng hội thoại) |
| `new_user_tokens_per_turn` | Số lượng token prompt của user tại mỗi turn (turn 1 có thêm 2 khối prefix) |
| `output_tokens_per_turn_pinned` | Số lượng token output tại mỗi turn |
| `arrival` | Nhịp đến của các request |

### Hạ tầng & Môi trường đánh giá

Toàn bộ quá trình chạy benchmark được thực hiện tự động trên hệ thống của BTC. Thí sinh sẽ serve endpoint trên 1 instance MiG và BTC sẽ thực hiện benchmark trực tiếp vào endpoint đó:

- **Hạ tầng Hardware:** 1 instance **MiG H200 (18GB VRAM**, 3 Core CPU, 8GB RAM) được cấp phát tự động cho mỗi lượt chấm.
- **Hệ điều hành & Driver (host):** Ubuntu 24.04 LTS, NVIDIA driver 590.x (hỗ trợ CUDA 13.x).
- **Model:** `LiquidAI/LFM2.5-1.2B-Instruct`
- **Weights:** https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct

---

## 2. Cách tính điểm

Effective Request Score được đánh giá dựa theo tốc độ trên 2 metrics TTFT và TPOT. Công thức cụ thể như sau:

$$\text{ERS} = \frac{1}{N} \sum_{i=1}^{N} S_{\text{request}, i} \quad \in [0, 1]$$

với $N$ là tổng số request.

Trong đó:

$$S_{\text{request}} = \begin{cases} 0 & \text{nếu lỗi / timeout / trả về 0 token} \\ w \cdot s_{\text{ttft}} + (1 - w) \cdot s_{\text{tpot}} & \text{nếu xử lý thành công} \end{cases}$$

$$s_{\text{ttft}} = (x_{\text{ttft}})^\gamma = \left[ \text{clamp}\left( \frac{C_{\text{ttft}} - \text{TTFT}}{C_{\text{ttft}} - F_{\text{ttft}}}, 0, 1 \right) \right]^\gamma$$

và

$$s_{\text{tpot}} = (x_{\text{tpot}})^\gamma = \left[ \text{clamp}\left( \frac{C_{\text{tpot}} - \text{TPOT}_{\text{mean}}}{C_{\text{tpot}} - F_{\text{tpot}}}, 0, 1 \right) \right]^\gamma$$

### Tham số cấu hình công bố:

| Ký hiệu | Ý nghĩa | Giá trị |
|---|---|---|
| $F_{\text{ttft}}$ | Floor của TTFT | **10 ms** |
| $C_{\text{ttft}}$ | Ceiling của TTFT | **400 ms** |
| $F_{\text{tpot}}$ | Floor của TPOT | **1 ms** |
| $C_{\text{tpot}}$ | Ceiling của TPOT | **10 ms** |
| $\gamma$ | Hệ số lũy thừa | **2** |
| $w$ | Trọng số của TTFT | **0.5** |

### Accuracy Gate - sau vòng online:

Không chấm GPQA trên từng lượt nộp online. Sau khi vòng online kết thúc, đội chọn thủ công tối đa 5 submissions tốt nhất. BTC lần lượt: (1) hậu kiểm tính hợp lệ phương án; (2) dựng endpoint và chạy GPQA full. Độ sụt giảm chất lượng ($\Delta$) so với baseline BF16 (mặc định 0.4):

$$\Delta = \text{Accuracy}_{\text{baseline}} - \text{Accuracy}_{\text{submission}}$$

(Trong đó, $\text{Accuracy}_{\text{baseline}}$ là accuracy tham chiếu của mô hình gốc chạy bằng trọng số BF16; $\text{Accuracy}_{\text{submission}}$ là accuracy bài nộp của đội.)

Dựa trên $\Delta$, áp dụng $f(\Delta)$:

$$f(\Delta) = \begin{cases} 1.0 & \text{nếu } \Delta \le 0.10 \\ 1.0 - \frac{\Delta - 0.10}{0.06} & \text{nếu } 0.10 < \Delta < 0.16 \\ 0.0 & \text{nếu } \Delta \ge 0.16 \end{cases}$$

Điểm cuối mỗi submission hợp lệ:

$$\text{Score} = 100 \times \text{ERS} \times f(\Delta)$$

(ERS lấy từ lần chấm online của đúng bài đó). Điểm đội = Score tốt nhất trong các bài còn hợp lệ.

Trong đó:
- **ERS:** Điểm trung bình hiệu năng trên trace, chấm trong vòng online.
- $f(\Delta)$: Hệ số phạt accuracy, chỉ có sau bước GPQA post-online.

---

## 3. Không gian Tối ưu

Thí sinh chỉ được phép sử dụng serving framework **vLLM** cho bài thi này. Các hướng tiếp cận bao gồm:

- **Quantization:** Các kỹ thuật Online Quantization.
- **KV Cache & Memory:** Tối đa hóa lượng request xử lý đồng thời bằng Paged Attention; KV cache quantization (FP8, INT8); Prefix caching và Semantic caching; Offloading xuống CPU/NVMe.
- **Serving & Scheduling:** Ứng dụng Dynamic/Continuous batching; Speculative decoding; Memory-aware scheduling.
- **System & Runtime:** Viết custom CUDA/Triton kernels; Tích hợp Fused attention kernels (FlashAttention, FlashInfer); Tối ưu hóa memory layout và CUDA Graphs.

---

## 4. Nộp bài & Tài nguyên

### Quy trình thực hiện:

1. **Develop & Package:** Thí sinh phát triển code giải pháp, tối ưu hệ thống và đóng gói toàn bộ thành một Docker Image.
2. **Push Image:** Đẩy (Push) Docker Image hoàn chỉnh lên Docker Hub cá nhân hoặc tổ chức dưới dạng công khai (Public).
3. **Submit:** Thí sinh truy cập hệ thống Portal của BTC, gửi file cấu hình `docker-compose.yml` (trong đó có khai báo chính xác đường dẫn Image trên Docker Hub và lệnh thực thi).
4. **Automated Evaluation:** Hệ thống tự động pull Image, dựng container trên MiG H200, healthcheck và chạy benchmark ERS (không chạy GPQA trên mỗi lượt nộp).
5. **Leaderboard:** Cập nhật theo ERS.
6. **Sau vòng online:** Đội chọn tối đa 5 submissions $\rightarrow$ BTC hậu kiểm hợp lệ $\rightarrow$ chấm GPQA full (`lm_eval` / `bench-gpqa-diamond.sh`) $\rightarrow$ chốt Score.

**Docker image baseline:**  
`vllm/vllm-openai:v0.22.1`  
https://hub.docker.com/layers/vllm/vllm-openai/v0.22.1/images/sha256-55c9bcee9fc66644b139fddae8a7a03e4c0c8a25ab5c64b0ce614554a8abf5d5

### File `docker-compose.yml` mẫu:

```yaml
services:
  model:
    image: vllm/vllm-openai:v0.22.1
    entrypoint:
      - python3 #Don't change this to vllm-server
      - -m  #Don't change this to vllm-server
      - vllm.entrypoints.openai.api_server #Don't change this to vllm-server
    command:
      - --model=/model #Don't change this to vllm-server
      - --served-model-name=LFM2.5-1.2B-Instruct #Don't change this to vllm-server
      - --host=0.0.0.0 #Don't change this to vllm-server
      - --port=8000 #Don't change this to vllm-server
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

## 5. Quy định & Phòng chống gian lận

- **Quy định chung:** Phase 1 áp dụng nghiêm ngặt các nguyên tắc Anti-Cheating đã được quy định tại tab Tổng quan.
- **Yêu cầu cốt lõi:** Giải pháp của các đội phải thực hiện serving LLM một cách trung thực trên tài nguyên GPU được BTC cấp phát.

### Các hành vi bị nghiêm cấm:

1. Pre-bake, hardcode kết quả, cơ chế dual-path hoặc lách luật (gaming) phương pháp đo lường.
2. Thực hiện các lệnh gọi mạng bên ngoài.
3. Can thiệp trái phép vào tokenizer hoặc weights của mô hình.
4. Tráo đổi Docker image sau khi đã nộp bài.

### Quy trình chấm điểm & Hậu kiểm:

- **Trong vòng Online:** Hệ thống sẽ chỉ chấm điểm tự động dựa trên chỉ số ERS.
- **Sau vòng thi:** Mỗi đội được quyền chọn ra tối đa 5 submissions. BTC sẽ tiến hành hậu kiểm tính hợp lệ của các bài nộp này, sau đó mới chạy đánh giá toàn diện trên tập dữ liệu GPQA full.
- **Xử lý vi phạm:** BTC bảo lưu quyền hủy bỏ kết quả đối với bất kỳ bài thi nào có dấu hiệu gian lận. Quyết định xử lý sẽ được thông báo chính thức đến đội thi qua email.

---

## 6. Hậu kiểm, Tie-break & Khiếu nại

### Tiêu chí phụ:

Trong trường hợp các đội có điểm số bám sát nhau (nằm trong biên độ nhiễu đo lường $\le$ 1–3 điểm), BTC sẽ phân định thứ hạng dựa trên các tiêu chí phụ theo thứ tự ưu tiên sau:

1. Mức độ suy giảm độ chính xác.
2. Chỉ số p95 TTFT.
3. Tốc độ sinh văn bản.
4. Thời điểm nộp bài (ưu tiên bài nộp sớm hơn).

### Quy trình Hậu kiểm & Chấm lại:

- BTC ưu tiên tiến hành hậu kiểm kỹ lưỡng đối với các cặp đội có tính cạnh tranh cao, đặc biệt là nhóm tranh chấp giải thưởng.
- BTC có quyền chấm lại và lấy điểm trung vị của các lần chạy trên Docker image đã được chốt để đảm bảo tính công bằng.

### Quy định Khiếu nại:

- Trước khi chốt bảng xếp hạng chung cuộc, BTC sẽ gửi email thông báo kết quả dự kiến.
- Mọi khiếu nại phải được gửi về BTC trong thời hạn tối đa **24 giờ** kể từ thời điểm nhận được email thông báo hoặc sau khi kết quả của Phase được công bố chính thức.
