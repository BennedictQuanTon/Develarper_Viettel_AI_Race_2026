# 🏆 BÁO CÁO TOÀN DIỆN DỰ ÁN VIETTEL AI RACE 2026 — CHALLENGE 3
## LLM Inference Optimization · Team Develarper · Author: Long Quân Tôn (QuanTon)

> **Cập nhật:** Sửa hoàn toàn lỗi hiển thị công thức LaTeX, bổ sung phân tích chuyên sâu file `grading-workload-spec.json`, phân tích dòng chảy token qua 6 turns, toán học ERS và chiến lược tối ưu CPU 3-core.

---

## 1. Phân Tích Chuyên Sâu Đề Bài & File `grading-workload-spec.json`

File `grading-workload-spec.json` xác định chính xác 100% kịch bản benchmark mà BTC chạy trên server:

```json
{
  "workload": "vLLM multi-turn (shared prefix + growing context, output length pinned)",
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

### 1.1 Chi Tiết Thông Số Workload
* **Tổng số Request:** 70 hội thoại $\times$ 6 lượt hỏi = **420 requests**.
* **Tổng số Token Output sinh ra:** 420 requests $\times$ 300 tokens/request = **126.000 tokens output**.
* **Shared System Prefix:** **1.000 tokens** (Dùng chung cho toàn bộ 70 hội thoại).
* **Per-Conversation Prefix:** **1.000 tokens** (Dữ liệu bối cảnh riêng cho Turn 1 của mỗi hội thoại).
* **New User Tokens Per Turn:** **150 tokens** (Đoạn câu hỏi mới của user mỗi turn).
* **Output Tokens Per Turn (Pinned):** **300 tokens** (Cố định độ dài sinh ra tại mỗi turn).
* **Quy luật arrival:** Phân phối Poisson (Random Seed = 42).

### 1.2 Biến Thiên Context Length Qua 6 Turn (Token Budget Breakdown)

| Turn | Prompt Input Tokens Breakdown | Tổng Prompt Tokens | Output Tokens | Tổng Context Sau Turn |
|---|---|---|---|---|
| **Turn 1** | Shared (1000) + Per-Conv (1000) + User (150) | **2.150 tokens** | 300 tokens | **2.450 tokens** |
| **Turn 2** | History (2.450) + User (150) | **2.600 tokens** | 300 tokens | **2.900 tokens** |
| **Turn 3** | History (2.900) + User (150) | **3.050 tokens** | 300 tokens | **3.350 tokens** |
| **Turn 4** | History (3.350) + User (150) | **3.500 tokens** | 300 tokens | **3.800 tokens** |
| **Turn 5** | History (3.800) + User (150) | **3.950 tokens** | 300 tokens | **4.250 tokens** |
| **Turn 6** | History (4.250) + User (150) | **4.400 tokens** | 300 tokens | **4.700 tokens** (Peak) |

> 📌 **Phát hiện quan trọng về Memory Management:**
> * Context dài nhất trong toàn bộ benchmark chỉ đạt **4.700 tokens** ở cuối Turn 6.
> * Việc đặt `--max-model-len=32768` (32K) làm vLLM cấp phát trước bảng quản lý block KV cache quá lớn so với mức đỉnh 4.700 tokens. Điều chỉnh `--max-model-len=8192` hoặc `16384` hoàn toàn đáp ứng đủ peak context 4.700 tokens mà tiết kiệm VRAM quản lý bảng block.

---

## 2. Công Thức Tính Điểm & Yêu Cầu Kỹ Thuật (Đã Chuẩn Hóa Displays)

### 2.1 Vòng Online (Leaderboard — Điểm ERS)

Điểm **Effective Request Score (ERS)** là trung bình cộng điểm số của $N = 420$ requests:

$$\text{ERS} = \frac{1}{N} \sum_{i=1}^{N} S_{\text{request}, i} \quad \in [0, 1]$$

Điểm của mỗi request thành công:

$$S_{\text{request}} = 0.5 \cdot s_{\text{ttft}} + 0.5 \cdot s_{\text{tpot}}$$

*(Nếu request bị lỗi, timeout hoặc trả về 0 token thì $S_{\text{request}} = 0$).*

Công thức tính thành phần độ trễ:

$$s_{\text{ttft}} = \left[ \text{clamp}\left( \frac{400 - \text{TTFT}}{400 - 10}, 0, 1 \right) \right]^2$$

$$s_{\text{tpot}} = \left[ \text{clamp}\left( \frac{10 - \text{TPOT}}{10 - 1}, 0, 1 \right) \right]^2$$

#### Bảng Tra Cứu Điểm $s_{\text{tpot}}$ Theo TBT (TPOT):

| TBT (TPOT) | Giá trị clamp | $s_{\text{tpot}} = (\text{clamp})^2$ | Điểm đóng góp vào ERS ($0.5 \cdot s_{\text{tpot}}$) |
|---|---|---|---|
| **1.0 ms** (Floor) | 1.000 | **1.0000** | +0.5000 |
| **2.0 ms** (Target Z1) | 0.888 | **0.7894** | +0.3947 |
| **3.0 ms** | 0.777 | **0.6049** | +0.3025 |
| **4.0 ms (Mốc E1+/E2/Z1)** | 0.666 | **0.4444** | **+0.2222** |
| **6.0 ms (Mốc P0/S1)** | 0.444 | **0.1975** | **+0.0988** |
| **10.0 ms** (Ceiling) | 0.000 | **0.0000** | +0.0000 |

> ⚡ **Nhận xét toán học:** TBT nén từ 6ms xuống 4ms làm $s_{\text{tpot}}$ tăng từ 0.1975 lên 0.4444 ($\Delta +0.2469$), đây chính là nguyên nhân giải thích vì sao ERS tăng vọt **+9.76 điểm** ở lần nộp FP8!

### 2.2 Vòng Hậu Kiểm (Post-Online — Accuracy Gate GPQA)

Sau vòng online, đội chọn tối đa **5 bài nộp**. BTC chạy GPQA full kiểm tra độ sụt giảm accuracy $\Delta$:

$$\Delta = \text{Accuracy}_{\text{baseline}} - \text{Accuracy}_{\text{submission}}$$

Hệ số phạt $f(\Delta)$ được tính theo điều kiện:

$$f(\Delta) = 1.0 \quad \text{nếu } \Delta \le 0.10$$

$$f(\Delta) = 1.0 - \frac{\Delta - 0.10}{0.06} \quad \text{nếu } 0.10 < \Delta < 0.16$$

$$f(\Delta) = 0.0 \quad \text{nếu } \Delta \ge 0.16$$

Điểm chung cuộc của bài nộp:

$$\text{Score}_{\text{final}} = 100 \times \text{ERS} \times f(\Delta)$$

---

## 3. Bảng Tổng Hợp Lịch Sử Nộp Bài Cá Nhân (Long Quân Tôn)

| ID Bài nộp | Version / Image | ERS | TBT (ms) | TTFT p50 / p95 | Fail / 420 | Trạng thái | Nguyên nhân kỹ thuật & Cấu hình chính |
|---|---|---|---|---|---|---|---|
| **P0 Baseline** | `lfm25:p0` (v0.23.0) | **49.81** | 6 ms | 48ms / 84ms | 7 | ✅ Keep (Anchor) | BF16 Base, Prefix Caching ON. Đạt chuẩn neo Accuracy Gate ($\Delta=0$). |
| **S1S2** | `lfm25:p0` (v0.23.0) | **48.45** | 6 ms | 55ms / 87ms | 7 | ❌ Reject | Maxlen 8192, Chunked 2048. Không hạ được TBT mà làm tăng TTFT nhẹ. |
| **X1 Decode** | `lfm25:p0` (v0.23.0) | **FAIL** | — | — | — | ❌ Reject | Speculative N-gram làm container crash khi BTC chạy probe long-context. |
| **E1+ FP8** | `lfm25:p0` (v0.23.0) | **59.57** | **4 ms** | 52ms / 73ms | 7 | ✅ Keep (ERS) | Ép FP8 Weight + FP8 KV Cache. Nén size weight từ 2.4GB $\rightarrow$ 0.6GB (**+9.76 điểm**). |
| **E2-Safe** | `lfm25:p1-v25` (v0.25.1) | **59.57** | **4 ms** | 53ms / 72ms | **5** | ✅ Keep (ERS) | Nâng vLLM v0.25.1 + Chunked Prefill 1024. Cứu 2 request lỗi (fail 7 $\rightarrow$ 5). |
| **Z1 FlashMamba**| `lfm25:p2-fi` (v0.25.1) | **61.18** | **4 ms** | **48ms / 68ms** | **4** | 🏆 **BEST** | Native FlashInfer SSM + Mamba State Cache Align + bt=512 (**+1.61 điểm**, Fail 5 $\rightarrow$ 4). |

---

## 4. Phân Tích Sâu Chi Tiết Từng Lần Nộp Bài & Nguyên Nhân Kết Quả

### 4.1 Lần 1 — P0 Baseline (Điểm: 49.81 | TBT: 6ms | Fail: 7/420)
* **Cấu hình:** `vLLM v0.23.0`, Weight BF16 gốc (~2.4GB), `--enable-prefix-caching`, `--max-model-len=32768`, `--gpu-memory-utilization=0.95`.
* **Phân tích:**
  * Prefix caching cache 1.000 tokens system prompt chung $\rightarrow$ TTFT p50 tốt (48ms).
  * Trọng số BF16 làm mỗi decode step đọc 2.4GB qua VRAM Bus. Chưa có chunked prefill nên request mới prefill làm gián đoạn decode loop của 70 conversations đồng thời $\rightarrow$ TBT bị đẩy lên 6ms ($s_{\text{tpot}} = 0.1975$).
  * 7 request bị fail do peak tải Poisson gây timeout.

### 4.2 Lần 2 — S1S2 One-shot (Điểm: 48.45 | TBT: 6ms | Fail: 7/420)
* **Cấu hình:** Hạ `--max-model-len=8192`, `--enable-chunked-prefill`, `--max-num-batched-tokens=2048`.
* **Phân tích:**
  * Lower `max-model-len` giải phóng VRAM nhưng bottleneck BF16 nằm ở Memory Read Speed chứ không phải VRAM capacity.
  * Chunked prefill 2048 quá lớn làm phân đoạn prefill thêm overhead mà không hạ được TBT decode $\rightarrow$ TTFT p50 tăng lên 55ms, ERS giảm 1.36 điểm.

### 4.3 Lần 3 — X1 Speculative Probe (Điểm: FAIL)
* **Cấu hình:** Speculative N-gram (5 tokens), `--optimization-level=3`, `--performance-mode=interactivity`.
* **Phân tích:** Speculative N-gram trên vLLM v0.23.0 chưa ổn định với kiến trúc Hybrid Mamba (`LFM2.5`), bị crash container khi BTC gửi probe test.

### 4.4 Lần 4 — E1+ FP8 Quantization (Điểm: 59.57 | TBT: 4ms | Fail: 7/420) — *Bước Ngoặt 1*
* **Cấu hình:** `--quantization=fp8` và `--kv-cache-dtype=fp8`.
* **Phân tích:**
  * FP8 nén size weights từ 2.4GB xuống **0.6GB** (giảm 4 lần khối lượng đọc VRAM Bus mỗi step).
  * TBT hạ từ 6ms $\rightarrow$ 4ms làm $s_{\text{tpot}}$ tăng từ 0.1975 lên **0.4444** $\rightarrow$ ERS nhảy vọt **+9.76 điểm**.

### 4.5 Lần 5 — E2-Safe (Điểm: 59.57 | TBT: 4ms | Fail: 5/420)
* **Cấu hình:** Image `v0.25.1` (`:p1-v25`), `--enable-chunked-prefill` với `--max-num-batched-tokens=1024`.
* **Phân tích:** Chunked prefill 1024 chia nhỏ prefill dài, cứu 2 request khỏi timeout (fail 7 $\rightarrow$ 5). TBT giữ 4ms do Mamba SSM state chưa được cache.

### 4.6 Lần 6 — Z1 FlashMamba (Điểm: 61.18 | TBT: 4ms | Fail: 4/420) — *Bước Ngoặt 2 (Kỷ Lục)*
* **Cấu hình:** Image `longquanton/develarper-lfm25:p2-fi` (vLLM v0.25.1 + `flashinfer-python>=0.6.4`), `--mamba-backend=flashinfer`, `--mamba-cache-mode=align`, `--max-num-batched-tokens=512`.
* **Phân tích:**
  * `--mamba-cache-mode=align` cache **toàn bộ Mamba SSM State** tại block boundaries song song với KV cache $\rightarrow$ Skip recompute SSM state cho 1.000 tokens prefix $\rightarrow$ TTFT p50 tối ưu về **48ms**.
  * Giảm `max-num-batched-tokens` xuống 512 giúp lồng ghép decode steps dày hơn, cứu thêm 1 request lỗi (fail 5 $\rightarrow$ 4).

---

## 5. Thắt Cổ Chai Cốt Lõi: CPU Scheduler Contention (3 vCPU Cores)

Mốc TBT = 4ms bị giữ nguyên qua 4 lượt nộp (E1+, E2, Z1). Phân tích hệ thống khẳng định:

* **GPU H200 (18GB) xử lý model FP8 (0.6GB) cực nhanh.** Bottleneck KHÔNG NẰM Ở GPU.
* **Bottleneck nằm ở 3 vCPU Cores:** BTC chỉ cấp 3 CPU cores. Trên 3 cores này, vLLM phải chạy đồng thời:
  1. HTTP API Server (FastAPI / Uvicorn).
  2. Async Continuous Batching Scheduler Loop cho 70 requests.
  3. Tokenize / Detokenize cho 126.000 output tokens.
  4. CUDA Kernel Launch overhead.

TBT 4ms chính là **CPU Scheduling Delay Threshold**.

---

## 6. Danh Sách 5 Bài Nộp Chốt Hậu Kiểm (Accuracy Gate Insurance)

1. **P0 Baseline (BF16, ERS 49.81):** Neo bảo hiểm accuracy ($\Delta = 0 \rightarrow f(\Delta) = 1.0$).
2. **E1+ (FP8 Weight + FP8 KV, ERS 59.57):** Dự phòng bản FP8 v0.23.0.
3. **E2-Safe (v0.25.1 + Chunked 1024, ERS 59.57):** Dự phòng bản FP8 v0.25.1.
4. **Z1 FlashMamba (`p2-fi`, ERS 61.18):** Bài ERS cao nhất hiện tại.
5. **Z2 CPU-Tuned (Target ERS >62):** Z1 bổ sung `--max-num-seqs=80` để giảm gánh nặng CPU scheduling.
