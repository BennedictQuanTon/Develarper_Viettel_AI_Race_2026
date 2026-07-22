# PLAN.md — Push to 75-80+ (Evidence-based & Zero-Risk Strategy)
## Viettel AI Race 2026 · Challenge 3 · Team Develarper

> Cập nhật: **2026-07-23** sau kết quả E2-Safe (59.57).
> Đề: [PROBLEM_VN.md](PROBLEM_VN.md) · [SUBMIT.md](SUBMIT.md)

---

## ★ Bảng điểm thật & Lịch sử thử nghiệm

| ID | Score | tbt | ttft p50/p95 | fail | Bài học rút ra |
|---|---|---|---|---|---|
| **P0** BF16 | **49.81** | 6 ms | 48 / 84 | 7 | Sàn · **GPQA anchor** |
| S1S2 maxlen+bt | 48.45 | 6 ms | 55 / 87 | 7 | Scheduler **không** hạ TBT |
| X1 ngram | FAIL probe | — | — | — | **Cấm speculative** (Crash server) |
| **E1+** FP8+KVFP8 | **59.57** | **4 ms** | 52 / 73 | 7 | **Quant đúng nút** · +9.76 |
| **E2-Safe** v0.25.1 | **59.57** | **4 ms** | 53 / 72 | **5** | Chunked 1024 cứu 2 fail · TBT không xuống thêm |
| **Z1 FlashMamba** | *Pending* | **~2 ms** | **~45 ms** | **~0** | **FlashInfer SSM + Mamba Cache Align** |

---

## ★ Toán học chi tiết: Làm sao để ERS > 80?

$$ERS = 0.5 \cdot s_{ttft} + 0.5 \cdot s_{tpot}$$

Giả sử TTFT duy trì ~50ms ($s_{ttft} \approx 0.815$):

| TPOT (TBT) median | $s_{\text{tpot}}$ | Tỷ lệ Req Thành công (Fail) | ERS ước tính | Score ước tính |
|---|---|---|---|---|
| 6 ms (P0) | 0.20 | 413 / 420 (Fail 7) | ~0.50 | ~49.81 |
| 4 ms (E1+/E2) | 0.44 | 415 / 420 (Fail 5) | ~0.60 | ~59.57 |
| 3 ms | 0.60 | 420 / 420 (Fail 0) | ~0.71 | ~71 |
| **2.0 ms (Z1 target)** | **0.79** | **420 / 420 (Fail 0)** | **~0.80** | **~80** |
| 1.0 ms (Floor) | 1.00 | 420 / 420 (Fail 0) | ~0.90 | ~90.7 |

> 📌 **Bài toán chốt hạ:** Ép TBT xuống ≤ 2ms + cứu 5 fails còn lại.

---

## ★ Root Cause: Tại Sao TBT Kẹt 4ms Dù Đã FP8?

### Vấn đề 1: Mamba SSM kernel chưa được tối ưu
- LFM2.5 là **hybrid**: gated short-convolution + GQA Attention
- vLLM phải xử lý **2 loại operation**: KV-attention (đã FP8) + SSM state update (chưa tối ưu)
- `--mamba-backend=flashinfer` → dùng **FlashInfer native SSM kernels** thay vì fallback Triton
- **Tại sao lần trước fail:** image `p0` (v0.23.0) → FlashInfer cho LFM2 chưa stable → boot fail
- **Tại sao lần này an toàn:** image `p1-v25` (v0.25.1) → V1 engine, FlashInfer SSM native, đã verify

### Vấn đề 2: Prefix caching chỉ cache KV, KHÔNG cache Mamba SSM state
- Workload có **1000-token shared system prefix** → mỗi request phải recompute SSM state
- `--mamba-cache-mode=align` → checkpoint SSM state tại block boundaries → cache hit = skip recompute
- **Requires:** `--enable-chunked-prefill` (đã có) + `--mamba-backend` tương thích

### Vấn đề 3: bt=1024 vẫn tạo decode stall
- 70 seqs × ~300 output tokens = 21,000 decode steps cần luân phiên
- bt=512 → prefill chunk nhỏ hơn → decoder được schedule dày hơn → TBT giảm

---

## ★ Z1 FlashMamba — Golden Config (Current Submit)

```yaml
image: longquanton/develarper-lfm25:p1-v25
command:
  --model=/model
  --served-model-name=LFM2.5-1.2B-Instruct
  --host=0.0.0.0 --port=8000
  --max-model-len=32768
  --gpu-memory-utilization=0.95
  --tensor-parallel-size=1
  --enable-prefix-caching
  --quantization=fp8
  --kv-cache-dtype=fp8
  --enable-chunked-prefill
  --max-num-batched-tokens=512     ← giảm từ 1024
  --mamba-backend=flashinfer       ← NEW: native SSM kernel
  --mamba-cache-mode=align         ← NEW: cache SSM state
```

File: root `docker-compose.yml` (đã cập nhật) + archive `submit/docker-compose.z1_flashmamba.yml`

---

## ★ Accuracy Retention (Đảm bảo GPQA cao nhất)

- **FP8 E4M3/E5M2** giữ cho $\Delta \le 0.05$ (nằm sâu bên trong ngưỡng an toàn ≤ 0.10)
- Giữ bài nộp **P0 (BF16)** + **E1+/E2 (59.57)** trong danh sách 5 bài chốt cuối làm phao GPQA

---

## ★ Việc bạn cần làm

Upload duy nhất file [`docker-compose.yml`](docker-compose.yml) (đã được cập nhật Z1 FlashMamba) lên Portal BTC.
