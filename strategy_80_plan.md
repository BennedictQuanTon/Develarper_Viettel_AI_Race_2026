# 🎯 Chiến Lược ERS → 80+ | Team Develarper

> **Phân tích từ kết quả thực tế E2-Safe (2026-07-23 00:02)**  
> ERS = 59.57 | TBT = 4ms | Fail = 5 | TTFT p50/p95 = 53/72ms

---

## 📊 Phân Tích Kết Quả E2-Safe

```
Điều tốt:    TTFT rất ổn (53/72ms << ceiling 400ms)
             Fail giảm 7→5 (chunked 1024 có tác dụng)
             f_delta = 1 (accuracy giữ nguyên ✅)
             
Vấn đề còn: TBT = 4ms (không đổi so với E1+)
             5 requests vẫn fail
             TPOT ceiling chỉ 10ms, γ=2 → cần ép 4→2ms
```

### Tại sao TBT không xuống dù đã dùng chunked + FP8?

**Root cause phân tích:**
1. **Memory bandwidth bottleneck**: MiG H200 18GB có memory bandwidth bị giới hạn theo slice. Model 1.2B FP8 ≈ 0.6GB weights. Mỗi decode step phải đọc toàn bộ weights qua memory bus.
2. **Batch contention**: 70 conversations đồng thời, decode phase xử lý nhiều sequences cùng lúc → contention trên memory bandwidth.
3. **Mamba SSM overhead**: LFM2.5 là hybrid (conv + GQA). SSM state update chưa được tối ưu nếu không dùng FlashInfer kernel cho Mamba.
4. **Chunked 1024 vẫn lớn**: Với decode 70 seqs × 300 tokens, mỗi step scheduler budget bị chiếm bởi chunk prefill 1024 tokens.

---

## 🚀 Các Đòn Bẩy Còn Lại (Xếp Theo Mức Ưu Tiên)

---

### 🔴 TIER 1 — Cao nhất (Xác suất thắng cao, ít rủi ro)

#### **[T1-A] `--mamba-backend=flashinfer` trên vLLM v0.25.1**

**Lý do khác với E2 cũ (đã bị loại):**
- E2 cũ dùng image `p0` (vLLM **v0.23.0**) + `--mamba-backend=flashinfer` → **BOOT FAIL** vì v0.23 FlashInfer cho LFM2 chưa ổn định
- Image hiện tại là `p1-v25` (vLLM **v0.25.1**) → FlashInfer cho Mamba **native + stable** trong V1 engine
- v0.25.1 tự động dispatch SSU operations sang FlashInfer kernel khi được bật → **TBT giảm 30-50%** nhờ optimized SSM kernels

**Config thêm vào E2-Safe hiện tại:**
```yaml
- --mamba-backend=flashinfer
```

**Rủi ro**: Thấp (v0.25.1 hỗ trợ tốt) · **Expected TBT**: 4→2-3ms · **Expected Score**: ~70-78

---

#### **[T1-B] `--mamba-cache-mode=align`**

**Lý do quan trọng:**
- LFM2.5 là Mamba hybrid. Hiện tại prefix caching chỉ cache KV của attention layers, KHÔNG cache Mamba SSM state
- `--mamba-cache-mode=align` → cache cả Mamba state tại block boundaries → giảm prefill computation mỗi turn
- Workload có 1000 token shared system prefix → **prefix cache hit rate tăng mạnh** → TTFT giảm → tăng time budget cho decode
- **Bắt buộc** kết hợp với `--enable-chunked-prefill` (đang có sẵn)

**Config:**
```yaml
- --mamba-cache-mode=align
```

**Rủi ro**: Rất thấp (documented feature cho LFM2) · **Expected TTFT**: 53→35ms · **Expected Score**: +3-5pts

---

#### **[T1-C] Giảm `--max-num-batched-tokens` xuống 512**

**Lý do:**
- Hiện tại bt=1024: Mỗi scheduler step có thể prefill 1024 tokens → block decode 70 seqs trong 1024/decode_speed giây
- Giảm xuống 512: Decode được xen kẽ nhiều hơn → TBT smooth hơn
- Workload: Turn 1 prefill tối đa ≈ 1000+1000+150 = 2150 tokens → cần 5 chunks thay vì 3 → decode không bị stall

**Config:**
```yaml
- --max-num-batched-tokens=512
```

**Trade-off**: TTFT có thể tăng nhẹ (thêm chunks) nhưng TTFT đang có rất nhiều headroom (53ms vs ceiling 400ms)

**Rủi ro**: Thấp · **Expected TBT**: 4→2.5-3ms · **Expected Score**: +5-8pts

---

### 🟡 TIER 2 — Trung bình (Cần test kỹ)

#### **[T2-A] `--max-num-seqs=64`**

**Lý do:**
- Default vLLM: 256 concurrent seqs
- Workload chỉ có 70 conversations → không cần admit nhiều hơn 70 seqs
- Nếu admission queue quá lớn → decoder batch quá to → contention → TBT tăng
- Giới hạn xuống 64-80 → batch decode chỉ 60-70 seqs → memory bandwidth dùng hiệu quả hơn

**Config:**
```yaml
- --max-num-seqs=80
```

**Rủi ro**: Trung bình (nếu quá thấp → queue → fail tăng) · **Expected TBT**: -10-20%

---

#### **[T2-B] `--performance-mode=interactivity`**

**Lý do khác với X1 (đã FAIL):**
- X1 FAIL vì kết hợp `--speculative-config ngram` + `--performance-mode=interactivity` → crash
- Speculative decoding là nguyên nhân crash, KHÔNG phải performance mode
- `--performance-mode=interactivity` **một mình** là an toàn → pre-tune scheduler ưu tiên TTFT/TPOT thay vì throughput

**Config (standalone, KHÔNG kết hợp speculative):**
```yaml
- --performance-mode=interactivity
```

**Rủi ro**: Trung bình · **Expected TTFT/TBT**: cải thiện ~10-15%

---

#### **[T2-C] Giữ chunked prefill nhưng giảm hơn nữa: bt=256**

**Lý do:**
- Workload output 300 tokens/request, 70 seqs = 21,000 decode tokens tổng
- Nếu bt=256: prefill chunk cực nhỏ → decode gần như uninterrupted
- Rủi ro: TTFT tăng (cần nhiều round trips hơn), nhưng TTFT có headroom lớn (353ms còn dư)

**Rủi ro**: Thấp-trung bình · **Expected TBT**: 4→2ms (aggressive)

---

### 🟢 TIER 3 — Bổ trợ (Kết hợp với TIER 1/2)

#### **[T3-A] `--gpu-memory-utilization=0.92`**

**Lý do:**
- Giảm nhẹ từ 0.95 → dành ~540MB buffer cho OS + CUDA context overhead
- Giảm OOM risk → giảm preemption → giảm fail count → cứu thêm 5 requests còn lại
- FP8 weights + FP8 KV đã rất nhỏ → 0.92 vẫn đủ KV cache cho 70 seqs × 6 turns

**Rủi ro**: Rất thấp

---

#### **[T3-B] `--max-model-len=8192` (thử lại kết hợp với FP8)**

**Lý do khác S1S2:**
- S1S2: BF16 + maxlen 8192 → TBT không đổi (vì bottleneck là compute, không phải KV)
- Nay: **FP8 + maxlen 8192** → KV đã nhỏ (FP8 KV) → giảm maxlen giải phóng thêm VRAM → tăng KV blocks → batch decode lớn hơn → throughput tăng
- Peak context turn 6 ≈ 4400 tokens → maxlen 8192 đủ thoải mái

**Rủi ro**: Thấp (đã dùng FP8 KV)

---

## 🏆 GOLDEN CONFIG ĐỀ XUẤT

### Config Z1 — "FlashMamba" (Ưu tiên #1)
```yaml
image: longquanton/develarper-lfm25:p1-v25   # vLLM v0.25.1
command:
  - --model=/model
  - --served-model-name=LFM2.5-1.2B-Instruct
  - --host=0.0.0.0
  - --port=8000
  - --max-model-len=32768
  - --gpu-memory-utilization=0.95
  - --tensor-parallel-size=1
  - --enable-prefix-caching
  - --quantization=fp8
  - --kv-cache-dtype=fp8
  - --enable-chunked-prefill
  - --max-num-batched-tokens=512      # Giảm từ 1024 → 512
  - --mamba-backend=flashinfer        # NEW: native SSM kernel
  - --mamba-cache-mode=align          # NEW: cache Mamba state
```

**Expected**: TBT 4→2ms, fail 5→0-2 → ERS ≈ **78-83**

---

### Config Z2 — "FlashMamba + Interactivity" (Ưu tiên #2)
```yaml
  # Tất cả như Z1 +
  - --performance-mode=interactivity  # NEW: ưu tiên latency
  - --max-num-seqs=80                 # NEW: giới hạn concurrent seqs
```

**Expected**: TBT 4→1.5-2ms, ERS ≈ **80-85**

---

### Config Z3 — "Conservative FlashMamba" (Dự phòng nếu Z1/Z2 boot fail)
```yaml
  # Z1 nhưng KHÔNG có flashinfer (fallback sang Triton)
  - --mamba-cache-mode=align
  - --max-num-batched-tokens=512
  - --gpu-memory-utilization=0.92
```

**Expected**: TBT 4→3ms, fail 5→1-2 → ERS ≈ **68-72**

---

## ⚠️ Những Gì KHÔNG Làm

| Đã thử/Cấm | Lý do |
|---|---|
| `--speculative-config ngram` | FAIL boot (X1) — cấm tuyệt đối |
| `--mamba-backend=flashinfer` trên **image p0** (v0.23) | Boot fail (E2 cũ) — chỉ an toàn trên v0.25.1 |
| Giảm maxlen xuống 8192 trên BF16 | S1S2: không hiệu quả (bottleneck là compute, không phải KV) |
| `--optimization-level=3` không kết hợp gì | X1b: chưa test, thứ tự ưu tiên thấp |
| Tăng bt lên 16384 | Y1: throughput mode, phản tác dụng với TPOT |

---

## 📐 Toán Học Kỳ Vọng

Với TTFT ≈ 50ms (s_ttft ≈ 0.816):

| TBT đạt | s_tpot | 5 fail fix → 0 | ERS | Score |
|---|---|---|---|---|
| 4ms (hiện tại) | 0.44 | +2.4pts | 0.595 | 59.57 |
| 3ms | 0.60 | +2.4pts | 0.713 | **71.3** |
| **2ms (Z1 target)** | **0.79** | **+2.4pts** | **0.804** | **~80** |
| 1.5ms | 0.87 | +2.4pts | 0.845 | ~84.5 |

> ✅ Z1 + cứu fail = **ước tính 78-83** nếu TBT đạt 2ms

---

## 🗺️ Thứ Tự Nộp Đề Xuất

```
Lần tiếp theo: Z1 (FlashMamba, bt=512, mamba-align)
Nếu Z1 > 70:   Z2 (+interactivity, max-num-seqs=80)
Nếu Z1 fail:   Z3 (conservative, KHÔNG flashinfer)
Giữ lại:       E1+ (59.57) + P0 (49.81) cho GPQA shortlist
```

---

## ✅ Phù Hợp Với Nội Quy

| Rule | Z1 compliance |
|---|---|
| Chỉ vLLM | ✅ vLLM v0.25.1 |
| Không network runtime | ✅ weights trong /model |
| Entrypoint đúng form | ✅ |
| Không speculative | ✅ (không có --speculative-config) |
| Không dual-path | ✅ |
| Accuracy safe | ✅ FP8 giữ Δ ≤ 0.05 |
