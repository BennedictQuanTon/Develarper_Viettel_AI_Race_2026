# PLAN.md — Evidence-first (không đoán mò scheduler nữa)
## Viettel AI Race 2026 · Challenge 3 · Team Develarper

> Cập nhật: **2026-07-21 23:55** — tách **đã chứng minh** vs **giả thuyết**; chốt nộp **E1 = P0+FP8**  
> Đề: [PROBLEM_VN.md](PROBLEM_VN.md) · [SUBMIT.md](SUBMIT.md) · [CONTEXT.md](CONTEXT.md)

---

## ★ Thẳng thắn: mình biết gì / chưa biết gì

### Đã chứng minh bằng Portal (không phải đoán)

| Evidence | Kết luận cứng |
|---|---|
| P0: Score **49.81**, **tbt=6ms**, fail=7, TTFT dư | Nút ERS = **TPOT/TBT**, không phải TTFT |
| S1S2: đổi maxlen+bt (scheduler/KV) → Score↓, **tbt vẫn đúng 6ms**, fail vẫn 7 | Lớp **batch/maxlen không đụng được TBT** trên MiG này |
| X1: +ngram speculative → **long-context probe 0% / dual-path** | **Cấm speculative** trên Portal |
| Công thức γ=2, TPOT ceiling 10ms | Mỗi ms TBT gần floor = nhảy điểm lớn |

### Chưa đo trên Portal (chỉ là giả thuyết)

| Ý | Status |
|---|---|
| `throughput` + `bt=16384` (Y1) | **Đoán ngược** S1S2 — chỉ chứng minh bt *nhỏ* không giúp, **không** chứng minh bt *lớn* giúp TBT. **Không ưu tiên nộp.** |
| `-O3` alone | Chưa isolate; docs còn nói O3≈O2 → có thể =0 |
| mamba flashinfer / fastokens image | Chưa đo |
| **FP8 weights** | **Chưa nộp lần nào** — đúng lớp *compute* mà evidence bảo cần thử |

### Suy luận hợp lệ cho lần nộp tiếp

```text
TBT không đổi khi đụng scheduler
  → nút thắt ≈ chi phí decode / matmul BF16
  → đòn bẩy cùng lớp chưa thử + đề cho phép online quant
  → --quantization=fp8 trên nền P0 (đúng 1 biến mới)
```

Giữ **P0 BF16** trong shortlist GPQA (FP8 có thể đụng Δ).

---

## ★ Nộp tiếp: E1+ (quant family — không chỉ weight 8-bit)

Không phải “chỉ quant rồi thôi” theo nghĩa bỏ hết tối ưu: **giữ full P0 an toàn probe**, và chồng **hai đòn quant đề cho phép**, cùng lớp với nút TBT.

| Thành phần | Flag | Việc nó làm | Ảnh hưởng |
|---|---|---|---|
| (giữ) Prefix | `--enable-prefix-caching` | Reuse system prefix 1000 tok | TTFT/prefill — đã đúng workload |
| (giữ) maxlen | `32768` | Qua long-context probe | Bắt buộc sau X1 |
| **Mới** Weight FP8 | `--quantization=fp8` | Matmul nhẹ hơn trên Hopper | **TBT/ERS↑** kỳ vọng; **Δ GPQA có thể↓** → giữ P0 shortlist |
| **Mới** KV FP8 | `--kv-cache-dtype=fp8` | KV nhỏ hơn → nhiều concurrent hơn trên 18GB | Hỗ trợ decode dưới tải 70 conv; Δ thường nhỏ hơn weight FP8 |
| Không thêm | speculative, bt nhỏ, maxlen↓, interactivity | Đã fail / không đụng TBT | — |

**Vì sao gộp weight+KV FP8 mà không gộp throughput/bt16384?**  
Cùng họ **quant/compute-memory** khớp evidence “TBT không đổi khi đụng scheduler”. Throughput/bt vẫn thuộc họ scheduler — chưa có bằng chứng TBT.

File: root [`docker-compose.yml`](docker-compose.yml).

### Sau E1+ — tối đa 1 shot

| Kết quả | Làm |
|---|---|
| Score↑ | Dừng · shortlist E1+ + **P0** |
| Fail probe/boot | Bỏ `kv-cache-dtype` còn mỗi weight FP8, hoặc về P0 |
| Score≤P0 | Coi P0 gần trần; không quay speculative/bt |

---

## Banlist (từ evidence)

- Speculative / ngram (X1)
- Siết `bt`/`maxlen` kiểu S1S2 làm “tối ưu TBT”
- Flag không có trên v0.23 · overwrite `p0`

---

## Archive điểm

| ID | Score | Role |
|---|---|---|
| P0 | **49.81** | Champion BF16 · GPQA anchor |
| S1S2 | 48.45 | Chứng minh scheduler ≠ TBT |
| X1 | FAIL probe | Cấm speculative |
| **E1+** | *next* | P0 + weight FP8 + KV FP8 |

---

## Probe / đề (nhắc)

Entrypoint BTC · `/model` · served-name · host/port · maxlen 32768 · image public · offline · không dual-path.
