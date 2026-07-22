# PLAN.md — Push to 80 (evidence after E1+)
## Viettel AI Race 2026 · Challenge 3 · Team Develarper

> Cập nhật: **2026-07-22** sau E1+ Score **59.57**  
> Đề: [PROBLEM_VN.md](PROBLEM_VN.md) · [SUBMIT.md](SUBMIT.md)

---

## ★ Bảng điểm thật (Portal)

| ID | Score | tbt | ttft p50/p95 | fail | Bài học |
|---|---|---|---|---|---|
| **P0** BF16 | **49.81** | 6 ms | 48 / 84 | 7 | Sàn · **GPQA anchor** |
| S1S2 maxlen+bt | 48.45 | 6 ms | 55 / 87 | 7 | Scheduler **không** hạ TBT |
| X1 ngram | FAIL probe | — | — | — | **Cấm speculative** trên Portal |
| **E1+** FP8+KVFP8 | **59.57** | **4 ms** | 52 / 73 | 7 | **Quant đúng nút** · +9.76 · champion ERS online |

Online: `f_delta=1`, `accuracy_drop=0` (chưa GPQA). Δ thật chỉ sau khi chọn ≤5 bài.

---

## ★ Toán đường tới 80–90 (không ảo)

ERS ≈ `0.5·s_ttft + 0.5·s_tpot` (w=0.5, γ=2), fail≈7 chỉ “ăn” ~1 điểm nếu sửa hết → **không đủ** để lên 80.

Giả sử TTFT giữ ~50ms (`s_ttft≈0.80`):

| TBT median | `s_tpot` | ERS ước | Score ước |
|---|---|---|---|
| 6 ms (P0) | 0.20 | ~0.50 | ~50 |
| **4 ms (E1+)** | **0.44** | **~0.62** | **~60** (khớp 59.57) |
| 3 ms | 0.60 | ~0.70 | ~70 |
| 2 ms | 0.79 | ~0.80 | **~80** |
| ≤1 ms (floor) | 1.0 | ~0.90 | **~90** |

**Kết luận:** 80 cần TBT ~**2ms**; 90 cần sát **floor 1ms**. E1+ đã chứng minh hướng **compute quant**. Tiếp theo = chồng thêm kernel/runtime **Δ-safe**, không speculative (đã vỡ probe), không siết bt/maxlen (đã chứng minh vô ích cho TBT).

Mục tiêu thực tế theo evidence: **M1 ≥65 · M2 ≥70 · Stretch 80**. 90 = extreme.

---

## ★ Banlist / Allowlist (rút từ trải nghiệm)

| Cấm nộp lại | Cho phép & ưu tiên |
|---|---|
| Speculative / ngram (X1 probe) | Giữ E1+ quant (đã thắng) |
| `bt=2048` / cắt maxlen (S1S2) | `maxlen=32768` (probe) |
| Flag không có trên v0.23 | `--mamba-backend=flashinfer` (LFM2) |
| Overwrite tag `p0` | `--optimization-level=3` (Δ≈0) |
| Dual-path / mạng runtime | Image `p1` fastokens (TTFT/p95) |
| | Offline FP8 bake sau nếu online ổn |

---

## ★ Plan tối ưu “xịn” — tối đa 2 nộp tiếp

### Champion hiện tại
- **ERS online:** E1+ (59.57)  
- **Accuracy Gate:** luôn giữ **P0** trong ≤5 bài (BF16)

### Nộp #1 — **E2** (Δ-safe stack trên E1+)

Cùng image `p0`, **không** speculative:

```text
E1+  +  --optimization-level=3  +  --mamba-backend=flashinfer
```

| Vì sao | Evidence |
|---|---|
| `-O3` | Compile/CUDA graph aggress · Δ≈0 · chưa isolate |
| `flashinfer` | Đề cho phép FlashInfer · LFM2 hybrid SSU · chưa nộp |
| Giữ FP8+KVFP8 | Đã hạ TBT 6→4; không bỏ |

**Kỳ vọng:** TBT 4→~3ms → Score **~65–72**.  
**Risk:** boot fail FI → nộp lại chỉ E1+`+-O3` (bỏ mamba).  
**File:** [`docker-compose.yml`](docker-compose.yml) = [`submit/docker-compose.e2_fp8_o3_mamba.yml`](submit/docker-compose.e2_fp8_o3_mamba.yml)

### Nộp #2 — chỉ nếu E2 < 70 hoặc muốn đẩy 80

Chọn **một** (không gộp loạn):

| Option | Việc | Khi nào |
|---|---|---|
| **E2b** | Image **`p1`** (`Dockerfile.p1` fastokens) + flags E2 winner | E2 ổn, cần thêm TTFT/p95 + chút decode |
| **E3** | Offline FP8 bake `/model` + serve BF16-path flags tối thiểu | Online FP8 OK nhưng muốn ổn định hơn |
| **Stop** | Freeze E2/E1+ + P0 | Đã ≥70 hoặc hết ngày quý |

**Không** mở lại speculative trừ khi BTC đổi probe / có chứng minh pass long-context.

---

## ★ Shortlist GPQA (cuối phase)

1. **P0** — Δ an toàn (bắt buộc)  
2. **Best ERS** (E1+ hoặc E2…) — chỉ giữ nếu smoke GPQA Δ ≤0.10  
3–5. Dự phòng  

Online `f_delta=1` **không** nghĩa Δ thật =0 sau GPQA.

---

## ★ Việc bạn làm

1. Nộp **E2** (`docker-compose.yml` đã sẵn) khi còn quota / ≥600s.  
2. Gửi Score + tbt + fail + ttft.  
3. Quyết định E2b / dừng theo bảng trên — **không** ladder 5–10 shot.

---

## Probe / luật (nhắc)

Entrypoint BTC · `/model` · served-name · host/port · public image · offline · một behavior · maxlen 32768 · không cheat dual-path.
