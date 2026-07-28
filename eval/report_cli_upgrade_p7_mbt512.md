# Báo cáo CLI Upgrade — p7-mbt512 vs baseline p7-oneshot

**Portal:** 2026-07-27 ~08:35 · Team Develarper · Challenge 3  
**Canary ID:** `p7-mbt512`  
**Baseline vàng:** `p7-oneshot` (ERS **62.01**)

---

## 1. Mục tiêu canary

Sau khi kernel (p8/p9) và nhiều knob CLI đã eval âm hoặc bão hòa **TBT = 4ms**, residual duy nhất còn lý do học trên đúng image p7:

> Chỉ đổi `--max-num-batched-tokens`: **768 → 512**, giữ nguyên mọi thứ khác (image `:p7-oneshot`, seqs=128, mem=0.96, FP8, FlashInfer align, block=32, maxlen=8192).

**Giả thuyết:** Z1 từng đạt **fail = 4** với `bt=512` → trên p7 v0.26 có thể fail↓ nhẹ mà không phá TTFT → ERS nhích ~62.2–62.5.

**Không kỳ vọng:** TBT &lt; 4 hay ERS 65+.

---

## 2. Cấu hình đối chiếu

| Knob | p7-oneshot (baseline) | p7-mbt512 (canary) |
|---|---|---|
| Image | `longquanton/develarper-lfm25:p7-oneshot` | **giống** |
| vLLM | 0.26.0 | giống |
| `--max-num-batched-tokens` | **768** | **512** |
| `--max-num-seqs` | 128 | 128 |
| `--gpu-memory-utilization` | 0.96 | 0.96 |
| `--block-size` | 32 | 32 |
| `--max-model-len` | 8192 | 8192 |
| FP8 W+KV / prefix / chunked | ON | ON |
| `--mamba-backend` / cache-mode | flashinfer / align | giống |
| Kernel / sitecustomize patch | Không | Không |

Compose nộp: root `docker-compose.yml` (archive: `submit/docker-compose.p7_mbt512.yml`).  
Rollback vàng: `submit/docker-compose.p7_oneshot.yml`.

---

## 3. Kết quả Portal

| Metric | p7-oneshot | **p7-mbt512** | Delta |
|---|---|---|---|
| **ERS / Score** | **62.01** | **61.05** | **−0.96** |
| TBT median (ms) | 4 | **4** | 0 |
| TTFT p50 (ms) | 48 | **49** | +1 |
| TTFT p95 (ms) | 68 | **77** | **+9** |
| fail / 420 | 5 | **5** | 0 |
| tokens_per_sec (Portal) | — | 0.0556 | — |
| f_delta / accuracy_drop | placeholder | 1 / 0 | online only |

---

## 4. Phân tích kỹ thuật

### 4.1 Giả thuyết fail↓ — **không đúng trên p7**

Fail giữ **5/420**. Lợi thế `bt=512` của Z1 (fail=4) gắn với **v0.25.1 + stack/bt khác**, không replicate khi chỉ hạ mbt trên p7 v0.26 + `seqs=128` + `mbt` từng là 768.

### 4.2 TTFT bị hại — đúng với cơ chế chunked prefill

Turn-1 prompt ~**2150** tokens:

- `mbt=768` → ~3 chunks prefill  
- `mbt=512` → ~5 chunks prefill  

Thêm bước scheduler/GPU round → **TTFT p95 68→77 ms (+13%)**, p50 cũng hơi xấu (48→49). Với γ=2 và w=0.5, đuôi TTFT kéo điểm request xuống đủ để giải thích **~−1 ERS** khi TBT và fail không đổi.

### 4.3 TBT không đổi

Khẳng định lại trần CLI: đổi mbt **không** đụng decode TPOT median — vẫn **4 ms**.

### 4.4 Vì sao “upgrade” thành hạ điểm

Canary tối ưu **sai hướng** trên baseline đã sweet-spot: p7 chọn `768` cân bằng prefill chunk vs ổn định. Hạ xuống `512` chỉ làm prefill mảnh hơn → TTFT xấu, **không** đổi fail/TBT → net **−0.96**.

---

## 5. Kết luận & khuyến nghị

1. **Baseline tối ưu CLI vẫn là p7-oneshot (`mbt=768`) · ERS 62.01** — không thay bằng mbt=512.
2. Residual CLI “học Z1 fail” trên p7: **eval âm** → **đóng chương vặn CLI serving**.
3. BTC giữ best → điểm đội online vẫn neo **62.01** (p7); bài 61.05 chỉ ablation.
4. Shortlist GPQA: ưu tiên **P0** (accuracy anchor) + **p7** (best ERS) + các bài FP8/FlashInfer ổn (E1+/E2/Z1/#10); **không** chọn p7-mbt512 / p8 / p9 làm đại diện tốc độ.

---

## 6. Bảng đóng trần CLI (tóm tắt)

| Hướng CLI còn lại trước canary | Sau p7-mbt512 |
|---|---|
| mbt=512 one-knob trên p7 | **Đã thử — thua** |
| mem / compilation / seqs thấp / multistep / OMP / static FP8 | Đã cấm hoặc đã âm trước đó |
| Kỳ vọng TBT&lt;4 bằng CLI | **Không khả thi** trên evidence đầy đủ |

**File liên quan:** `eval/ablation_sheet.md` · `submit/docker-compose.p7_oneshot.yml` · `submit/docker-compose.p7_mbt512.yml`
