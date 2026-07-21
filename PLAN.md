# PLAN.md — Optimization Playbook (Post-P0)
## Viettel AI Race 2026 · Challenge 3 · Team Develarper

> Chiến lược nền: [CONTEXT.md](CONTEXT.md) · Đề: [PROBLEM_VN.md](PROBLEM_VN.md) · Nộp: [SUBMIT.md](SUBMIT.md)  
> **Model:** `LiquidAI/LFM2.5-1.2B-Instruct` · **Slice:** MiG H200 **18GB** · **Engine:** **vLLM only**  
> **Image baseline:** `longquanton/develarper-lfm25:p0` (vLLM **0.23.0**, weights bake `/model`)  
> **Deadline Phase 1:** 30/07/2026 · Cập nhật: **2026-07-21 đêm** — gói **X1 Decode-Max** (thay B1 nhỏ lẻ)

---

## ★ X1 DECODE-MAX — lần nộp “lớn” tiếp theo (Δ-safe + ERS)

### Plan cũ (B1 chỉ +FP8) đổi thành gì?

| Trước (sau S1S2) | **Nay (X1)** |
|---|---|
| +1 flag `--quantization=fp8` | **Gói 3 đòn bẩy compute/latency**, **không** quant weights |
| Rủi ro Δ GPQA nếu FP8 | **Giữ BF16** → f(Δ) / Accuracy Gate thân thiện |
| Lặp kiểu “lí nhí” | Một submit **đổi lớp** so với P0 & S1S2 |

### Vì sao không lặp S1S2 / không FP8 ngay?

- S1S2 chứng minh: `maxlen` + `bt=2048` **không** hạ TBT (vẫn 6ms), còn làm Score↓.
- FP8 có thể ↑ERS nhưng **đốt Δ** — bạn muốn ERS cao **và** delta còn tốt cho hậu kiểm → để FP8 là **X2** chỉ khi X1 chưa đủ.

### Gói X1 (cùng image `p0`, không rebuild)

| Thành phần | Flag (đã có trên vLLM **0.23**) | Tác dụng | Δ / anti-cheat |
|---|---|---|---|
| Prefill reuse | `--enable-prefix-caching` (giữ P0) | Shared 1000-tok prefix | Trung thực |
| Compile aggress | `--optimization-level=3` | Recipe LFM2.5 ~+steady decode | Cùng weights → Δ≈0 |
| Latency mode | `--performance-mode=interactivity` | CUDA graph/kernel thiên **latency / ITL** (đánh TBT) | Δ≈0 |
| Spec decode | `--speculative-config` **ngram** (5 tok, lookup 4) | Nhiều token/verify bước; **target model verify** → phân phối giữ | Hợp lệ vLLM; không draft lạ, không đổi `/model` |
| **Cố ý bỏ** | maxlen↓, bt=2048, FP8, `--disable-log-requests` | Đã fail / sai tên / rủi ro Δ | — |

**File nộp:** root [`docker-compose.yml`](docker-compose.yml) = [`submit/docker-compose.x1_decode_max.yml`](submit/docker-compose.x1_decode_max.yml)

**Locked BTC (không đụng):** entrypoint `python3 -m vllm.entrypoints.openai.api_server` · `--model=/model` · served-name · host/port · image public `p0` · không mạng runtime · không dual-path.

### Kỳ vọng số (không cam kết)

| | P0 | X1 mục tiêu |
|---|---|---|
| tbt_median | 6 ms | **≤4 ms** (ngram + interactivity + O3) |
| Score | 49.81 | **≥55** nếu TBT về ~4; stretch **≥60** nếu ~3 |
| fail | 7 | ≤7 (ngram không kỳ vọng sửa fail systematic) |
| f_delta online | 1 | 1; GPQA sau vẫn ưu tiên BF16 path |

### Rủi ro X1 & xử lý

| Rủi ro | Mitigate |
|---|---|
| Startup lâu vì `-O3` → healthcheck fail | Nếu exit sớm: nộp lại bản **X1b** bỏ `-O3`, giữ ngram+interactivity |
| Ngram acceptance thấp dưới tải → Score≤P0 | Rollback P0; thử **X2** = P0+FP8 (đổi Δ) hoặc suffix speculative |
| JSON speculative parse lỗi Portal | Fallback `X1c`: `--spec-method=ngram` + `--spec-tokens=5` |

### Bạn làm gì để nộp X1

1. Còn quota ngày / ≥600s từ S1S2.  
2. Upload **root `docker-compose.yml`** (đã = X1). Không push image.  
3. Gửi Score, fail, ttft_p50/p95, tbt → cập nhật ablation.  
4. Shortlist cuối: luôn giữ **P0** (+ X1 nếu thắng và Δ ổn).

---

## ★ Ngân sách eval còn lại (ước lượng)

| Constraint BTC | Số |
|---|---|
| Max / ngày | **5** |
| Gap | **≥600s** |
| Deadline Phase 1 | **30/07/2026** (~9 ngày từ 21/07) |
| Trần lý thuyết | 9×5 ≈ **45** |
| Thực tế nên dùng | **~10–15** submit có ý nghĩa (mỗi lần 1 giả thuyết rõ) |
| Đã có điểm | P0 49.81 · S1S2 48.45 (**2**) |
| Shortlist GPQA | chọn **≤5** bài cuối (P0 bắt buộc trong đó) |

**Lịch đề xuất (không đốt 5/ngày vô ích):**

```text
Ngày A:  X1 Decode-Max          ← bạn sắp nộp
Ngày B:  nhánh theo X1 (1 shot)
         · thắng → X1-tune (ngram 3 hoặc 7) HOẶC giữ + thử X2 FP8 riêng
         · thua  → X1b (bỏ O3) hoặc X2 FP8
Ngày C–E: tối đa 1–2/ngày tinh chỉnh winner
Cuối kỳ: freeze ≤5 (P0 + best ERS Δ-safe + …)
```

→ Khoảng **6–10 eval nữa** là đủ bao phủ hướng lớn; không cần 45.

---

## ★ ARCHIVE nhanh

| ID | Score | Lesson |
|---|---|---|
| P0 | **49.81** | Champion BF16 · GPQA |
| S1S2 | 48.45 | maxlen+bt **không** hạ TBT — cấm lặp |
| B1 FP8 đơn | (hoãn) | Thành **X2** sau X1 nếu cần ERS thêm và chấp nhận rủi ro Δ |

---

## 0. Executive verdict

| | |
|---|---|
| Champion | **P0 = 49.81** |
| Next | **X1 Decode-Max** (O3 + interactivity + ngram, BF16) |
| Triết lý | Sáng tạo trong biên vLLM; **ERS↑ không đánh đổi Δ** ở shot này |
| Cấm | S1S2-style · disable-log-requests · engine khác · tráo image `p0` |

---

## 1. Khóa đề & biên hợp lệ (không đàm phán)

### 1.1 Công thức

```text
Score = 100 × ERS × f(Δ)
Online  = tối đa hóa ERS (+ reliability)
Post    = chọn ≤5 submission → GPQA; cần Δ an toàn (≤0.10 giữ hệ số tốt)
```

| Param | Value |
|---|---|
| F_ttft / C_ttft | 10 / 400 ms |
| F_tpot / C_tpot | **1 / 10 ms** |
| γ | 2 |
| w | 0.5 |
| Acc baseline (ref) | ~0.4 |

### 1.2 BTC cho phép (đòn bẩy “cao cấp” vẫn hợp lệ)

| Được | Không được |
|---|---|
| Custom public Docker image | TRT-LLM / SGLang / engine khác |
| Sửa **source vLLM** trong image (admin forum) | Đổi weights sang model khác / “lookalike” |
| Online FP8 / KV-FP8 / flags engine | Pull HF / mạng ngoài lúc container chạy |
| Prefix cache, chunked prefill, CUDA graphs, compile | Dual-path (bench vs GPQA khác nhau) |
| Speculative **trong vLLM** nếu vẫn serve đúng `/model` | Entrypoint lệch form mẫu BTC |
| Bake weights + (tuỳ chọn) torch.compile cache | Pre-bake câu trả lời / fake streaming |

**Entrypoint bắt buộc:** `python3 -m vllm.entrypoints.openai.api_server`  
**File nộp:** đúng tên `docker-compose.yml` · Image Hub **public** · ≥600s giữa 2 submit · ≤5 submit/ngày.

### 1.3 Workload thật (lab duy nhất đáng tin = Portal)

Meta BTC (`eval/traces/btc_workload_meta.json`):

| Field | Value |
|---|---|
| Conversations × turns | 70 × 6 = **420** requests |
| shared_system_prefix | 1000 tok |
| per_conversation_prefix | 1000 tok |
| new_user / turn | 150 |
| output pinned / turn | **300** |
| Arrival | Poisson seed 42 |

**Ước context peak / hội thoại (turn 6):**

```text
input ≈ 1000 + 1000 + 6×150 + 5×300 = 4400 tokens
(+ margin chat template / special tokens) → thiết kế max-model-len ≈ 5120–8192 là đủ
```

→ Giữ `32768` như P0 **lãng phí KV trên 18GB** → ít concurrency / dễ preemption → **TPOT xấu + fail**. Đây là insight #1 sau P0.

---

## 2. Chẩn đoán P0 bằng toán ERS (vì sao ~50 điểm)

Approximation với median quan sát (P0):

| Trục | Latency | `s = ((C−x)/(C−F))^γ` |
|---|---|---|
| TTFT | 48 ms | `((400−48)/(400−10))^2 ≈ **0.81**` |
| TPOT | 6 ms | `((10−6)/(10−1))^2 ≈ **0.20**` |
| ERS≈0.5·0.81 + 0.5·0.20 | | **≈ 0.50** (khớp Score 49.81) |

**Sensitivity (γ=2):**

| Nếu TPOT median | s_tpot | ERS ước (TTFT giữ 48ms) |
|---|---|---|
| 6.0 ms (P0) | 0.20 | ~0.50 |
| 4.0 ms | 0.44 | ~0.63 |
| 3.0 ms | 0.60 | ~0.71 |
| 2.0 ms | 0.79 | ~0.80 |
| 1.5 ms | 0.89 | ~0.85 |

→ **Mỗi 1ms TPOT gần floor = nhảy điểm lớn.** TTFT từ 48→20ms chỉ nâng s_ttft ~0.81→0.95 → ERS +~0.07 — kém hơn hạ TPOT 6→3ms.

**Fail penalty:** 7/420 ≈ 1.7% request S=0; ưu tiên về **0 fail** trước khi chase 0.01 ERS bằng quant rủi ro.

---

## 3. Bản đồ đòn bẩy A→Z (ranked)

Thứ tự = **Expected ΔERS × P(hợp lệ) / Risk(Δ,fail,OOM)**. Chỉ 1 biến chính / submit Portal.

### Tier S — làm sớm (compose-only, image `p0` giữ nguyên)

| ID | Kỹ thuật | Vì sao | Rủi ro | File / flag |
|---|---|---|---|---|
| **S1** | **Chunked + `max-num-batched-tokens`** | V1 ưu tiên decode; nhỏ hơn → ITL/TPOT tốt hơn dưới prefill chồng | TTFT có thể ↑ nhẹ (OK vì còn dư) | `submit/docker-compose.a2_chunked.yml` (4096) |
| **S2** | **Hạ `max-model-len` → 8192 rồi 6144/5120** | Workload peak ~4.4k; giải phóng KV → concurrency ↑, preemption ↓ | Quá thấp → reject/fail nếu BTC template phình | Compose mới `a3_maxlen*` |
| **S3** | Sweep batched-tokens **2048 / 4096 / 8192** | Docs vLLM: nhỏ = ITL tốt; lớn = TTFT/throughput | Quá nhỏ → TTFT xấu / under-util | Compose `a4_bt*` |
| **S4** | `gpu-memory-utilization` 0.90 chỉ khi OOM/fail | P0 đã 0.95; đừng hạ sớm | Mất KV | `a1_mem90` |

### Tier A — quant / dtype (vẫn vLLM, cùng `/model`)

| ID | Kỹ thuật | Vì sao trên H200 MiG | Rủi ro Accuracy Gate |
|---|---|---|---|
| **A1** | `--quantization fp8` (online) | Matmul nhanh hơn BF16 trên Hopper-class; model 1.2B nhẹ | Δ GPQA — **giữ P0 BF16** làm ứng viên f(Δ)=1 |
| **A2** | `--kv-cache-dtype fp8` | Gấp đôi “chỗ” KV → batch lớn hơn → TPOT | Đôi khi ITL xấu hơn (issue cộng đồng); A/B bắt buộc |
| **A3** | FP8 weights **+** KV-FP8 | Chỉ sau khi A1/A2 đơn lẻ thắng | Cộng dồn Δ |

### Tier B — compilation / runtime (image mới `p1`+)

| ID | Kỹ thuật | Nguồn | Ghi chú |
|---|---|---|---|
| **B1** | `--optimization-level 3` | Liquid LFM2.5 recipe: ~+2% steady-state | Startup lâu hơn — bake/warm OK trên BTC nếu health timeout đủ |
| **B2** | Bake **torch.compile cache** vào image + `VLLM_FORCE_AOT_LOAD=1` (nếu match GPU) | vLLM optimization docs | **Cẩn thận:** cache H200 ≠ máy build; ưu tiên compile **trên cùng arch** hoặc bỏ force |
| **B3** | Tắt log thừa (`--no-enable-log-requests` nếu flag tồn tại trên 0.23) | Giảm overhead CPU path | Đã học: `--disable-log-requests` **sai** trên 0.23 |
| **B4** | `VLLM_USE_FASTOKENS=1` (+ cài `fastokens`) | Recipe: hạ TTFT tokenization-bound | Ít đụng TPOT; optional |
| **B5** | Mamba/Lfm knobs: `--mamba-backend`, cache mode | Recipe “experiment if you like” | Ưu tiên thấp; 1 flag/lần |

### Tier C — kỹ thuật cao vẫn trong biên vLLM (sau khi S+A bão hòa)

| ID | Kỹ thuật | Hợp lệ? | Khi nào |
|---|---|---|---|
| **C1** | **N-gram / suffix speculative** (`--speculative-config`) | Có — không cần draft model ngoài, vẫn target `/model` | Khi TPOT còn >3ms sau S/A; đo acceptance + fail |
| **C2** | Draft-model speculative (LFM 350M/230M bake thêm) | **Xám** — vẫn verify target = Instruct 1.2B; tăng image + VRAM | Chỉ nếu C1 có tín hiệu và còn VRAM |
| **C3** | Patch source vLLM (scheduler bias decode, prefix block size, Lfm2 kernel path) | Admin cho phép sửa vLLM | Khi flag hết room; cần tag `p2-src` + changelog nội bộ |
| **C4** | PD-disagg / Mooncake / multi-GPU | **Không** — 1× MiG 18GB | Bỏ |

### Tier X — cố ý không làm

- Đổi sang TRT-LLM / SGLang / TensorRT
- Serve model khác hoặc quant “lạ” làm logits lệch nặng
- Network / HF token lúc runtime
- Dual behavior theo header / env “BENCH”
- Overwrite tag `p0` đã chấm (dùng `p1`, `p2`…)
- Speculative làm **đổi phân phối output** mà không smoke GPQA

---

## 4. Lịch ablation Portal (sau P0)

Constraint: **≤5 submit/ngày**, cách **≥600s**, 1 biến chính / lần, ghi [`eval/ablation_sheet.md`](eval/ablation_sheet.md).

```mermaid
flowchart TD
  P0[P0 Score 49.81 ERS~0.50 fail7] --> S1[S1 A2 chunked bt=4096]
  S1 --> D1{ERS↑ và fail≤7?}
  D1 -->|Yes| S2[S2 max-model-len 8192]
  D1 -->|No| S2b[Thử bt=2048 hoặc maxlen trước]
  S2 --> D2{ERS↑ fail↓?}
  D2 -->|Yes| S3[S3 bt sweep trên maxlen thắng]
  D2 -->|No| Rollback[Giữ best BF16]
  S3 --> A1[A1 FP8 online]
  A1 --> A2[A2 KV-FP8]
  A2 --> B1[B1 -O3 image p1]
  B1 --> C1[C1 ngram speculative nếu còn room]
  C1 --> Gate[Chọn ≤5: best ERS + ≥1 BF16 sạch]
```

### Hàng đợi nộp

| # | ID | Nội dung | Mục tiêu |
|---|---|---|---|
| ✓ | P0 | BF16 baseline | Champion **49.81** |
| ✓ | S1S2 | maxlen+bt | Fail — không lặp |
| **1** | **X1** | O3 + interactivity + ngram BF16 | **Nộp lớn tiếp theo** |
| 2 | X1b/X1c | fallback nếu boot/JSON | Theo PLAN ★ X1 |
| 3 | X2 | FP8 trên P0 | Chỉ nếu cần ERS thêm (rủi ro Δ) |

**Winner rule:** ERS↑ **và** failed_count không tăng → giữ; ngược lại rollback.  
**Accuracy:** luôn giữ **≥1** submission BF16 (P0 hoặc best BF16) trong shortlist GPQA.

---

## 5. Target số học (North Star)

| Mốc | TPOT median | Fail | ERS ước* | Score ước (f=1) |
|---|---|---|---|---|
| P0 (đã có) | 6 ms | 7 | ~0.50 | ~50 |
| M1 | ≤4 ms | ≤3 | ~0.62 | ~62 |
| M2 | ≤3 ms | 0 | ~0.70 | ~70 |
| Stretch | ≤2 ms | 0 | ~0.80 | ~80 |

\*Giả định TTFT p50 giữ ~40–60ms. Không phải cam kết — là **hướng tune**.

---

## 6. Image tags & DevOps hygiene

| Tag | Nội dung | Khi tạo |
|---|---|---|
| `p0` | BF16 bake, v0.23.0 — **đã chấm, không overwrite** | Done |
| `p1` | + `-O3` / fastokens / env compile (nếu cần) | Sau khi compose-only bão hòa |
| `p2` | + speculative config bake / patch vLLM nhỏ | Chỉ khi Tier C |
| `gpqa-safe` | Alias digest BF16 sạch | Trước chọn Accuracy Gate |

Checklist mỗi lần nộp:

- [ ] Filename đúng `docker-compose.yml`
- [ ] Image **public**, digest ghi ablation sheet
- [ ] Entrypoint form BTC
- [ ] `--model=/model`, `--served-model-name=LFM2.5-1.2B-Instruct`
- [ ] Không flag lạ đã fail CLI (vd. `--disable-log-requests`)
- [ ] Offline runtime (không HF)
- [ ] ≥600s từ submit trước

---

## 7. Roles (giữ)

| ID | Owns | Decide |
|---|---|---|
| **AI1** | Quant / Δ GPQA smoke | Có giữ FP8 vào shortlist không |
| **AI2** | Flag order, ers_sim, ablation sheet | Nộp ID nào tiếp |
| **DO** | Hub tag, compose Portal, anti-cheat | Freeze digest |

Daily 15’: Score mới / fail / còn mấy lượt / ID tiếp theo trong §4.

---

## 8. Definition of Done — Phase 1 (cập nhật)

- [x] Image Hub public + P0 ERS > 0
- [x] Prefix caching ON trên baseline
- [ ] ≥3 ablation có kiểm soát **sau P0** (S1, S2, …)
- [ ] failed_count → **0** trên best
- [ ] Best ERS ≥ **M1 (~0.62)** hoặc chứng minh ceiling phần cứng
- [ ] ≥1 BF16 digest trong ứng viên Accuracy Gate
- [ ] Anti-cheat checklist xanh trên winner

---

## 9. Next actions

1. Nộp **X1** — root `docker-compose.yml` (đã sẵn).  
2. Giữ P0 trong shortlist GPQA.  
3. Nhánh sau X1: xem § ★ X1 (X1b / X2 FP8).

Chi tiết: **[SUBMIT.md](SUBMIT.md)**.

---

## 10. Phụ lục — vì sao không ưu tiên một số “kỹ thuật hay”

| Ý tưởng | Lý do xếp thấp / loại |
|---|---|
| Tensor parallel | Model 1.2B fit 1 GPU; recipe Liquid: TP **không** giúp |
| `max-model-len=32768` mãi | Workload ~4.4k; phí KV trên 18GB |
| Chase TTFT <20ms | Marginal ERS vs effort; TPOT quan trọng hơn |
| Mooncake / disagg PD | Không đủ GPU / ngoài scope MiG đơn |
| Offline GPTQ nặng | Online FP8 đề khuyến khích; GPTQ thêm pipeline + rủi ro Δ |
| Sửa sampling server-side | BTC client kiểm soát generation; đừng lệch behavior |

---

*Plan này là nguồn sự thật cho tối ưu sau P0. Mọi thay đổi flag/image phải map về một ID trong §3–§4 và một dòng trong ablation sheet.*
