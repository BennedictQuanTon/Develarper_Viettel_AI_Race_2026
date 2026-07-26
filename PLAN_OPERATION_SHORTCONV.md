# PLAN OPERATION SHORTCONV — Mac-only p8 canary (neo p7)

| | |
|---|---|
| **Baseline vàng** | `p7-oneshot` · ERS **62.01** · TBT **4ms** · TTFT 48/68 · fail 5 |
| **Canary** | `longquanton/develarper-lfm25:p8-shortconv` |
| **Máy** | MacBook only · **không** thuê cluster |
| **Slot** | 2 — slot1 p8 · slot2 rollback p7 nếu hỏng/tụt |
| **Nguyên tắc** | CLI p7 nguyên văn · fail-open · một đòn ShortConv decode |

---

## 0. Verdict công tâm

1. **SHORTCONV đúng chỗ:** FlashInfer chỉ tăng Mamba SSU; LFM2 ShortConv decode vẫn `causal_conv1d_update` + 2 pointwise (`B*x`, `C*y`).
2. **Cửa 65+ = hạ TPOT/TBT**, không phải đánh TTFT p50/p95 (xem §2).
3. **Mac không compile CUDA SM90** — ship **Triton source**, BTC H200 JIT.
4. **Không hứa 65+.** Mục tiêu: canary sạch + tín hiệu TBT; sống sót Portal là ràng buộc cứng.

---

## 1. Bảng tử thần (cập nhật)

### Win

| Đòn | Δ | Ghi chú |
|---|---|---|
| Online FP8 | TBT 6→4 · **+9.76** | GPU Linear |
| FlashInfer + align | TTFT/fail · **+1.61** | Prefill/cache |
| p7 v0.26 | p95 70→68 · **+0.35** | TBT vẫn 4 |

### CẤM vĩnh viễn

| Đòn | Kết quả |
|---|---|
| Speculative n-gram | Crash / probe |
| `mamba-backend=CUDA` / `=true` | Exit 1/2 |
| `bt=256` + `seqs=80` | TTFT p95 137 · 56.44 |
| OMP / TOKENIZERS env | 58.34 |
| Offline static FP8 | TBT 4→6 · 50.46 |

---

## 2. So sánh ERS: TBT/TPOT vs TTFT p50/p95

**Bạn đúng phần lõi:** muốn 65+ phải kéo **TPOT ~4 → ≤3.5–3.7ms**.

**Sửa hiểu nhầm:**
- ERS **không** chấm median/p50/p95 — trung bình 420 request, mỗi request `0.5·s_ttft + 0.5·s_tpot`, γ=2.
- TTFT p50 đã ~48ms từ P0→p7 (gần bão hòa band 10–400ms).
- Biên tế gần p7: **~+8 pts / 1ms TPOT** vs **~+0.23 pts / 1ms TTFT**.

| Thay đổi | Δ ERS ước lượng |
|---|---|
| TPOT 4→3.7 | ~+2.3 (cửa ~64–65) |
| TPOT 4→3.5 | ~+3.9 (cửa ~66–67) |
| TPOT 4→3.0 | ~+8 (cửa ~70) |
| Chỉ phẳng TTFT p95 68→48 | ~+0.2 |
| Fail 5→0 | ~+0.7 |

→ Canary p8 **nhắm TBT**; TTFT chỉ bảo vệ không regress (Z2).

---

## 3. Kiến trúc p8

```text
Image :p8-shortconv
  = vLLM 0.26.0 + flashinfer + /model (BF16)
  + develarper_opt p08_shortconv_fuse
  + sitecustomize → patch ShortConv.forward_cuda trước api_server

Decode (width=3, state_len=2):
  fused Triton: y = C * conv_update(B * x)   # 1 launch / layer
Prefill: stock causal_conv1d_fn
Fail-open: mọi lỗi patch/fuse → stock p7 path (server vẫn lên)
```

**Compose Portal:** entrypoint khóa BTC giữ nguyên; **chỉ đổi `image:`**.

File:
- [`Dockerfile.p8_shortconv`](Dockerfile.p8_shortconv)
- [`develarper_opt/`](develarper_opt/)
- [`scripts/build_push_p8_shortconv.sh`](scripts/build_push_p8_shortconv.sh)
- Root [`docker-compose.yml`](docker-compose.yml) → p8
- Rollback [`submit/docker-compose.p7_oneshot.yml`](submit/docker-compose.p7_oneshot.yml)

---

## 4. Ngân sách dung lượng (Mac: pull → build → push)

**p8 build = `FROM :p7-oneshot` + layer opt nhỏ** (không kéo lại v0.26 / không COPY lại 2.2GB weights).

| Hạng mục | Typical (máy đã có p7) | Worst case |
|---|---|---|
| Download base p7 | **~0** | ~10.8 GB nếu mất local |
| Build context | chỉ `develarper_opt` (dockerignore) | — |
| Push `:p8-shortconv` | **~0.01–0.2 GB** (Hub đã có layer p7) | ~11 GB nếu không reuse |
| **Tổng mạng** | **≪ 1 GB** (buffer **~2 GB** đủ) | ~12–20 GB |

Disk trống: vài GB thêm cho layer mới là đủ (không cần 30 GB như bản build-from-scratch cũ).

---

## 5. Lệnh bạn tự chạy (khi sẵn sàng)

```bash
cd /Users/davark/Downloads/Everything/Github/Develarper_Viettel_AI_Race_2026

# Docker Desktop ON + đã docker login
bash scripts/build_push_p8_shortconv.sh
```

Sau push OK → upload **root `docker-compose.yml`** lên Portal.

**Slot 2 rollback:** copy `submit/docker-compose.p7_oneshot.yml` → `docker-compose.yml` rồi nộp lại.

---

## 6. Gate đọc kết quả Portal

| Kết quả | Hành động |
|---|---|
| Boot FAIL | Slot2 → p7 ngay |
| TBT vẫn 4 · ERS ~61–62 | Giữ p7; fuse không ăn / fallback |
| TBT ≤ 3.7 · ERS ↑ | Freeze p8 |
| Điểm &lt; ~58 / TBT xấu | Rollback p7; cấm lặp fuse này |

---

## 7. Thẻ bỏ túi

```text
VÀNG:  p7 62.01 — CLI không đụng
P8:    image only + Triton ShortConv decode fuse (fail-open)
CẤM:   OMP | bt=256 | speculative | static FP8 | mamba=CUDA
TOÁN:  TBT 3.5 → ~65+ ; TTFT p50/p95 không đủ một mình
MÁY:   Mac buildx amd64 ; JIT thật trên H200 BTC
```
