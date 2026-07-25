# PLAN CLASSIC — Operation SHORTCONV
## Viettel AI Race 2026 · Challenge 3 · Team Develarper  
### Bản kế hoạch kinh điển sau toàn bộ lịch sử Portal (P0 → p5)

| | |
|---|---|
| **Điểm vàng bất biến** | Yoshio **#10 · ERS 61.66 · TBT 4ms · TTFT 48/70 · fail 5** |
| **Mục tiêu kế tiếp** | Phá sàn TBT **4 → ≤ 3.5ms** (cửa ERS **~65+**); stretch TBT **≤ 3.0** (cửa **~70**) |
| **Máy hiện có** | Mac only (Docker OK, **không NVIDIA**) |
| **Nguyên tắc** | Đúng luật · một đòn / một tag · đo trước nộp · không đốt Portal |

---

## 0. Tuyên ngôn (sau quá nhiều lần nộp)

Chúng ta **không** còn ở pha “vặn compose may rủi”.

Portal đã dạy một định lý cứng:

> **Mọi đòn không đụng đường decode GPU của LFM2 thì không hạ được TBT dưới 4ms.**  
> Mọi đòn đụng nhầm đường decode (static FP8) có thể **đẩy TBT về 6ms**.

Vàng #10 là **trần serving chuẩn** (online FP8 + FlashInfer + align + bt=512).  
Còn lại đúng một mặt trận còn mở: **short-convolution layers** — phần hybrid chưa được “E1-hóa”.

Đây là plan **duy nhất còn EV** để tăng điểm thật. Khó. Làm được trên Mac phần lớn; **chỉ thuê GPU Hopper vài giờ** để compile + benchmark + bake image.

---

## 1. Bảng tử thần — đúc từ Portal (không bàn lại)

### 1.1 Win đã chứng minh

| Đòn | Δ metric | Δ ERS | Họ |
|---|---|---|---|
| Online FP8 W+KV | TBT 6→**4** | **+9.76** | **GPU decode Linear** |
| FlashInfer + `mamba-cache-mode=align` + bt=512 | TTFT đuôi + fail | **+1.61** | **Prefill/cache hybrid** |
| #10: block32 + maxlen8192 + mem0.96 | tinh chỉnh | **+0.48** | **Resource** |

### 1.2 Fail đã chứng minh — CẤM VĨNH VIỄN trên Portal

| Đòn | Kết quả | Cấm |
|---|---|---|
| Speculative n-gram | Crash / probe | ✅ |
| `mamba-backend=CUDA` / cờ sai / `=true` | Exit 1/2 | ✅ |
| `bt=256` + `max-num-seqs=80` | TTFT p95 137 · 56.44 | ✅ |
| OMP / TOKENIZERS env pin | TTFT xấu · 58.34 | ✅ |
| Offline static FP8 (compressed-tensors) | TBT **4→6** · **50.46** | ✅ |
| Vặn CLI thêm trên `:p2-fi` kỳ vọng lớn | Trần #10 | ✅ |

### 1.3 Hai sàn cứng

```text
TBT p50/median ≈ 4 ms     ← chỉ E1 từng phá (và p5 phá ngược)
TTFT p50        ≈ 48 ms   ← đã có từ P0 (prefix); gần như sàn
```

Toán ERS (γ=2, w=0.5), TTFT giữ 48:

| TBT | ERS lý thuyết ≈ |
|---|---|
| 4.0 | ~62–63 (khớp #10) |
| **3.5** | **~66–67** ← mục tiêu chính Operation SHORTCONV |
| **3.0** | **~70–71** ← stretch |

Fail 5→0 chỉ ~**+0.7** điểm — không đủ một mình.

---

## 2. Chẩn đoán kiến trúc — vì sao còn room

Từ `config.json` LFM2.5-1.2B:

```text
16 layers:
  conv            × 10  (62.5%)   ← short-conv, conv_L_cache = 3
  full_attention  ×  6  (37.5%)   ← GQA 32/8, đã hưởng FP8 + prefix/KV-FP8
```

**Insight kinh điển:**

1. E1 FP8 tối **Linear / attention matmul** → TBT 6→4.  
2. 10/16 layer là **conv cửa sổ 3** — mỗi token decode vẫn launch kernel(s) riêng (conv → gate/SiLU → …).  
3. Z3 chứng minh: **không phải** “chỉ thiếu OMP=1”.  
4. p5 chứng minh: **đổi format weight Linear** thêm lần nữa **không** giúp — còn phá path đang thắng.  
5. Z1 chứng minh: hybrid **prefill** còn ăn nhờ **align SSM/state**; decode token-by-token vẫn dính **short-conv**.

→ Operation SHORTCONV = giảm **chi phí / số launch** của 10 layer conv mỗi decode step, **không** đụng CLI vàng #10.

---

## 3. Chiến lược Operation SHORTCONV (1 mục tiêu · 3 tầng)

### Tầng S0 — Bảo hiểm (không đàm phán)

- Compose Portal mặc định = **pure Yoshio #10** (`:p2-fi` + online FP8 + flashinfer + align + bt=512…).  
- Shortlist GPQA: **P0 + #10 + Z1…** — **không** p5/Z3.  
- Tag mới chỉ `:p7-shortconv` (hoặc tương đương). **Không đè** `:p0`/`:p2-fi`.

### Tầng S1 — Kernel path (đòn chính · phá TBT)

**Mục tiêu kỹ thuật:** fused (hoặc SM90-tuned) **causal short-conv L=3** cho LFM2:

1. Inventory: trong vLLM 0.25.1, LFM2 conv đang gọi kernel nào (`causal_conv1d`, Triton, eager)?  
2. Đóng image với **`causal-conv1d` / tương đương build `TORCH_CUDA_ARCH_LIST=9.0`** (H200 = Hopper SM90) — tránh PTX JIT lạnh / kernel generic.  
3. (Nâng cao) Triton **fuse** `causal_conv1d + activation/gate` thành 1 launch / layer.  
4. Giữ numerics: so khớp logits vs #10 trên vài prompt (Δ nhỏ) — bảo vệ Accuracy Gate.

**Không** làm trong S1: đổi quant scheme, speculative, bt, OMP.

### Tầng S2 — Bake phục vụ (đòn phụ · bảo vệ TTFT)

Chỉ **sau** khi S1 đã chứng minh TBT↓ trên GPU thuê:

- Warmup AOT Triton/vLLM cache **trên Hopper**, nhét vào cùng tag (hoặc `:p7b`).  
- Không nộp AOT đơn độc kỳ vọng 65–70.

### Tầng S3 — Kỷ luật Portal

```text
IF local_or_rented_bench.TBT_median > 3.7 ms:  DO NOT SUBMIT
IF boot_smoke fails:                           DO NOT SUBMIT
IF logits_delta vs #10 quá lớn:                DO NOT SUBMIT
ELSE: submit ONE canary, record ablation row
```

---

## 4. Làm được gì trên **chỉ một máy Mac**?

Mac **không** chạy H200. Plan chia 2 vòng đời:

### Vòng A — Mac (90% công việc trí tuệ · miễn phí)

| Việc | Output |
|---|---|
| Đọc source LFM2 trong image `:p2-fi` / docs vLLM | Bản đồ file kernel hiện tại |
| Viết Triton fuse prototype (logic + unit test CPU/numpy) | `kernels/shortconv_fuse.py` |
| Viết Dockerfile.p7 + build script “GPU required” | Sẵn sàng 1 lệnh trên máy thuê |
| Viết `bench_tbt.py` (OpenAI stream → TTFT/TBT) | Đo được ngay khi có GPU |
| Viết checklist preflight = tử thần §1.2 | Không nộp sai pattern |
| Giữ Docker Hub login + script push | Như p5 (Mac build amd64 được nếu binary CUDA đã compile sẵn trong context — thường **phải build trên CUDA host**) |

### Vòng B — Thuê GPU **tối thiểu** (bắt buộc cho S1)

Không cần mua máy. **1–3 giờ** RunPod / Vast / Lambda **H100/H200** (hoặc A100 kém hơn một chút):

```text
1. rsync repo + weights
2. bash scripts/build_push_p7_shortconv.sh   # compile SM90 + bake /model từ p2-fi recipe
3. bench_tbt.py against #10 baseline container
4. IF TBT≤3.5: push :p7-shortconv, về Mac chỉ cp compose + nộp
```

**Mac alone không compile CUDA SM90 được** — đó là ràng buộc vật lý, không phải thiếu cố gắng. Plan “outstanding” = **tối ưu số giờ GPU thuê → 0 nếu bench fail sớm**.

---

## 5. Lộ trình 7 ngày (classic campaign)

### Ngày 1 — Archaeology (Mac)
- `docker run --entrypoint bash :p2-fi` → tìm `lfm2`, `causal_conv`, `short_conv` trong site-packages.  
- Ghi: entry kernel, có/không `causal_conv1d`, version flashinfer.  
- Deliverable: `eval/shortconv_archaeology.md`

### Ngày 2–3 — Design + Triton draft (Mac)
- Spec fuse: input `[B,D,L=3 cache + 1]` → output hidden.  
- Unit test vs PyTorch reference conv (CPU).  
- Deliverable: `kernels/` + tests xanh trên Mac.

### Ngày 4 — Dockerfile.p7 (Mac viết · GPU chạy)
- FROM `vllm/vllm-openai:v0.25.1`  
- Install flashinfer (như p2-fi)  
- Build `causal-conv1d` / custom wheel `TORCH_CUDA_ARCH_LIST=9.0+PTX`  
- COPY BF16 weights (như p2-fi) — **vẫn online `--quantization=fp8` như #10**  
- **Cấm** COPY compressed-tensors p5  

### Ngày 5 — Thuê GPU · Build + Bench
- Baseline: chạy container #10 equivalent, đo TBT.  
- Candidate: `:p7-shortconv`, đo TBT.  
- Gate §3 S3.

### Ngày 6 — (Optional) AOT bake nếu TBT đã thắng
- Warmup trên Hopper, commit cache vào tag `p7` hoặc `p7-aot`.

### Ngày 7 — Portal canary **một lần**
- Compose = #10 CLI nguyên văn, **chỉ đổi `image:`**.  
- Ghi ablation.  
- Thắng → freeze. Thua → dừng, giữ #10.

---

## 6. Compose nộp khi (và chỉ khi) gate đạt

```yaml
# image ONLY change vs Yoshio #10
image: longquanton/develarper-lfm25:p7-shortconv
# command: y hệt #10 kể cả --quantization=fp8
```

Mọi flag khác = phản bội định lý §0.

---

## 7. Expect điểm (thành thật · gắn TBT)

| Kết quả bench thuê GPU | Hành động | Expect Portal |
|---|---|---|
| TBT vẫn ~4.0 | **Không nộp** | — |
| TBT ~3.5–3.7 | Nộp 1 canary | **~64–67** |
| TBT ~3.0–3.4 | Nộp 1 canary | **~68–71** |
| TBT ≥4 hoặc boot lỗi | Abort | Giữ **61.66** |

Không hứa 70 trước khi thấy TBT ≤3 trên GPU thật.

---

## 8. Tại sao đây là plan “outstanding” chứ không phải slide ảo

1. **Bám evidence Portal**, không bám ước mơ roadmap 85–95.  
2. **Đúng chỗ còn lại** trong model (10/16 conv), không lặp họ đòn đã chết.  
3. **Tách Mac / GPU** — dùng hết máy hiện có, không giả vờ compile CUDA trên Mac.  
4. **Go/no-go** trước Portal — hết kiểu Z2/Z3/p5 đốt điểm.  
5. **Giữ #10** làm bảo hiểm — cuộc thi là điểm tốt nhất, không phải lần nộp cuối.

---

## 9. Việc làm **ngay trên Mac hôm nay** (bắt đầu Operation SHORTCONV)

1. Giữ `docker-compose.yml` = pure #10 (đã revert).  
2. Tạo nhánh làm việc / folder `kernels/` + `eval/shortconv_archaeology.md`.  
3. Chạy archaeology trên image `:p2-fi` (Docker Desktop đã có).  
4. Song song: chọn vendor GPU thuê (H100/H200), chuẩn bị SSH key + budget 1–3 giờ.  
5. **Không** nộp Portal cho đến khi có số TBT bench.

---

## 10. Thẻ bỏ túi

```text
VÀNG:  #10 61.66 — không đụng CLI
CẤM:   env OMP | bt=256 | seqs=80 | speculative | static FP8 p5 | cờ bịa
MỞ:    short-conv SM90 / fuse · tag p7 · đo TBT rồi mới nộp
TOÁN:  TBT 3.5 → ~65+ ; TBT 3.0 → ~70 ; TBT 4 = trần serving
MÁY:   Mac = nghiên cứu + Docker + Hub ; GPU thuê = compile + bench + bake
```

---

**Ký tên plan:** Operation SHORTCONV — *một đòn đúng chỗ, một lần nộp khi đã đo.*

Bước tiếp theo nếu team approve: **Day 1 archaeology trên `:p2-fi`** (mở container, map kernel LFM2 conv) — làm được 100% trên Mac + Docker.
