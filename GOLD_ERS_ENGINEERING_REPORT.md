# GOLD ERS Report — sau Z3-ENV 58.34 · hướng gộp Static FP8 + AOT
## Viettel AI Race 2026 · Team Develarper · 2026-07-25

---

## Kết quả vừa rồi

| | Yoshio #10 | Z3-ENV |
|---|---|---|
| ERS | **61.66** | **58.34** |
| TBT | 4 ms | 4 ms |
| TTFT p50/p95 | 48/70 | 56/88 |

→ **Cấm** OMP/TOKENIZERS env pin. Compose vàng = pure #10 trên `:p2-fi`.

---

## Plan đang làm: **một tag gộp hai hướng** = `p6-sf8-aot`

Bạn chọn **hai hướng một lúc** → một image / một lần nộp:

| Trong image | Mục tiêu metric |
|---|---|
| **Static FP8** (`llm-compressor` FP8_DYNAMIC → `/model`) | Hạ **TBT** dưới 4ms |
| **AOT warmup** (Triton/vLLM cache bake lúc build) | Ổn **TTFT** / bớt cold-start |

**Rủi ro gộp:** nếu điểm tụt, không biết cái nào hại → lần sau tách `p5-sf8` và `p4-aot`.

**Đúng luật:** quant + CUDA graphs được phép; không dual-path; tag mới không đè `:p2-fi`.

---

## Việc phải làm (máy **có GPU** + Docker)

Máy Mac hiện tại: **Docker down + không GPU** → chưa build/push được tại chỗ. Làm trên VM CUDA (cloud):

```bash
# 1) Quant offline
pip install llmcompressor transformers
python3 scripts/quant_fp8.py \
  --model model_weights/LFM2.5-1.2B-Instruct \
  --out model_weights/LFM2.5-1.2B-Instruct-FP8

# 2) Build + push (AOT bật; cần GPU lúc docker build)
chmod +x scripts/build_push_p6_combo.sh scripts/warmup_aot_cache.sh
ENABLE_AOT=1 bash scripts/build_push_p6_combo.sh

# Nếu build host không GPU cho AOT:
# ENABLE_AOT=0 bash scripts/build_push_p6_combo.sh   # chỉ static FP8

# 3) Verify pull rồi nộp
docker pull longquanton/develarper-lfm25:p6-sf8-aot
cp submit/docker-compose.p6_combo.yml docker-compose.yml
# upload docker-compose.yml lên Portal
```

### Compose khác #10 chỗ nào?
- Image → `:p6-sf8-aot`
- **Bỏ** `--quantization=fp8` (weights đã compressed-tensors)
- Giữ kv-fp8, flashinfer, align, bt=512, block32, maxlen 8192, mem 0.96
- **Không** `environment:` OMP

---

## File mới

| File | Vai trò |
|---|---|
| `scripts/quant_fp8.py` | Offline FP8 |
| `scripts/warmup_aot_cache.sh` | Warmup cache |
| `Dockerfile.p6_combo` | Image gộp |
| `scripts/build_push_p6_combo.sh` | Build/push |
| `submit/docker-compose.p6_combo.yml` | File nộp sau khi Hub có image |
| `docker-compose.yml` | Vẫn pure #10 (an toàn) cho đến khi p6 sẵn sàng |

---

## Thẻ cấm

```text
CẤM: OMP env pin | bt=256 | seqs=80 | speculative | đè :p2-fi
NỘP p6: chỉ khi docker pull :p6-sf8-aot OK
```
