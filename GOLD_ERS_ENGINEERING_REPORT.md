# Canary p5-sf8 — Yoshio #10 + Static FP8 only
## Target band: 65–70 (expect trung vị ~64–66 nếu TBT xuống; 70 = stretch)

**Baseline vàng:** Yoshio #10 · ERS **61.66** · TBT 4 · TTFT 48/70 · image `:p2-fi`  
**Yếu tố mới (một đòn):** offline **FP8_DYNAMIC** (llm-compressor) bake vào `:p5-sf8`  
**Cấm trong canary này:** OMP env · AOT gộp · bt=256 · speculative

---

## Khác #10 chỗ nào?

| | #10 gold | p5-sf8 |
|---|---|---|
| Image | `:p2-fi` (BF16 trong `/model`, online `--quantization=fp8`) | `:p5-sf8` (FP8 checkpoint trong `/model`) |
| `--quantization=fp8` | Có | **Bỏ** (đã nén sẵn) |
| Còn lại CLI | flashinfer+align+bt512+block32+8192+mem0.96+kv-fp8 | **Giữ nguyên** |

---

## Lệnh triển khai (máy có GPU + Docker)

```bash
# Hub login + Docker up trước
pip install llmcompressor transformers

# All-in-one: quant → build → push → copy compose
bash scripts/prepare_and_submit_p5_sf8.sh

# Hoặc tách bước:
python3 scripts/quant_fp8.py \
  --model model_weights/LFM2.5-1.2B-Instruct \
  --out model_weights/LFM2.5-1.2B-Instruct-FP8
bash scripts/build_push_p5_sf8.sh
docker pull longquanton/develarper-lfm25:p5-sf8
cp submit/docker-compose.p5_sf8.yml docker-compose.yml
# Upload docker-compose.yml lên Portal
```

**Máy Mac không GPU:** chỉ dry-run được (`python3 scripts/quant_fp8.py --dry-run`). Quant thật + build phải trên VM CUDA.

---

## Gate đọc điểm Portal

| tbt_median_ms | Expect ERS |
|---|---|
| vẫn **4** | ~61–63 (hòa #10) |
| **≤ 3.5** | **~65–67** ← mục tiêu chính |
| **≤ 3.0** | cửa **~70** |
| TTFT p95 ≫ 70 | nghi boot/quant sai path — so với #10 |

---

## File

| Path | Role |
|---|---|
| `Dockerfile.p5_sf8` | Image static FP8 |
| `scripts/quant_fp8.py` | Offline quant |
| `scripts/build_push_p5_sf8.sh` | Build/push |
| `scripts/prepare_and_submit_p5_sf8.sh` | Full pipeline |
| `submit/docker-compose.p5_sf8.yml` | Compose nộp |
| `docker-compose.yml` | Giữ pure #10 until bạn `cp` sau push |

`p6` combo (SF8+AOT) giữ trong repo nhưng **không** dùng cho canary này.
