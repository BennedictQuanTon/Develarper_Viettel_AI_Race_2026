# Báo cáo CLI tối ưu — `p7-oneshot` (ERS 62.01)


| | |
|---|---|
| **Bài vàng (CLI)** | `p7-oneshot` |
| **ERS / Score** | **62.01** |
| **TBT median** | **4 ms** |
| **TTFT p50 / p95** | **48 / 68 ms** |
| **Fail** | **5 / 420** |
| **Image** | `longquanton/develarper-lfm25:p7-oneshot` (vLLM **0.26.0**) |
| **Compose** | `submit/docker-compose.p7_oneshot.yml` |
| **Kết luận** | Đây là cấu hình **CLI serving tốt nhất** đội đã đạt. Vặn thêm CLI (mbt, seqs, mem, OMP…) không vượt được; kernel canary (p8/p9) cũng không phá trần TBT=4. |

---

## 1. Cấu hình nộp bài (gold)

```yaml
services:
  model:
    image: longquanton/develarper-lfm25:p7-oneshot
    entrypoint: [python3, -m, vllm.entrypoints.openai.api_server]
    command:
      - --model=/model
      - --served-model-name=LFM2.5-1.2B-Instruct
      - --host=0.0.0.0
      - --port=8000
      - --max-model-len=8192
      - --gpu-memory-utilization=0.96
      - --tensor-parallel-size=1
      - --enable-prefix-caching
      - --quantization=fp8
      - --kv-cache-dtype=fp8
      - --enable-chunked-prefill
      - --max-num-batched-tokens=768
      - --max-num-seqs=128
      - --block-size=32
      - --mamba-backend=flashinfer
      - --mamba-cache-mode=align
    ports: ["8000:8000"]
    shm_size: "2g"
```

### Knobs quan trọng

| Knob | Giá trị | Vai trò |
|---|---|---|
| `--quantization=fp8` + `--kv-cache-dtype=fp8` | ON | Win lớn nhất: TBT 6→4 ms (~+9.8 ERS từ P0) |
| `--mamba-backend=flashinfer` | ON | SSM path nhanh hơn default |
| `--mamba-cache-mode=align` | ON | Cache Mamba state theo block → TTFT ổn với prefix dài |
| `--enable-prefix-caching` | ON | System prompt ~1k tokens tái dùng |
| `--enable-chunked-prefill` | ON | Prefill dài không block decode quá nặng |
| `--max-num-batched-tokens` | **768** | Sweet spot TTFT trên p7 (không dùng 512) |
| `--max-num-seqs` | **128** | Cân bằng concurrency 70 conv |
| `--max-model-len` | 8192 | Đủ workload, tiết kiệm KV |
| `--block-size` | 32 | Khớp FlashInfer / align |
| `--gpu-memory-utilization` | 0.96 | An toàn trên 18GB MiG |

---


## 2. Vì sao đây là “CLI tối ưu nhất”

1. **TBT = 4 ms** — sau FP8, mọi thử CLI/kernel nhẹ **không** xuống dưới 4. Trần decode median gắn với CPU scheduler (3 vCPU) hơn là “thiếu flag”.
2. **TTFT 48/68** — tốt ngang Z1; p95 tốt hơn Backup/#10 → nhích ~+0.3–0.4 ERS lên **62.01**.
3. **Fail 5/420** — ổn định; không đổi được bằng vặn `mbt` trên đúng image p7.
4. **Không cần patch kernel** — chỉ image + `docker-compose` flags → dễ rollback, dễ share team.

---

## 3. Thử sau p7 (để team khỏi lặp)

| Canary | ERS | Δ vs p7 | Kết luận |
|---|---|---|---|
| **p7-mbt512** (chỉ `mbt` 768→512) | 61.05 | **−0.96** | Fail vẫn 5; TTFT p95 **68→77**. Đóng residual CLI. |
| p8 ShortConv fuse | 61.90 | −0.11 | TBT vẫn 4 |
| p9 fuse + RMSNorm + mem/cudagraph | 61.54 | −0.47 | TBT vẫn 4 |

**Không khuyến nghị lặp:** speculative n-gram, `mamba-backend=CUDA`, OMP/TOKENIZERS env, offline static FP8, `bt=256`+seqs thấp, fake CLI flags, `num-scheduler-steps` trên v0.26.

---

## 4. Khuyến nghị cho team

1. **Giữ `p7-oneshot` + compose gold (`mbt=768`) làm baseline / bài tốc độ chính.**
2. GPQA shortlist nên có **P0 (accuracy)** + **p7 (speed)**; không chọn p7-mbt512 / p8 / p9 làm champion tốc độ.
3. Muốn nhảy **65+** cần phá **TBT ≪ 4** (profiling H200 / đường khác CLI) — **không** kỳ vọng từ vặn thêm serving flags trên Mac-only.

---



