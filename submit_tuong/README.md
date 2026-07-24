# submit_tuong — Patch Platform submission

Base: **Yoshio `fp8_flash` (61.66 pts)** + `develarper_opt/` patch platform.

**vLLM version (locked):** `v0.25.1` — cùng Yoshio (`vllm/vllm-openai:v0.25.1`).

| File | Mục đích |
|---|---|
| `docker_compose_fp8_flash_opt.yaml` | **Bài nộp chính.** SOTA flags + platform + payload p01. |
| `docker_compose_cudagraphs_pivot.yaml` | **Pivot** khi Gate 2 FAIL: giữ SOTA + ép `--compilation-config` capture CUDA graphs. |

## Image

```
asterios2707/develarper-agent:opt-platform-v1
```

### Cách build image riêng (khuyên dùng — 1 Dockerfile, không phụ thuộc Yoshio Hub)

```bash
make download-model
docker build --platform linux/amd64 -f Dockerfile.tuong \
  -t <YOUR_DOCKERHUB>/develarper-lfm25:tuong-opt-v1 .
docker push <YOUR_DOCKERHUB>/develarper-lfm25:tuong-opt-v1
```

Sửa `image:` trong compose thành `<YOUR_DOCKERHUB>/develarper-lfm25:tuong-opt-v1`.

### Cách build layer trên Yoshio image (nhanh hơn, vẫn dùng v0.25.1)

Build từ `Dockerfile.opt_platform` với base `asterios2707/develarper-agent:latest`
(Yoshio đã bake weights + v0.25.1 + flashinfer).

## Runtime env

| ENV | Default | Ý nghĩa |
|---|---|---|
| `ENABLE_DEVELARPER_OPT` | `1` | Master kill-switch |
| `DEVELARPER_PAYLOADS` | `p01_fused_rmsnorm` | Danh sách payload bật |
| `DEVELARPER_STRICT` | `0` | **Phải là 0** cho BTC (fail-open) |
| `DEVELARPER_LOG_LEVEL` | `INFO` | Log verbosity |

## Rollback nhanh (không rebuild image)

- Tắt hoàn toàn platform → giữ hành vi Yoshio nguyên bản:
  `ENABLE_DEVELARPER_OPT=0`
- Bật thêm payload p02:
  `DEVELARPER_PAYLOADS=p01_fused_rmsnorm,p02_fused_silu_mul`
