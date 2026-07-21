# SUBMIT — Phase 1 nộp bài (checklist vận hành)

## Bạn nộp gì?

1. **Docker image public** trên Docker Hub (weights đã bake tại `/model`, runtime offline)
2. File **`docker-compose.yml`** trên Portal (dùng [`docker-compose.yml`](docker-compose.yml))

BTC pull image → chạy trên **MiG H200 18GB** → healthcheck → chấm **ERS**.

## Lệnh chuẩn (máy có Docker Desktop đang chạy + mạng)

> Mac: mở **Docker Desktop** trước (`docker info` phải OK). Build `linux/amd64` qua buildx.

```bash
# 0) Đăng nhập Hub (một lần)
docker login

# 1) Tải model (~2–3GB)
make download-model

# 2) Build linux/amd64 (Mac M-series vẫn build được qua buildx)
make build IMAGE_REPO=<username>/develarper-lfm25 TAG=p0

# 3) Kiểm tra local
make preflight   # sẽ FAIL nếu vẫn còn YOUR_DOCKERHUB trong compose — đúng kỳ vọng trước khi sửa

# 4) Push public
make push IMAGE_REPO=<username>/develarper-lfm25 TAG=p0

# 5) Sửa docker-compose.yml:
#    image: <username>/develarper-lfm25:p0
#    (tốt hơn: pin digest sha256:... từ make push)

# 6) Đổi tên / upload file đó lên Portal như docker-compose.yml
```

## Vì sao không dùng đúng `v0.22.1` của mẫu BTC?

Docs Liquid/vLLM: LFM2.5 cần **vLLM ≥ 0.23.0**. Image mặc định trong `Dockerfile` là `vllm/vllm-openai:latest-cu130` (CUDA 13 khớp driver 590).

Nếu BTC bắt buộc đúng tag 0.22.1: thử `make build-baseline` và verify load; fail thì giữ ≥0.23.

## Thứ tự nộp ablation (1 file compose / lần)

| Thứ tự | File | Thay đổi |
|---|---|---|
| P0 | `docker-compose.yml` | prefix ON, mem=0.95 |
| A1 | `submit/docker-compose.a1_mem90.yml` | mem=0.90 |
| A2 | `submit/docker-compose.a2_chunked.yml` | chunked prefill |
| B1 | `submit/docker-compose.b1_fp8.yml` | `--quantization=fp8` |
| B2 | `submit/docker-compose.b2_kvfp8.yml` | `--kv-cache-dtype=fp8` |

Ghi ERS vào [`eval/ablation_sheet.md`](eval/ablation_sheet.md). Giữ config nếu ERS↑ và không lỗi.

## Anti-cheat (bắt buộc tick trước nộp)

- [ ] Image **public**
- [ ] Không gọi mạng ngoài lúc container chạy
- [ ] `/model` nằm trong image
- [ ] Entrypoint: `python3 -m vllm.entrypoints.openai.api_server`
- [ ] Prefix caching ON (trừ khi A/B chứng minh hại)
- [ ] Không dual-path / hardcode

## Việc máy Mac không làm được

- Không đo TTFT/TPOT thật trên H200 MiG → lab = **mỗi lần nộp Portal**
- Build image CUDA trên Mac OK; **chạy** container GPU cần máy NVIDIA (hoặc tin BTC)

## ERS sim nhanh (ranking, không phải điểm thật)

```bash
make ers-sim
```
