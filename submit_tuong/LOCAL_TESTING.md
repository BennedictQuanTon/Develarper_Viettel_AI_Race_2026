# Hướng dẫn 1 — Chạy test local

Toàn bộ platform được thiết kế để **test được không cần H200**. Có 3 tầng test tách bạch:

```
Tier 1 · Unit tests (CPU, <10s)      ← chạy đầu tiên, chạy mỗi lần chỉnh code
Tier 2 · Kernel tests (GPU, <30s)     ← chạy khi có GPU (bất kỳ NVIDIA GPU nào)
Tier 3 · E2E serve tests (Docker+GPU) ← chạy trước khi build image submit
```

---

## 0. Cài đặt môi trường 1 lần

```powershell
cd d:\Coding\Prj\AiRace26\Develarper_Viettel_AI_Race_2026
python -m pip install -r requirements-dev.txt
```

Trên máy có GPU thêm Triton (Linux/WSL) hoặc chạy trong Docker (xem Tier 3):

```bash
python -m pip install triton
```

---

## Tier 1 — Unit tests (không cần GPU)

Chạy toàn bộ:

```powershell
python -m pytest develarper_opt/tests -q
```

Kỳ vọng output cuối:

```
41 passed, 3 skipped in 8.xx s
```

3 skipped là các test gắn marker `@pytest.mark.cuda` — chúng sẽ tự chạy khi có GPU.

### Chạy từng nhóm

```powershell
# Chỉ flags parsing (nhanh nhất, <1s)
python -m pytest develarper_opt/tests/test_flags.py -v

# Chỉ registry lifecycle
python -m pytest develarper_opt/tests/test_registry.py -v

# Chỉ correctness của kernels (CPU fallback)
python -m pytest develarper_opt/tests/test_kernels_correctness.py -v

# Chỉ payload p01 (mock vLLM)
python -m pytest develarper_opt/tests/test_payload_p01.py -v

# Chỉ sitecustomize hook (subprocess test)
python -m pytest develarper_opt/tests/test_sitecustomize.py -v
```

### Bảng test — nội dung + tiêu chí PASS

| File | Số test | Mục đích | Gate |
|---|---|---|---|
| `test_flags.py` | 9 | Parse ENV → `Flags` dataclass | Boolean/list/default đúng, STRICT mặc định = 0 |
| `test_registry.py` | 7 | Đăng ký / apply / fail-open / idempotent | Master OFF → không apply; payload lỗi → không raise |
| `test_kernels_correctness.py` | 8 (CPU) + 3 (GPU) | RMSNorm & SiLU×Mul vs PyTorch ref | `max_diff < 1e-5` CPU, `< 1e-2` GPU fp16 |
| `test_payload_p01.py` | 7 | Monkey-patch RMSNorm trên fake vLLM | Output/residual khớp reference; `restore()` revert |
| `test_sitecustomize.py` | 3 | Hook script không raise ở bất kỳ ENV nào | Exit 0, `dry_run` liệt kê đúng payload |

### Ví dụ output chi tiết 1 test

```powershell
python -m pytest develarper_opt/tests/test_kernels_correctness.py::TestRMSNormFallbackMatchesReference -v
```

```
PASSED develarper_opt/tests/.../test_matches_reference_on_cpu[hidden=1024-batch=1]
PASSED develarper_opt/tests/.../test_matches_reference_on_cpu[hidden=2048-batch=32]
...
```

---

## Tier 2 — Kernel tests trên GPU

Trên máy có NVIDIA GPU + Triton (WSL2 hoặc Linux):

```bash
cd Develarper_Viettel_AI_Race_2026
python -m pytest develarper_opt/tests -q -m cuda
```

Test sẽ báo kết quả **benchmark thực**:

```
[RMSNorm decode-batch] triton=42.1us ref=68.5us speedup=1.63x
```

### Gate 2 — quyết định go/no-go nộp bài

| Chỉ số | Ngưỡng bắt buộc | Ghi chú |
|---|---|---|
| `max_diff` RMSNorm fp16 | `< 1e-2` | Correctness fail-safe |
| `max_diff` SiLU×Mul fp16 | `< 1e-2` | Correctness fail-safe |
| Speedup RMSNorm batch=1 | `>= 1.3x` | Trên H200; máy khác chỉ cần `>= 1.05x` (test cứng) |
| Warmup Triton JIT | `< 60s` | Đo bằng `time python -c "..."` |

Nếu **speedup < 1.3× trên H200** → **KHÔNG nộp bản Triton**, chuyển sang pivot CUDA Graphs.

### Bật cả 2 payload (khi p01 đã ổn định)

```bash
DEVELARPER_PAYLOADS=p01_fused_rmsnorm,p02_fused_silu_mul \
python -m pytest develarper_opt/tests -q -m cuda
```

---

## Tier 3 — E2E serve test (Docker + GPU)

Đây là bước gần BTC nhất. Yêu cầu: Docker + NVIDIA Container Toolkit + weights đã bake vào image base `asterios2707/develarper-agent:latest`.

### 3.1 Build image opt-platform

```bash
cd Develarper_Viettel_AI_Race_2026
docker build -f Dockerfile.opt_platform \
  -t asterios2707/develarper-agent:opt-platform-v1 .
```

Build sẽ **fail sớm** nếu:
- Sitecustomize không copy được vào site-packages
- `dry_run` phát hiện payload lỗi syntax hoặc import missing
- Triton không có trong base image (warning, không fail — payload sẽ fail-open)

### 3.2 Smoke test 1 lệnh

```bash
bash scripts/opt_smoke.sh submit_tuong/docker_compose_fp8_flash_opt.yaml
```

Script sẽ:
1. `docker compose up -d` compose file
2. Chờ `/health` (tối đa 120s)
3. Bắn 20 request streaming song song 4
4. In TTFT/TPOT p50/p95 + approx ERS
5. Grep log kiểm tra `DEVELARPER_OPT` events
6. Tear down

**Kỳ vọng log:**

```
DEVELARPER_OPT INFO ...: platform boot | enabled=True strict=False payloads=['p01_fused_rmsnorm'] ...
DEVELARPER_OPT INFO ...: [p01_fused_rmsnorm] applied: RMSNorm.forward replaced by triton_rmsnorm
DEVELARPER_OPT INFO ...: platform ready | applied=1/1
```

Nếu thấy `status=skipped` hoặc `status=failed` → **KHÔNG nộp** — debug trước.

### 3.3 A/B compare vs Yoshio SOTA

```bash
A_COMPOSE=submit_yoshio/docker_compose_fp8_flash.yaml \
B_COMPOSE=submit_tuong/docker_compose_fp8_flash_opt.yaml \
bash scripts/opt_ab_compare.sh
```

**Quyết định go/no-go nộp:**

| Điều kiện | Hành động |
|---|---|
| TPOT B < TPOT A (rõ ràng, ví dụ ≥ 5%) và fail B ≤ fail A | ✅ Nộp `docker_compose_fp8_flash_opt.yaml` |
| TPOT B ≈ TPOT A, fail B ≤ fail A | ⚠️ Cân nhắc — patch trung tính, không hại. Có thể nộp. |
| TPOT B > TPOT A hoặc fail B > fail A | ❌ **Pivot** sang `docker_compose_cudagraphs_pivot.yaml` |

### 3.4 Test pivot CUDA Graphs

```bash
bash scripts/opt_smoke.sh submit_tuong/docker_compose_cudagraphs_pivot.yaml
```

Kỳ vọng log:

```
Capturing CUDA graph shapes: [1, 2, 4, 8, 16, 32, 64, 128]
Graph capturing finished in ~28s, took ~1.2 GiB
```

Nếu thấy `Enforce eager mode is set. Skip CUDA graph capturing.` → capture không xảy ra, không nộp pivot này.

---

## Cheat-sheet dòng lệnh

```powershell
# Tier 1 — full unit tests
python -m pytest develarper_opt/tests -q

# Kiểm tra flags reload đúng theo ENV
$env:DEVELARPER_PAYLOADS = "p01_fused_rmsnorm,p02_fused_silu_mul"
python -c "from develarper_opt.platform.feature_flags import load_flags; print(load_flags())"

# Dry-run trước build Docker
python -c "from develarper_opt.platform.registry import dry_run; dry_run()"

# Đọc log platform lúc container chạy
docker logs <container> 2>&1 | Select-String DEVELARPER_OPT
```

## Debug thường gặp

| Triệu chứng | Nguyên nhân | Fix |
|---|---|---|
| `dry_run` fail `No module named triton` | Base image thiếu Triton | Đổi base sang image có triton, hoặc chấp nhận fail-open |
| `test_payload_p01` fail `vllm_rmsnorm_import_failed` | vLLM chưa cài local | Bình thường; test này có mock, chỉ fail nếu mock lỗi |
| Container start > 120s | Triton JIT + CUDA graph warmup | Giảm `--compilation-config` capture sizes |
| Log không có `DEVELARPER_OPT` | Sitecustomize không được load | Kiểm tra `docker exec ls /usr/lib/python3/dist-packages/sitecustomize.py` |
| Log có `master switch OFF` | `ENABLE_DEVELARPER_OPT` bị override | Ktra `environment:` trong compose |
