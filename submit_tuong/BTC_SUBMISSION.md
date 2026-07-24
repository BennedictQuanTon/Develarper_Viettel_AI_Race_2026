# Hướng dẫn 2 — Chuẩn bị & nộp bài BTC

Dựa trên `AI_Race_2026_Context.md` §6–§10 (quy trình nộp, chống gian lận, tie-break).

**Bối cảnh:** đây là submission cuối. Không có cửa sổ retry cùng ngày (đã hết quota) hoặc rất hẹp. Vì thế mọi bước phải qua checklist trước khi push image ra Docker Hub.

---

## Bản đồ 2 phương án

```
                    ┌──── Gate 2 PASS (TPOT giảm ≥ 5%) ────► Nộp file A
                    │
Gate 2 (Tier 3 A/B) ┤
                    │
                    └──── Gate 2 FAIL ─────────────────────► Nộp file B (pivot)

  File A: submit_tuong/docker_compose_fp8_flash_opt.yaml       (Triton payload p01)
  File B: submit_tuong/docker_compose_cudagraphs_pivot.yaml    (CUDA graphs config)

  Cả hai đều dùng cùng 1 image: asterios2707/develarper-agent:opt-platform-v1
```

**Ưu điểm chung 1 image:** đổi phương án chỉ cần đổi compose (bao gồm ENV `ENABLE_DEVELARPER_OPT` và `--compilation-config`), **không rebuild**.

---

## Bước 1 — Xác minh môi trường build

```powershell
cd d:\Coding\Prj\AiRace26\Develarper_Viettel_AI_Race_2026
git branch --show-current    # phải là feat/tuong
git status                    # commit hết thay đổi trước khi build
```

### 1.1 Đảm bảo Base image Yoshio có sẵn local

```bash
docker pull asterios2707/develarper-agent:latest
docker image inspect asterios2707/develarper-agent:latest \
  --format '{{.Size}} {{.Created}}'
```

Nếu image chưa có, hỏi Yoshio hoặc dùng credentials của team để pull.

### 1.2 Đảm bảo weights đã ở `/model` trong base image

```bash
docker run --rm asterios2707/develarper-agent:latest ls -lah /model
# phải thấy config.json, tokenizer.json, model.safetensors, ...
```

Nếu base image thiếu `/model` (BTC yêu cầu weights baked, không tải mạng) → dừng lại, không build đè.

---

## Bước 2 — Build image opt-platform

```bash
docker build -f Dockerfile.opt_platform \
  -t asterios2707/develarper-agent:opt-platform-v1 .
```

**Bắt buộc thấy trong build log:**

```
sitecustomize installed at /usr/lib/python3.??/site-packages/sitecustomize.py
develarper_opt dry_run OK; registered payloads: ['p01_fused_rmsnorm', 'p02_fused_silu_mul', 'p03_cudagraph_hint']
Triton x.y.z available
```

Nếu `dry_run` fail → **fix trước khi push**. Không dùng `--no-cache` trừ khi cần.

### Cấm

- Không tag `:latest` (đó là SOTA Yoshio 61.66 — không được ghi đè).
- Không dùng cùng tag như bất kỳ submission cũ. Tag `opt-platform-v1` là mới.

---

## Bước 3 — Chạy Tier 3 smoke + A/B

Theo `LOCAL_TESTING.md` §3.2 và §3.3. Bắt buộc:

- [ ] Container start `/health` trong **< 120s**
- [ ] Log có dòng `DEVELARPER_OPT ... [p01_fused_rmsnorm] applied`
- [ ] 20/20 request trả đủ token, không truncate
- [ ] Long-context probe (prompt > 3000 tokens) → output đủ 300 token
- [ ] TPOT B ≤ TPOT A (nếu nộp file A)

Ghi lại metrics vào file:

```bash
bash scripts/opt_ab_compare.sh 2>&1 | tee eval/ab_2026-07-24.log
```

---

## Bước 4 — Push Docker Hub (Public)

```bash
docker login -u asterios2707
docker push asterios2707/develarper-agent:opt-platform-v1
```

Sau push, verify từ máy khác (hoặc `docker system prune -a` rồi pull lại):

```bash
docker pull asterios2707/develarper-agent:opt-platform-v1
docker image inspect asterios2707/develarper-agent:opt-platform-v1 \
  --format 'digest={{.Id}} size={{.Size}}'
```

Ghi lại **digest** vào `eval/portal_notes.md` — nếu BTC hỏi audit, đây là dấu vân tay của image.

---

## Bước 5 — Pre-submit checklist (bám sát `AI_Race_2026_Context.md` §7, §8)

### 5.1 Cấu hình compose

Kiểm tra file **compose sẽ upload** (A hoặc B):

- [ ] Entrypoint = `python3 -m vllm.entrypoints.openai.api_server` (không thay đổi)
- [ ] `--model=/model`, `--served-model-name=LFM2.5-1.2B-Instruct` giữ nguyên
- [ ] `--host=0.0.0.0`, `--port=8000`
- [ ] Không có `--enable-chunked-prefill=true` (chỉ được `- --enable-chunked-prefill`, không `=true`)
- [ ] Không có `--mamba-backend=CUDA`, không có `--speculative-config`
- [ ] Không có `--disable-log-requests`
- [ ] `image:` trỏ đúng `asterios2707/develarper-agent:opt-platform-v1`

### 5.2 Anti-cheat (§8 chống gian lận)

- [ ] Không hardcode / prebake response (image chỉ có weights + kernels + platform)
- [ ] Không dual-path: cùng 1 forward path bất kể request đến từ đâu (dễ vi phạm nhất — hãy đọc lại `_patched_forward`)
- [ ] Không network external: image không có `RUN pip install` sau bake weights, không có kết nối ngoài lúc runtime
- [ ] Không tampering weights / tokenizer: `develarper_opt/` chỉ patch **compute layers**, không đụng weights
- [ ] Không tráo image: chỉ push 1 tag `opt-platform-v1`, không đè sau khi nộp

### 5.3 Accuracy hedge

Vì chỉ can thiệp RMSNorm compute (không đổi weights/quant), rủi ro Δ > 10% thấp — nhưng KHÔNG tự tin tuyệt đối. Đảm bảo trong 5 submissions hậu kiểm còn ít nhất 1 slot cho bài Yoshio SOTA làm bảo hiểm accuracy.

---

## Bước 6 — Nộp Portal BTC

Theo `AI_Race_2026_Context.md` §6:

1. Login portal BTC.
2. Upload **file compose** (không upload image — BTC pull từ Docker Hub).

   - Nếu Gate 2 PASS: `submit_tuong/docker_compose_fp8_flash_opt.yaml`
   - Nếu Gate 2 FAIL: `submit_tuong/docker_compose_cudagraphs_pivot.yaml`

3. Ghi chú submission trên portal (nếu có ô ghi chú): `opt-platform-v1 + p01 RMSNorm` hoặc `opt-platform-v1 + cudagraphs pivot`.
4. Nộp trong khung giờ **có capacity thấp** trên hệ thống BTC nếu có thể (giảm variance benchmark).
5. Ghi nhận **timestamp nộp** vào `eval/portal_notes.md`.

**Deadline:** 30/07/2026 (theo master doc). Đừng nộp quá gần deadline — cần thời gian retry nếu BTC báo lỗi build.

---

## Bước 7 — Sau khi có kết quả

### Nếu điểm > 62 (vượt SOTA):

- Ghi vào `IssueAnalysis/Tuong-<date>.md`: config, tag, digest, ERS, TTFT/TPOT.
- Cập nhật `master_technical_reference.md` §11 với record mới.
- Tag git: `git tag -a opt-platform-v1 -m "opt-platform-v1: ERS <x>"`.
- Đối với hậu kiểm 5 submissions: chọn **cả** bài này **và** Yoshio SOTA (accuracy hedge).

### Nếu điểm ≤ 62 (không cải thiện, patch fail-open):

- Không mất submission "chết": platform đã live trên hạ tầng BTC, có bằng chứng chạy được.
- Sau vòng online, nếu team còn cửa sổ, mở payload p02/p03 trên cùng image mà không phải rebuild.

### Nếu điểm giảm rõ rệt hoặc fail:

- Đọc log qua portal (BTC có tab logs).
- Kiểm tra 3 khả năng theo thứ tự:
  1. Anti-cheat probe fail → xem có truncation không (bug trong `_patched_forward` với residual dài).
  2. Warmup timeout → giảm capture sizes trong file B, không dùng file A.
  3. Accuracy Δ > 10% → chỉ xảy ra khi hậu kiểm GPQA, không phải vòng online.

---

## Tie-break awareness (`AI_Race_2026_Context.md` §9)

Nếu điểm gần đối thủ trong biên ≤ 3 điểm, BTC dùng thứ tự:

1. `Δ` accuracy thấp hơn thắng → **giữ Yoshio SOTA làm bảo hiểm khi chọn 5 hậu kiểm**.
2. **p95 TTFT** thấp hơn thắng → File A có thể tăng TTFT nhẹ; nếu điểm ngang, ưu tiên file B (cudagraphs, TTFT tốt hơn).
3. Token/s cao hơn thắng → decode throughput; File A hướng vào đây.
4. Nộp sớm thắng → **nộp càng sớm càng tốt sau khi Tier 3 PASS**.

---

## Phụ lục — 1 file compose = 1 lệnh reset không rebuild

Trong trường hợp cần dùng lại image `opt-platform-v1` cho lần nộp sau nhưng muốn:

- Tắt hẳn platform: sửa `ENABLE_DEVELARPER_OPT=0` trong compose.
- Bật thêm p02: `DEVELARPER_PAYLOADS=p01_fused_rmsnorm,p02_fused_silu_mul`.
- Bật CUDA Graphs cùng lúc với p01: thêm dòng `- --compilation-config={"cudagraph_capture_sizes":[1,2,4,8,16,32,64,128]}` vào `command:`.

Không cần build/push lại image — chỉ upload compose mới.
