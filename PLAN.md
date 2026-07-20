# PLAN.md — Execution Playbook Phase 1 (Updated)
## Viettel AI Race 2026 · Challenge 3 · Team Develarper

> Chiến lược / lý do: [CONTEXT.md](CONTEXT.md) · Đề: [PROBLEM_VN.md](PROBLEM_VN.md)  
> **Model:** `LiquidAI/LFM2.5-1.2B-Instruct` · **Slice:** MiG H200 **18GB** · **Engine:** vLLM only  
> **Deadline Phase 1:** 30/07/2026

---

## 0. Hiểu đề trong 1 phút

| Câu hỏi | Trả lời ngắn |
|---|---|
| Làm gì? | Serve model 1.2B bằng vLLM cho nhanh trên MiG 18GB |
| Điểm online? | ERS từ TTFT + TPOT (w=0.5, γ=2) |
| Floor/Ceiling? | TTFT 10–400ms · TPOT **1–10ms** |
| Sau online? | Chọn ≤5 bài → GPQA; baseline ~0.4; Score=100×ERS×f(Δ) |
| Nộp gì? | Docker Hub **public** image + `docker-compose.yml` |
| Đòn bẩy #1? | **Prefix caching** (shared system prefix) |

### 10 ngày — học thuộc

1. **Ngày 1–2:** Compose đúng mẫu BTC + bake model + verify boot (local GPU nếu có)  
2. **Ngày 3:** ERS sim với params thật + hiểu trace fields  
3. **Ngày 4–5:** Push Hub + nộp **P0** (prefix ON, gần baseline flags) lấy ERS  
4. **Ngày 6–7:** Ablation 1 biến/lần (mem-util, batched-tokens, chunked, quant)  
5. **Ngày 8–9:** Chọn winner; smoke GPQA; anti-cheat  
6. **Ngày 10:** Freeze digest + nộp cuối  

**AMD:** không còn bắt buộc (model 1.2B). Giữ credit dự phòng nếu cần GPU NVIDIA tạm.

---

## 1. North Star & ERS params (khóa)

```text
Score = 100 × ERS × f(Δ)
```

| Param | Value |
|---|---|
| F_ttft / C_ttft | 10 ms / 400 ms |
| F_tpot / C_tpot | 1 ms / 10 ms |
| gamma | 2 |
| w | 0.5 |
| Acc baseline (ref) | 0.4 |

File: [`configs/ers_params.example.json`](configs/ers_params.example.json) đã khớp số này.

---

## 2. Roles

| ID | Owns | Decide |
|---|---|---|
| **AI1** | Weights bake path, online/FP8 quant, GPQA smoke | Có giữ quant không (theo Δ) |
| **AI2** | vLLM flags, ers_sim, ablation sheet | Nút nào submit tiếp |
| **DO** | Dockerfile, Hub push, compose form BTC, portal, checklist | Freeze tag/digest |

Daily 15’: portal / ERS mới / OOM? / submit gì hôm nay?

---

## 3. Repo layout (đã scaffold — cập nhật theo đề)

```text
Dockerfile                 # bake LiquidAI/LFM2.5-1.2B-Instruct -> /model
docker-compose.yml         # BTC entrypoint form + mock profile
docker-compose.submit.yml  # bản nộp Portal (image Hub của đội)
configs/p0_safe.env        # flags P0
configs/p1_aggressive.env
configs/ers_params.example.json
scripts/smoke_openai.py
scripts/ers_sim.py
scripts/mock_openai_server.py
scripts/quant_fp8.py       # optional; model nhỏ
eval/ablation_sheet.md
eval/portal_notes.md
eval/credit_sheet.md
```

---

## 4. Timeline chi tiết

### D1–D2 — Boot hợp lệ

| Task | Owner | Done when |
|---|---|---|
| Confirm vLLM tag load được LFM2.5 (`v0.22.1` trước; fallback tag mới hơn nếu fail) | AI1+DO | `GET /v1/models` OK |
| Dockerfile: COPY/download-at-build weights vào `/model` (build máy có net; **runtime offline**) | DO | Image chạy không cần HF token lúc start |
| `docker-compose.yml` đúng entrypoint mẫu BTC | DO | Compose up → port 8000 |
| `smoke_openai.py` stream pass | AI2 | OK |
| Ghi tag vào `eval/portal_notes.md` | DO | |

### D3 — Trace & simulator

| Task | Owner | Done when |
|---|---|---|
| Parse / giả lập theo fields: shared_system_prefix, per_conv_prefix, turns, arrival | AI2 | |
| `ers_sim.py --params configs/ers_params.example.json` | AI2 | Ranking configs |
| Ước concurrency từ `num_conversations` | AI2 | Note 1 trang |

### D4–D5 — P0 submit

| Task | Owner | Done when |
|---|---|---|
| Push image **public** Docker Hub | DO | URL Hub |
| Compose submit: image đội + flags P0 (`--enable-prefix-caching`, mem~0.95, max-model-len hợp lý) | DO+AI2 | |
| Nộp Portal → có ERS | Cả team | ERS trên leaderboard |
| Dòng P0 trong ablation sheet | AI2 | |

**P0 command gợi ý (bắt đầu gần mẫu BTC):**

```text
--model=/model
--served-model-name=LFM2.5-1.2B-Instruct
--host=0.0.0.0 --port=8000
--max-model-len=32768
--gpu-memory-utilization=0.95
--tensor-parallel-size=1
--enable-prefix-caching
```

### D6–D7 — Ablation (1 biến / submit)

Thứ tự:

1. `gpu-memory-utilization`: 0.90 ↔ 0.95  
2. `max-num-batched-tokens` (nếu dùng): 2048 / 4096 / 8192  
3. chunked prefill on/off  
4. online FP8 / quantization flag  
5. `kv-cache-dtype=fp8`  

Rule: ERS↑ và không tăng lỗi → giữ; rollback nếu timeout/OOM.

### D8–D9 — Quality + hygiene

| Task | Owner |
|---|---|
| GPQA smoke (local/Firework tiết kiệm) vs baseline 0.4 | AI1 |
| Anti-cheat checklist | DO |
| Chọn 1–2 digest ứng viên top-5 sau này | Team |

### D10 — Freeze

Pin digest, rebuild sạch, nộp cuối, không sửa sát giờ.

---

## 5. Ablation sheet

Xem [`eval/ablation_sheet.md`](eval/ablation_sheet.md) — cập nhật cột flags cho LFM2.5 / MiG 18GB.

---

## 6. Anti-cheat checklist (mỗi lần nộp)

- [ ] Image public; tag/digest khớp compose  
- [ ] Không mạng ngoài khi container chạy  
- [ ] Weights trong `/model` (hoặc path compose khai báo), không pull HF lúc start  
- [ ] Entrypoint đúng form BTC (`python3 -m vllm.entrypoints.openai.api_server`)  
- [ ] Một behavior (không dual-path)  
- [ ] Streaming chat completions OK  
- [ ] Ghi ablation sheet  

---

## 7. Credit sheet (cập nhật tư duy)

| Nguồn | Cách dùng mới |
|---|---|
| Firework | Ít lần GPQA smoke |
| AMD | Optional; ưu tiên local NVIDIA / BTC submit làm lab |
| BTC submits | Lab ERS chính |

---

## 8. Definition of Done — Phase 1

- [ ] Image Hub public chạy offline trên giả lập GPU  
- [ ] Compose form BTC nộp được, ERS > 0  
- [ ] ≥3 dòng ablation có kiểm soát  
- [ ] Prefix caching ON trên winner  
- [ ] Smoke Δ cảm giác an toàn vs ~0.4  
- [ ] Checklist anti-cheat xanh  

---

## 9. Next actions (submit-ready pipeline)

Đã có trong repo: `Makefile`, `SUBMIT.md`, `submit/*.yml`, bake Dockerfile (≥v0.23/cu130), `ers_sim` theo meta trace, `scripts/set_image.sh`.

```bash
make download-model
make build IMAGE_REPO=<you>/develarper-lfm25 TAG=p0
./scripts/set_image.sh <you>/develarper-lfm25:p0
make push IMAGE_REPO=<you>/develarper-lfm25 TAG=p0
# Portal: upload docker-compose.submit.yml
```

Chi tiết: **[SUBMIT.md](SUBMIT.md)**.
