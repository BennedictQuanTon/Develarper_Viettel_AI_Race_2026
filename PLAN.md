# PLAN.md — Execution Playbook Phase 1  
## Viettel AI Race 2026 · Challenge 3 · Team Develarper

> Plan vận hành A→Z để **execute**. Chiến lược kỹ thuật / lý do xem [CONTEXT.md](CONTEXT.md).  
> **Deadline Phase 1:** 30/07/2026 · **Hôm nay giả định:** 20/07/2026 (~10 ngày).  
> **Constraints:** BTC chưa công bố model / chưa cấp cluster; local 16–32GB ×3; ~100 Firework; ~50 AMD; team = 2 AI + 1 DevOps.

---

## 0. North Star

| Mục tiêu | Đo bằng |
|---|---|
| Có submission Compose chạy trên H200 BTC | Health + điểm ERS trên leaderboard |
| ERS tăng có kiểm soát | Bảng ablation (mỗi lần đổi 1 biến) |
| Sẵn sàng Accuracy Gate | Δ smoke ≪ 0.10; anti-cheat checklist xanh |
| Không cháy credit | Sheet credit cập nhật mỗi ngày |

**Công thức thắng:** `Score = 100 × ERS × f(Δ)` với error rate → 0.

---

## 1. Roles & RACI

| ID | Vai trò | Người | Owns | Decide |
|---|---|---|---|---|
| **AI1** | Model & Quant | Eng AI 1 | Weights FP8, recipe, Δ smoke, chat template | Scheme quant / giữ hay bỏ KV-FP8 về phía accuracy |
| **AI2** | Serving & ERS | Eng AI 2 | vLLM flags, ERS simulator, ablation sheet | Giá trị `batched-tokens`, mem-util, bật/tắt spec |
| **DO** | DevOps & Gate | Eng DevOps | Compose, offline image, smoke scripts, submit log, anti-cheat | Freeze digest nộp bài |

**Daily standup (15’):** (1) portal có model/trace chưa? (2) credit còn? (3) ERS mới? (4) blocker? (5) nộp gì hôm nay?

---

## 2. Credit Budget (cứng)

### 2.1 Firework (~100)

| Khoản | Budget | Owner |
|---|---|---|
| Baseline / proxy accuracy | 30–40 | AI1 |
| Smoke sau quant (vài chục Q) | 20–30 | AI1 |
| Buffer sự cố | 30+ | Lead giữ |

**Cấm:** loop evaluation, so latency qua API.

### 2.2 AMD (~50) — thuê gì?

| Ưu tiên | GPU | Mục đích |
|---|---|---|
| **#1** | **MI300X 192GB** | Quant FP8 + load full model + smoke serve |
| #2 | MI250 / nhỏ hơn | Chỉ khi MI300X hết quota / quá đắt |

**Session plan (tối đa 2–3 session):**

| Session | Mục tiêu | Chuẩn bị trước (bắt buộc) | Owner |
|---|---|---|---|
| **A** | Quant + export FP8 artifact | Script compress đã dry-run syntax local; biết model ID | AI1 + DO |
| **B** | `vllm serve` FP8 + smoke lm_eval/subset | Compose/entrypoint sẵn; list prompt smoke | AI2 + DO |
| **C (buffer)** | Re-quant / fix load lỗi | Chỉ khi A/B fail | AI1 |

**Cấm:** dùng AMD để sweep TTFT/TPOT chọn config thắng.

---

## 3. Repo Skeleton cần dựng (hiện repo gần như chỉ docs)

Tạo cấu trúc tối thiểu (DO lead, AI hỗ trợ):

```text
.
├── CONTEXT.md          # strategy (đã có)
├── PLAN.md             # playbook (file này)
├── PROBLEM.md / PROBLEM_VN.md
├── README.md
├── docker-compose.yml  # Phase 1 submit
├── Dockerfile
├── scripts/
│   ├── serve.sh
│   ├── smoke_openai.py
│   ├── ers_sim.py
│   ├── quant_fp8.py
│   └── lm_eval_smoke.sh
├── configs/
│   ├── p0_safe.env
│   └── p1_aggressive.env
├── eval/
│   └── ablation_sheet.md
└── model_weights/      # gitignore; pack vào image lúc build
```

**DoD skeleton:** `docker compose up` với proxy model 7B/8B (hoặc mock) trả streaming chat completions.

---

## 4. Timeline 10 ngày (A→Z)

### D0–D1 (20–21/07) — Unblock & Scaffold  
**Goal:** repo chạy được contract API; theo dõi portal.

| Task | Owner | Output |
|---|---|---|
| Check portal 2 lần/ngày: model, trace, F/C/w/γ, compose template | DO | Note trong `eval/portal_notes.md` |
| Dockerfile + compose skeleton (CUDA 13 / Ubuntu 24.04 target) | DO | File compose build được |
| `smoke_openai.py` (stream, TTFT/TPOT thô client-side) | AI2 | Pass với bất kỳ endpoint |
| Phân công board + credit sheet | DO | Sheet shared |

### D2 — Proxy correctness  
**Goal:** logic serving đúng dù chưa có model BTC.

| Task | Owner | Output |
|---|---|---|
| Chạy proxy Llama/Qwen 7B–8B local hoặc nhỏ trên cloud rẻ | AI1+AI2 | Chat template + stream OK |
| Liệt kê failure modes (OOM, empty, non-stream) | AI2 | Checklist |

### D3 — ERS simulator  
**Goal:** sẵn sàng khi có redacted trace.

| Task | Owner | Output |
|---|---|---|
| Parser trace (arrival, isl, osl, turn id) | AI2 | `ers_sim.py` v0 |
| Công thức \(s_{ttft}, s_{tpot}\) parametrized | AI2 | Config YAML cho F/C/w/γ |
| Profile giả định concurrency | AI2 | 1 trang note pattern |

*Nếu chưa có trace:* viết sim với synthetic multi-turn; thay file thật sau 1 giờ có data.

### D4–D5 — AMD Session A (Quant)  
**Trigger:** có model ID. Nếu chưa có model → kéo scaffold/tests, **không đốt AMD**.

| Task | Owner | Output |
|---|---|---|
| Thuê **MI300X** | DO | Instance |
| Chạy `quant_fp8.py`, ignore `lm_head` | AI1 | Thư mục FP8 |
| Verify load compressed-tensors | AI1+AI2 | Log OK |
| Upload/pack artifact vào build context | DO | Sẵn build image |

### D5–D6 — P0 ship  
| Task | Owner | Output |
|---|---|---|
| `configs/p0_safe.env` + `serve.sh` | AI2 | Prefix + chunked ON; mem 0.90; batched-tokens 4096 |
| Build image offline (no net on start) | DO | Image local |
| Nộp submission P0 lên BTC | DO | Digest + ERS #1 |
| Ghi ablation sheet dòng baseline | AI2 | Sheet |

### D7 — Online ablation (1 biến / submission)  
Thứ tự bắt buộc:

1. `max-num-batched-tokens`: 2048 → 4096 → 8192  
2. `gpu-memory-utilization`: 0.88 → 0.90 → 0.92  
3. prefix on/off (chỉ để confirm; kỳ vọng ON thắng)

| Rule | Chi tiết |
|---|---|
| Giữ winner | Chỉ giữ thay đổi nếu ERS↑ và không tăng lỗi |
| Stop | Nếu error xuất hiện → rollback ngay |

### D8 — P1 optional  
| Thí nghiệm | Điều kiện | Owner |
|---|---|---|
| `--kv-cache-dtype fp8` | P0 ổn định | AI2+AI1 |
| Speculative K=2 | Có official draft | AI2 |
| Accuracy smoke | Credit còn | AI1 |

### D9 — Gate & anti-cheat  
| Task | Owner |
|---|---|
| Chạy audit checklist mục 6 | DO |
| lm_eval smoke (nếu còn AMD/Firework) | AI1+DO |
| Chọn 1–2 digest “an toàn” | Cả team |

### D10 (29–30/07) — Freeze  
| Task | Owner |
|---|---|
| Freeze config thắng | AI2 |
| Rebuild pin versions | DO |
| Nộp cuối + backup digest | DO |
| Viết mô tả submission theo yêu cầu BTC | DO+AI2 |

---

## 5. Ablation Sheet (template)

Copy vào `eval/ablation_sheet.md`:

| ID | Date | Digest | Flags changed | ERS | Notes / errors | Keep? |
|---|---|---|---|---|---|---|
| P0 | | | prefix+chunked, bt=4096, mem=0.90 | | | Y |
| A1 | | | bt=2048 | | | |
| A2 | | | bt=8192 | | | |
| A3 | | | mem=0.92 | | | |
| B1 | | | kv=fp8 | | | |

---

## 6. Anti-Cheat & Submit Checklist

Trước mỗi lần nộp (DO tick):

- [ ] Không gọi mạng ngoài khi container chạy  
- [ ] Weights nằm trong image / volume hợp lệ theo đề  
- [ ] Một entrypoint — không nhánh “nếu GPQA thì …”  
- [ ] Streaming chat completions đúng contract  
- [ ] Không hardcode đáp án / prebake trace  
- [ ] Version pin ghi trong README submission  
- [ ] Health: `GET /v1/models` hoặc tương đương  
- [ ] Log digest + flags vào ablation sheet  

---

## 7. Decision Tree — Khi BTC công bố Model

```text
Model announced
 ├─ Dense → AI1 quant FP8; AI2 P0 flags; DO pack
 ├─ MoE   → AI1 đọc doc MoE FP8 vLLM; không quant mù; escalate nếu OOM
 └─ MLA   → AI2 confirm vLLM support; fallback SGLang chỉ nếu unblock > 48h fail
          └─ EAGLE draft official? yes → P1 K=2; no → skip spec
```

**SLA:** trong **24h** sau công bố model phải có (1) plan quant cụ thể, (2) lịch AMD Session A, (3) `max-model-len` đề xuất.

---

## 8. Commands Cheat Sheet (sẽ chỉnh khi có path model)

### Quant (AMD Session A)

```bash
# pseudocode — AI1 hoàn thiện script trong scripts/quant_fp8.py
python scripts/quant_fp8.py \
  --model "$MODEL_ID" \
  --out /artifacts/LLM-FP8 \
  --ignore-lm-head
```

### Serve P0

```bash
vllm serve /model_weights/LLM-FP8 \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096
```

### lm_eval smoke

```bash
lm_eval --model local-chat-completions \
  --tasks gpqa_diamond \
  --model_args "model=FP8,base_url=http://127.0.0.1:8000/v1/chat/completions,tokenized_requests=False,add_bos_token=True" \
  --num_fewshot 0
```

### Client smoke

```bash
python scripts/smoke_openai.py --base-url http://127.0.0.1:8000/v1 --stream
```

---

## 9. Risk Register (operational)

| Risk | Trigger | Action |
|---|---|---|
| Chưa có model đến D4 | Portal trống | Không thuê AMD; hoàn thiện sim + compose + docs nộp |
| Quant fail / OOM | Session A | Giảm sequence calibration; bỏ KV-FP8; hỏi thêm credit / MI300X |
| ERS online = 0 hàng loạt | Sau submit | Kiểm tra stream, max tokens, OOM, port, compose GPU |
| Credit AMD < 20% trước D8 | Sheet | Khoá P0; bỏ P1 |
| Δ smoke xấu | Sau quant | Rollback BF16/FP16 serve nếu image cho phép; hoặc quant nhẹ hơn |

---

## 10. Definition of Done — Phase 1

- [ ] `docker-compose.yml` nộp được theo thể lệ  
- [ ] Ít nhất **1** P0 có ERS > 0 trên leaderboard  
- [ ] Ablation sheet ≥ 3 dòng thử có kiểm soát  
- [ ] Anti-cheat checklist xanh trên digest cuối  
- [ ] Credit sheet cập nhật; còn buffer hoặc đã freeze có chủ đích  
- [ ] CONTEXT + PLAN khớp thực tế đã làm (retro ngắn D10)  

---

## 11. Ngay bây giờ (next 48h) — việc cụ thể

1. **DO:** tạo skeleton repo + `docker-compose.yml` + Dockerfile target CUDA 13 / Ubuntu 24.04.  
2. **AI2:** `smoke_openai.py` + khung `ers_sim.py`.  
3. **AI1:** khung `quant_fp8.py` + theo dõi portal model; chuẩn bị account AMD MI300X.  
4. **Cả team:** credit sheet + standup 15’.  
5. **Không** thuê AMD cho đến khi có model ID + script sẵn.

Khi BTC công bố model: nhảy thẳng **§7 + Session A (D4–D5)** — bỏ qua chờ lịch nếu script đã sẵn.
