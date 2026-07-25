# Hướng dẫn — Chạy test local

```
Tier 1 · Unit tests (CPU)     → python -m pytest develarper_opt/tests -q
Tier 2 · Kernel tests (GPU)   → python -m pytest develarper_opt/tests -q -m cuda
Tier 3 · E2E serve (Docker)     → bash scripts/opt_smoke.sh submit/docker-compose.yml
```

## Tier 3 — Build & smoke

```bash
cd Develarper_Viettel_AI_Race_2026
docker build --platform linux/amd64 -f Dockerfile.tuong \
  -t nakituonghuynh/develarper-lfm25:tuong-opt-v1 .

bash scripts/opt_smoke.sh submit/docker-compose.yml
```

Pivot test:

```bash
bash scripts/opt_smoke.sh submit/docker-compose.tuong_cudagraphs_pivot.yml
```

See full unit test tables in repo history or run `pytest -v` per file under `develarper_opt/tests/`.
