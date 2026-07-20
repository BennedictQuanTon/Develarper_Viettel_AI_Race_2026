# Ablation — LFM2.5-1.2B on MiG H200 18GB (ONE flag change per Portal submit)

| ID | Date | Hub tag/digest | Compose file | Flags | ERS | Errors | Keep? |
|---|---|---|---|---|---|---|---|
| P0 | | | docker-compose.submit.yml | prefix ON, mem=0.95, max-len=32768 | | | |
| A1 | | | submit/docker-compose.a1_mem90.yml | mem=0.90 | | | |
| A2 | | | submit/docker-compose.a2_chunked.yml | chunked + batched-tokens=4096 | | | |
| B1 | | | submit/docker-compose.b1_fp8.yml | quantization=fp8 | | check Δ | |
| B2 | | | submit/docker-compose.b2_kvfp8.yml | kv-cache-dtype=fp8 | | | |
