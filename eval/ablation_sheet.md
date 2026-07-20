# Ablation sheet — change ONE flag per online submission

| ID | Date | Digest | Flags changed | ERS | Errors / notes | Keep? |
|---|---|---|---|---|---|---|
| P0 | | | prefix+chunked, bt=4096, mem=0.90 | | baseline | |
| A1 | | | max-num-batched-tokens=2048 | | | |
| A2 | | | max-num-batched-tokens=8192 | | | |
| A3 | | | gpu-memory-utilization=0.88 | | | |
| A4 | | | gpu-memory-utilization=0.92 | | | |
| B1 | | | kv-cache-dtype=fp8 | | | |
| B2 | | | speculative K=2 | | only if draft exists | |
