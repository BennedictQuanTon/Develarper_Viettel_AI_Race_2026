# Ablation — LFM2.5-1.2B on MiG H200 18GB (change ONE flag per submit)

| ID | Date | Hub digest/tag | Flags changed | ERS | Errors / notes | Keep? |
|---|---|---|---|---|---|---|
| P0 | | | prefix ON, mem=0.95, max-model-len=32768 (BTC-like) | | baseline | |
| A1 | | | gpu-memory-utilization=0.90 | | | |
| A2 | | | max-num-batched-tokens=2048 | | | |
| A3 | | | max-num-batched-tokens=8192 | | | |
| A4 | | | enable-chunked-prefill | | | |
| B1 | | | quantization=fp8 (online) | | check Δ | |
| B2 | | | kv-cache-dtype=fp8 | | | |
