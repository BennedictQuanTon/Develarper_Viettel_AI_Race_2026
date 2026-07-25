# GOLD status — sau p5-sf8 = 50.46 (REJECT)

| | Yoshio #10 | p5-sf8 (vừa nộp) |
|---|---|---|
| ERS | **61.66** | **50.46** (−11.2) |
| TBT | **4 ms** | **6 ms** (mất win FP8) |
| TTFT p50/p95 | 48 / 70 | 49 / 72 |
| Fail | 5 | 6 |

**Kết luận:** Static FP8 offline **fail** trên LFM2.5 hybrid.  
**Compose local đã revert** về pure `#10` (`:p2-fi` + `--quantization=fp8`).

**Cấm tiếp:** p5-sf8 · OMP env · bt=256 · speculative  

**Vàng giữ:** Yoshio #10 — chưa có hướng Portal-proven >61.66.
