# Portal notes — check 2×/day until model + trace drop

| Date | Model ID? | Trace file? | F/C/w/γ? | Compose template? | Notes |
|---|---|---|---|---|---|
| 2026-07-20 | no | no | no | no | Scaffold only; do not rent AMD yet |
| | | | | | |

## Unblock checklist when model appears

- [ ] Record exact model id / revision / license path
- [ ] Download / mount instructions from BTC
- [ ] Update `MODEL_PATH` and `MAX_MODEL_LEN` in `configs/*.env`
- [ ] Schedule AMD MI300X Session A with `scripts/quant_fp8.py` ready
- [ ] Drop public redacted trace under `eval/traces/` and run `ers_sim.py`
