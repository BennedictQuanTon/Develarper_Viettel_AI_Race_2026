#!/usr/bin/env bash
# A/B comparison: Yoshio SOTA vs opt-platform image.
# Runs opt_bench against each compose in sequence, tearing down between runs.
set -euo pipefail

A_COMPOSE="${A_COMPOSE:-submit_yoshio/docker_compose_fp8_flash.yaml}"
B_COMPOSE="${B_COMPOSE:-submit_tuong/docker_compose_fp8_flash_opt.yaml}"

echo "═════════════════════════════════════════════════════════"
echo " A (baseline)  : ${A_COMPOSE}"
echo " B (patched)   : ${B_COMPOSE}"
echo "═════════════════════════════════════════════════════════"

echo; echo "── Running A ────────────────────────────────────────────"
bash scripts/opt_smoke.sh "${A_COMPOSE}" | tee /tmp/ab_A.log

echo; echo "── Running B ────────────────────────────────────────────"
bash scripts/opt_smoke.sh "${B_COMPOSE}" | tee /tmp/ab_B.log

echo; echo "── Diff (grep summary lines) ────────────────────────────"
grep -E "TTFT|TPOT|approx ERS|requests" /tmp/ab_A.log || true
echo "---"
grep -E "TTFT|TPOT|approx ERS|requests" /tmp/ab_B.log || true
