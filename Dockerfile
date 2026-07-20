# Competition image for MiG H200 (CUDA 13 / driver 590.x).
#
# IMPORTANT: Official LFM2.5 recipes require vLLM >= 0.23.0.
# BTC sample compose uses v0.22.1 — that tag may NOT load Lfm2ForCausalLM.
# Default here: CUDA 13 image. Override if needed:
#   docker build --build-arg VLLM_IMAGE=vllm/vllm-openai:v0.23.0 -t ...
#
# Build (from repo root, after downloading weights):
#   make download-model
#   make build
#
# Runtime must be offline (weights already in /model).

ARG VLLM_IMAGE=vllm/vllm-openai:latest-cu130
FROM ${VLLM_IMAGE}

# Expect local bake path produced by: scripts/download_model.sh
COPY model_weights/LFM2.5-1.2B-Instruct /model

# Sanity: fail build early if config missing
RUN test -f /model/config.json && ls -lah /model | head

EXPOSE 8000
