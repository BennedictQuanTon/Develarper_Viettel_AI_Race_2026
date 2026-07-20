# Target eval env: Ubuntu 24.04 + NVIDIA Driver 590.x / CUDA 13.x / H200
# Pin a concrete vLLM tag once BTC model + CUDA compatibility are confirmed.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.8.5
FROM ${VLLM_IMAGE}

WORKDIR /app

# Serving scripts + default P0 config (weights copied when artifact exists)
COPY scripts/serve.sh /app/scripts/serve.sh
COPY configs/p0_safe.env /app/configs/p0_safe.env
COPY configs/p1_aggressive.env /app/configs/p1_aggressive.env

RUN chmod +x /app/scripts/serve.sh

# When FP8 artifact is ready (AMD Session A), uncomment:
# COPY model_weights/LLM-FP8 /model_weights/LLM-FP8

ENV CONFIG_FILE=/app/configs/p0_safe.env
ENV MODEL_PATH=/model_weights/LLM-FP8
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Competition rule: no network fetch of weights at start — bake or mount offline.
ENTRYPOINT ["/bin/bash", "/app/scripts/serve.sh"]
