# Build-time may use network. Runtime on BTC MiG must be offline.
# BTC baseline: vllm/vllm-openai:v0.22.1
# If LFM2.5 does not load, rebuild with e.g.:
#   docker build --build-arg VLLM_IMAGE=vllm/vllm-openai:v0.23.0 -t ...

ARG VLLM_IMAGE=vllm/vllm-openai:v0.22.1
FROM ${VLLM_IMAGE}

ARG MODEL_ID=LiquidAI/LFM2.5-1.2B-Instruct
ENV MODEL_ID=${MODEL_ID}

# Bake weights into /model (requires network during `docker build` only).
RUN python3 -c "\
from huggingface_hub import snapshot_download;\
import os;\
mid=os.environ['MODEL_ID'];\
snapshot_download(repo_id=mid, local_dir='/model', local_dir_use_symlinks=False);\
print('OK', mid, '-> /model')"

EXPOSE 8000
