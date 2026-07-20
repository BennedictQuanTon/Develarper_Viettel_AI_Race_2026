# 🚀 Viettel AI Race 2026 — Challenge 3: LLM Inference Optimization

[![Competition](https://img.shields.io/badge/Viettel_AI_Race-2026-red.svg)](https://viettel.vn)
[![Target Hardware](https://img.shields.io/badge/GPU-NVIDIA_H200-green.svg)](https://www.nvidia.com)
[![Engine](https://img.shields.io/badge/Serving-vLLM-blue.svg)](https://github.com/vllm-project/vllm)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An optimized, high-throughput, low-latency LLM serving solution developed for **Viettel AI Race 2026 (Challenge 3: LLM Inference Optimization)**. This repository houses our team's end-to-end inference stack designed to maximize serving efficiency under realistic, multi-turn production workloads on **NVIDIA H200 GPU** hardware.

---

## 📌 Problem Overview

The objective of Challenge 3 is to optimize real-world enterprise LLM serving under strict operational constraints. Systems are evaluated on two primary criteria:

1. **Effective Request Score (ERS)**: Evaluates serving responsiveness across a realistic multi-turn workload trace based on:
   - **TTFT (Time-To-First-Token)**: Latency to initial token generation.
   - **TPOT (Time-Per-Output-Token)**: Inter-token latency during streaming output.
2. **Accuracy Gate ($f(\Delta)$)**: Post-competition quality evaluation using the **GPQA Diamond** benchmark against the FP16/BF16 baseline model to ensure quality is preserved ($\Delta \le 0.1$).

---

## 🛠 Target Environment

- **Hardware**: NVIDIA H200 GPU (141GB HBM3e, 4.8 TB/s memory bandwidth)
- **OS**: Ubuntu 24.04 LTS
- **Driver / CUDA**: NVIDIA Driver 590.x / CUDA 13.x
- **Interface**: OpenAI-Compatible REST API

---

## 💡 High-Level Optimization Strategy

To achieve an optimal balance between throughput, latency, and accuracy, our architecture focuses on four main optimization pillars:

```
+-------------------------------------------------------------------+
|                        Inference Stack                            |
+-------------------------------------------------------------------+
|  1. Quantization & Precision  | W8A8 FP8 execution & static scale |
|  2. KV Cache & Memory         | PagedAttention & Prefix Caching  |
|  3. Serving & Scheduling      | Continuous Batching & Chunked     |
|  4. Accelerated Generation    | Speculative Decoding Integration  |
+-------------------------------------------------------------------+
```

1. **Precision & Quantization Efficiency**
   - Leveraging hardware-native 8-bit floating point (**FP8**) support on Hopper Tensor Cores for memory-bandwidth bound decode acceleration while preserving accuracy.

2. **Advanced Memory & KV Cache Management**
   - Paged KV Cache allocation to eliminate internal fragmentation and maximize batch capacity.
   - Intelligent prefix caching to drastically reduce prefill overhead for multi-turn conversational traces.

3. **Latency-Aware Request Scheduling**
   - Iteration-level continuous batching combined with chunked prefill to prevent Head-of-Line blocking and stabilize streaming TPOT.

4. **Speculative Execution & Serving Stack**
   - Production-ready serving infrastructure built on **vLLM**, integrated with draft-based speculative decoding mechanisms for higher generation throughput.

---

## 📦 Containerization & Deployment

The solution is packaged as a fully self-contained Docker image conforming to strict offline serving rules:

```bash
# Pull and run the serving container
docker run --gpus all \
  -p 8000:8000 \
  --ipc=host \
  develarper-viettel-ai-race-2026:latest
```

The container exposes an OpenAI-compatible API endpoint:
- `POST /v1/chat/completions`

---

## 🧪 Evaluation & Quality Verification

Local verification against the accuracy gate is performed using `lm-evaluation-harness`:

```bash
lm_eval --model local-chat-completions \
  --tasks gpqa_diamond \
  --model_args "model=/path/to/model,base_url=http://localhost:8000/v1/chat/completions,add_bos_token=True" \
  --num_fewshot 0
```

---

## 👥 Team & Acknowledgments

Developed by **Develarper Team** for **Viettel AI Race 2026**.
