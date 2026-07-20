# [VIETTEL AI RACE 2026] Notice Regarding Update to Challenge 3

**Dear Teams,**

To ensure fairness in the competition, the Organizing Committee has released an update to **Challenge 3: LLM Inference Optimization**.

Submissions have been re-scored according to this update. Teams are kindly requested to update their description information and trace files to reflect these changes.

**Best regards,**  
*Organizing Committee of Viettel AI Race 2026*

---

The problem directly simulates the challenge faced by enterprise AI infrastructure teams: serving Large Language Models to simultaneously meet high throughput, low latency, stable accuracy, and efficiency on limited GPU resources.

Contestants will deploy and optimize the serving stack for a Large Language Model (specified by the Organizing Committee) on **NVIDIA H200** infrastructure, with workloads simulated according to the production trace of a real-world enterprise LLM serving system. The challenge allows complete freedom in choosing optimization methods—from quantization, KV cache management, prefix caching, to custom CUDA kernels and scheduling—to maximize the proportion of requests served effectively while ensuring output quality.

---

## 1. Introduction

The boom of Large Language Models (LLMs) in recent years is creating immense pressure on the computational infrastructure of organizations and enterprises. A production-grade inference system must not only achieve high throughput, but also simultaneously fulfill three core requirements:

- **Low Latency**
- **Stable Accuracy**
- **Efficient operation on limited GPU resources**

The **LLM Inference Optimization Challenge** directly simulates the practical problem facing AI infrastructure teams: *"How to serve a Large Language Model (LLM) with peak performance while ensuring response accuracy?"*

---

## 2. Problem Objectives

This is a quality-constrained LLM serving optimization problem. The overall objective of the competition is to **maximize the Effective Request Score (ERS)** over the entire fixed workload trace released by the Organizing Committee, while passing the quality check (**Accuracy Gate**). System responsiveness is evaluated request-by-request based on a continuous latency scoring mechanism.

- **Online Round (per submission)**: Scores ERS (latency / serving) only. Accuracy Gate is **NOT** run on every submission during the online round.
- **After the online round ends**: Teams manually select up to 5 best submissions; the committee verifies validity before running the full GPQA Diamond evaluation to compute $f(\Delta)$ and the final overall score.

- **Latency Bounds**: ERS score is continuously interpolated based on lower bounds (Floor) and upper bounds (Ceiling) of latency.
- **Accuracy Gate**: Accuracy must not degrade beyond specified thresholds compared to the BF16 baseline.

### Key Concepts:
- **TTFT (Time-To-First-Token)**: Time elapsed from request submission until receiving the first token.
- **TPOT (Time-Per-Output-Token)**: Time between two consecutive tokens in the streaming output.

---

## 3. Workload Trace & Scoring Methodology

### 3.1 Data Source
The Organizing Committee uses a dataset simulating real-world request traffic in large-scale LLM serving environments, selected to represent common traffic patterns. Trace structure:

- **Multi-turn**: Each conversation consists of multiple turns; subsequent turns are sent only after previous turns complete, with a "think time" interval simulating user behavior, maintaining true conversational causality.
- **Length limits**: Each prompt is constrained by context input length and output token count, reflecting realistic prefill/decode loads on the assigned GPU slice.
- **Public vs Evaluation trace**: Contestants receive a redacted text trace (containing only arrival timestamp + input/output token counts per turn); the Organizing Committee retains the full text trace and sends real prompts to endpoints during scoring—preventing pre-baking or memorization of contents.

### 3.2 ERS (Effective Request Score)
ERS is a request processing performance metric using a continuous scoring mechanism, simultaneously optimizing TTFT and TPOT. The system's ERS score is the arithmetic mean of all $N$ requests in the trace file:

$$\text{ERS} = \frac{1}{N} \sum_{i=1}^N S_{\text{request}, i} \in [0, 1]$$

The score for each request ($S_{\text{request}}$) is calculated as:

$$S_{\text{request}} = \begin{cases} 0 & \text{if error, timeout, or returns 0 tokens} \\ w \cdot s_{\text{ttft}} + (1 - w) \cdot s_{\text{tpot}} & \text{if processed successfully} \end{cases}$$

Where the latency component scores $s_{\text{ttft}}$ and $s_{\text{tpot}}$ are interpolated between the ideal threshold (Floor - $F$) and upper limit threshold (Ceiling - $C$):

$$s_{\text{ttft}} = \left[ \text{clamp}\left( \frac{C_{\text{ttft}} - \text{TTFT}}{C_{\text{ttft}} - F_{\text{ttft}}}, 0, 1 \right) \right]^\gamma$$

$$s_{\text{tpot}} = \left[ \text{clamp}\left( \frac{C_{\text{tpot}} - \text{TPOT}}{C_{\text{tpot}} - F_{\text{tpot}}}, 0, 1 \right) \right]^\gamma$$

**Configuration Parameters Explanation:**
- $F_{\text{ttft}}, F_{\text{tpot}}$: Lower bound (Floor)—latencies at or below this level receive maximum score ($s = 1$).
- $C_{\text{ttft}}, C_{\text{tpot}}$: Upper bound (Ceiling)—latencies at or above this level receive $0$ points ($s = 0$).
- $w$: Weight factor favoring TTFT ($0 < w < 1$).
- $\gamma$: Power exponent ($\gamma \ge 1$) defining the steepness of the penalty curve.
- $\text{clamp}(x, 0, 1)$: Constrains the value of $x$ to the range $[0, 1]$.

### 3.3 Accuracy Gate
Accuracy is not evaluated per submission during the online round. The online leaderboard mainly reflects ERS (latency score) and serves as a reference until the Accuracy Gate process completes.

**Process after the online round ends:**
1. **Submission Selection**: Each team manually selects up to 5 best submissions (images/digests submitted during the online round; images cannot be changed after selection).
2. **Validity Audit (Organizing Committee)**: The committee checks whether solutions comply with Rules & Anti-Cheating / production guidelines (image pinning, serving behavior, cheating flags, etc.). Invalid submissions are excluded from accuracy evaluation or voided.
3. **Full GPQA Diamond Benchmark**: For each remaining valid submission, the committee reconstructs the OpenAI-compatible endpoint and executes `lm-evaluation-harness` (`lm_eval`) on the published GPQA dataset (reference baseline BF16; strict-match filter).

The accuracy degradation $\Delta$ (accuracy drop) is calculated as:

$$\Delta = \text{Accuracy}_{\text{baseline}} - \text{Accuracy}_{\text{submission}}$$

*(Where $\text{Accuracy}_{\text{baseline}}$ is the reference accuracy of the baseline model running with BF16 weights as published by the Organizing Committee; $\text{Accuracy}_{\text{submission}}$ is the accuracy of the team's submission.)*

Based on $\Delta$, the system applies a piecewise linear penalty function $f(\Delta)$, with output in $[0, 1]$:

$$f(\Delta) = \begin{cases} 1.0 & \text{if } \Delta \le 0.10 \\ 1.0 - \frac{\Delta - 0.10}{0.06} & \text{if } 0.10 < \Delta < 0.16 \\ 0.0 & \text{if } \Delta \ge 0.16 \end{cases}$$

For each selected submission: $\text{Score}_i = 100 \times \text{ERS}_i \times f(\Delta_i)$. The official score of the team is the best score among remaining valid submissions after audit + GPQA (unless other aggregation rules are announced by the committee).

### 3.4 Overall Score Formula
The final score combines serving performance (ERS from the online round) with the accuracy degradation penalty after the Accuracy Gate:

$$\text{Score} = 100 \times \text{ERS} \times f(\Delta)$$

The above Score is finalized only after the team selects up to 5 submissions and the committee completes post-audit verification + full GPQA Diamond benchmarking.

---

## 4. Models Used

Specific models are designated and announced by the Organizing Committee per competition round.

---

## 5. Permitted Optimization Methods

Contestants are fully free to choose and combine optimization techniques, provided they do not violate competition rules. Encouraged approaches include:

- **KV Cache Optimization**: KV cache quantization (FP8, INT8), KV cache offloading (CPU/NVMe), prefix caching, semantic caching, Paged Attention, memory-aware scheduling.
- **Serving & Scheduling Optimization**: Dynamic/continuous batching, speculative decoding, disaggregated prefill/decode serving.
- **System-Level Optimization**: Custom CUDA / Triton kernels, fused attention kernels (FlashAttention, FlashInfer...), NCCL communication optimization, CUDA Graphs, memory layout optimization.
- **Runtime & Compiler Optimization**: Using vLLM.

---

## 6. Standardized Evaluation Environment

- **Hardware Infrastructure**: NVIDIA H200 GPU
- **Operating System**: Ubuntu 24.04 LTS
- **GPU Driver**: NVIDIA driver 590.x (supporting CUDA 13.x)

---

## 7. Rules & Anti-Cheating Policy

**Core Principle**: Solutions must honestly optimize the serving system for real-world deployment. Any trick aimed at fooling the metric system or operating exclusively on the contest workload trace will be treated as a severe violation.

### 7.1 Optimization Constraints

**❌ Strictly Prohibited Actions:**
- **Pre-bake / Hardcode**: Pre-computing answers instead of real-time inference at serving time.
- **Dual-path**: Branching processing behavior between latency measurement phase and accuracy testing phase.
- **Gaming metrics**: Dummy padding, unauthorized truncation of generated output to bypass quality checks.
- **Infrastructure tampering**: External network calls, modifying tokenizers/weights unexpectedly, corrupting system resources.
- **Procedural dishonesty**: Swapping Docker images after submission, data leakage.

### 7.2 Audit & Post-Verification Process
- Automated scores on the leaderboard are not final.
- The Organizing Committee will perform periodic or unannounced manual audits of Docker images, configurations, logs, and serving flows.
- Submissions flagged for cheating will have results voided or rankings adjusted directly.
- All enforcement decisions will be communicated transparently via email with summary justifications.

### 7.3 Tie-Breaking for Close Scores
For teams within measurement noise margin ($\le 1–3$ points), rankings will be determined sequentially by:
1. Lower accuracy degradation ($\Delta$).
2. Lower p95 TTFT latency.
3. Higher text generation speed.
4. Earlier valid submission timestamp.

### 7.4 Re-grading & Sanctions
- **Re-grading**: The committee reserves the right to run independent evaluations multiple times on the exact pinned Docker image to derive median scores. Top teams will be prioritized for review.
- **Sanctions**: Depending on severity, individuals or teams may face score forfeiture or complete disqualification from the competition.
- **Appeals**: The system accepts feedback within 24 hours from receipt of notification emails or ranking announcements.
