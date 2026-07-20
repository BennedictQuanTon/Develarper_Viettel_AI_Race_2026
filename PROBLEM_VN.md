# [VIETTEL AI RACE 2026] Thông báo về việc cập nhật Đề 3

**Thân gửi các đội thi,**

Để đảm bảo tính công bằng cho kỳ thi BTC có sự cập nhật mới trong đề bài 3 LLM Inference Optimization.

Submission của các đội đã được chấm lại theo update này. Các đội vui lòng cập nhật phần thông tin mô tả và file trace để cập nhật những thay đổi này.

**Trân trọng,**  
*Ban Tổ chức Viettel AI Race 2026*

---

Bài toán mô phỏng trực tiếp thách thức mà các đội ngũ hạ tầng AI doanh nghiệp đang đối mặt: phục vụ mô hình ngôn ngữ lớn sao cho đáp ứng đồng thời thông lượng cao, độ trễ thấp, độ chính xác ổn định và hiệu quả trên tài nguyên GPU hữu hạn.

Thí sinh sẽ deploy và tối ưu serving stack cho mô hình LLM (do Ban tổ chức chỉ định) trên hạ tầng NVIDIA H200, với workload được mô phỏng theo production trace của một hệ thống LLM serving quy mô thực tế. Bài toán cho phép tự do lựa chọn phương pháp tối ưu - từ quantization, KV cache management, prefix caching, đến custom CUDA kernel và scheduling - để tối đa hoá tỷ lệ request được đáp ứng đúng yêu cầu trong khi vẫn đảm bảo được chất lượng đầu ra.

---

## 1. Giới thiệu

Sự bùng nổ của các mô hình ngôn ngữ lớn (Large Language Models - LLM) trong những năm gần đây đang tạo ra áp lực rất lớn lên hạ tầng tính toán của các tổ chức và doanh nghiệp. Một hệ thống inference trong môi trường sản xuất thực tế không chỉ cần đạt thông lượng (throughput) cao, mà còn phải đảm bảo đồng thời ba yêu cầu cốt lõi:

- **Độ trễ thấp (Latency)**
- **Độ chính xác ổn định (Accuracy)**
- **Khả năng vận hành hiệu quả trên tài nguyên GPU hữu hạn.**

Cuộc thi **LLM Inference Optimization Challenge** mô phỏng trực tiếp bài toán thực tiễn mà các đội ngũ hạ tầng AI đang đối mặt: *"Làm thế nào để phục vụ một mô hình ngôn ngữ lớn (LLM) với mức hiệu năng cao nhất, trong khi vẫn đảm bảo độ chính xác của câu trả lời?"*

---

## 2. Mục tiêu bài toán

Đây là bài toán tối ưu hóa phục vụ LLM (LLM serving optimization) có ràng buộc về chất lượng. Mục tiêu chung của cuộc thi là: Tối đa hóa Effective Request Score (ERS) trên toàn bộ workload trace cố định do Ban tổ chức (BTC) phát hành, đồng thời vượt qua bài kiểm tra chất lượng (Accuracy Gate). Khả năng đáp ứng của hệ thống được đánh giá qua từng request dựa trên cơ chế chấm điểm độ trễ liên tục.

- **Vòng online (theo từng submission)**: Chỉ chấm ERS (độ trễ / phục vụ). Không chạy Accuracy Gate ngay trên mỗi lượt nộp.
- **Sau khi kết thúc vòng online**: Đội chọn thủ công tối đa 5 submissions tốt nhất; BTC hậu kiểm tính hợp lệ rồi mới chấm GPQA Diamond full để tính $f(\Delta)$ và điểm tổng.

- **Latency Bounds**: Điểm ERS được nội suy liên tục dựa trên cận dưới (Floor) và cận trên (Ceiling) của độ trễ.
- **Accuracy Gate**: Độ chính xác không được suy giảm quá ngưỡng quy định so với baseline BF16.

### Các khái niệm chính:
- **TTFT (Time-To-First-Token)**: Thời gian từ khi gửi request đến khi nhận token đầu tiên.
- **TPOT (Time-Per-Output-Token)**: Thời gian giữa 2 token liên tiếp trong stream output.

---

## 3. Workload Trace & Cách tính điểm

### 3.1 Nguồn dữ liệu
BTC sử dụng bộ dữ liệu mô phỏng luồng request thực tế trong môi trường LLM serving quy mô lớn, chọn lọc để đại diện cho các pattern traffic phổ biến. Cấu trúc trace:

- **Multi-turn**: mỗi hội thoại gồm nhiều lượt; lượt kế tiếp chỉ gửi sau khi lượt trước hoàn tất kèm khoảng "think" mô phỏng thời gian người dùng, giữ nguyên tính nhân-quả của hội thoại thật.
- **Giới hạn độ dài**: mỗi prompt bị giới hạn độ dài context input và số token output, phản ánh tải prefill/decode thực tế trên slice được cấp.
- **Bản công khai vs bản chấm**: thí sinh nhận bản trace đã lược text (chỉ arrival + số token in/out mỗi lượt); BTC giữ bản đầy đủ và chỉ gửi prompt thật tới endpoint lúc chấm - chống pre-bake/học tủ theo nội dung.

### 3.2 ERS (Effective Request Score)
ERS là chỉ số đánh giá hiệu năng xử lý request thông qua cơ chế chấm điểm liên tục, tối ưu đồng thời TTFT và TPOT. Điểm ERS của hệ thống là trung bình cộng điểm của tất cả $N$ requests trong file trace:

$$\text{ERS} = \frac{1}{N} \sum_{i=1}^N S_{\text{request}, i} \in [0, 1]$$

Điểm của từng request ($S_{\text{request}}$) được tính như sau:

$$S_{\text{request}} = \begin{cases} 0 & \text{nếu lỗi, timeout, hoặc trả về 0 token} \\ w \cdot s_{\text{ttft}} + (1 - w) \cdot s_{\text{tpot}} & \text{nếu xử lý thành công} \end{cases}$$

Trong đó, điểm thành phần độ trễ $s_{\text{ttft}}$ và $s_{\text{tpot}}$ được nội suy giữa ngưỡng lý tưởng (Floor - $F$) và ngưỡng giới hạn (Ceiling - $C$):

$$s_{\text{ttft}} = \left[ \text{clamp}\left( \frac{C_{\text{ttft}} - \text{TTFT}}{C_{\text{ttft}} - F_{\text{ttft}}}, 0, 1 \right) \right]^\gamma$$

$$s_{\text{tpot}} = \left[ \text{clamp}\left( \frac{C_{\text{tpot}} - \text{TPOT}}{C_{\text{tpot}} - F_{\text{tpot}}}, 0, 1 \right) \right]^\gamma$$

**Giải thích các tham số cấu hình:**
- $F_{\text{ttft}}, F_{\text{tpot}}$: Cận dưới (Floor) - độ trễ đạt mức này hoặc thấp hơn sẽ nhận điểm tối đa ($s = 1$).
- $C_{\text{ttft}}, C_{\text{tpot}}$: Cận trên (Ceiling) - độ trễ chạm mức này hoặc cao hơn sẽ bị tính $0$ điểm ($s = 0$).
- $w$: Trọng số ưu tiên của TTFT ($0 < w < 1$).
- $\gamma$: Hệ số lũy thừa ($\gamma \ge 1$) quy định độ dốc của hàm phạt (penalty curve).
- $\text{clamp}(x, 0, 1)$: Giới hạn giá trị của $x$ luôn nằm trong đoạn $[0, 1]$.

### 3.3 Accuracy Gate
Không chấm accuracy theo từng lượt nộp trong vòng online. Leaderboard online phản ánh chủ yếu ERS (điểm độ trễ) và mang tính tham khảo cho đến khi Accuracy Gate hoàn tất.

**Quy trình sau khi kết thúc vòng online:**
1. **Đội chọn submissions**: Mỗi đội chọn thủ công tối đa 5 bài submissions tốt nhất (image/digest đã nộp trong vòng online; không được đổi image sau khi chọn).
2. **Hậu kiểm tính hợp lệ (BTC)**: BTC kiểm tra phương án có tuân thủ Rule & Anti-Cheating / tinh thần production hay không (image pin, hành vi serving, dấu hiệu gian lận, v.v.). Bài không hợp lệ bị loại khỏi vòng accuracy / có thể void.
3. **Chấm GPQA Diamond full**: Với mỗi submission còn hợp lệ, BTC dựng lại endpoint OpenAI-compatible và chạy `lm-evaluation-harness` (`lm_eval`) trên bộ GPQA do BTC công bố (baseline reference BF16; filter strict-match).

Hàm suy giảm độ chính xác $\Delta$ (accuracy drop) được tính như sau:

$$\Delta = \text{Accuracy}_{\text{baseline}} - \text{Accuracy}_{\text{submission}}$$

*(Trong đó, $\text{Accuracy}_{\text{baseline}}$ là accuracy tham chiếu của mô hình gốc chạy bằng trọng số BF16 do BTC công bố; $\text{Accuracy}_{\text{submission}}$ là accuracy bài nộp của đội.)*

Dựa trên $\Delta$, hệ thống áp dụng hàm phạt $f(\Delta)$ - piecewise linear, đầu ra thuộc $[0, 1]$:

$$f(\Delta) = \begin{cases} 1.0 & \text{nếu } \Delta \le 0.10 \\ 1.0 - \frac{\Delta - 0.10}{0.06} & \text{nếu } 0.10 < \Delta < 0.16 \\ 0.0 & \text{nếu } \Delta \ge 0.16 \end{cases}$$

Với mỗi submission được chọn: $\text{Score}_i = 100 \times \text{ERS}_i \times f(\Delta_i)$. Điểm chính thức của đội là Score tốt nhất trong các bài còn hợp lệ sau hậu kiểm + GPQA (trừ khi BTC công bố quy tắc gộp khác).

### 3.4. Công thức tính điểm tổng
Điểm số cuối cùng kết hợp hiệu năng phục vụ (ERS từ vòng online) với hình phạt sụt giảm chất lượng sau Accuracy Gate:

$$\text{Score} = 100 \times \text{ERS} \times f(\Delta)$$

Score trên chỉ được chốt sau khi đội đã chọn tối đa 5 submissions và BTC hoàn tất hậu kiểm + GPQA Diamond full.

---

## 4. Mô hình sử dụng

Mô hình cụ thể do BTC chỉ định và công bố theo từng vòng.

---

## 5. Phương pháp tối ưu được phép

Thí sinh được toàn quyền tự do lựa chọn và kết hợp các phương pháp tối ưu, miễn không vi phạm quy định của cuộc thi. Các hướng tiếp cận được khuyến khích bao gồm:

- **KV Cache Optimization**: KV cache quantization (FP8, INT8), KV cache offloading (CPU/NVMe), prefix caching, semantic caching, Paged Attention, memory-aware scheduling.
- **Serving & Scheduling Optimization**: Dynamic/continuous batching, speculative decoding, disaggregated prefill/decode serving.
- **System-Level Optimization**: Custom CUDA / Triton kernels, fused attention kernels (FlashAttention, FlashInfer...), NCCL communication optimization, CUDA Graphs, memory layout optimization.
- **Runtime & Compiler Optimization**: Sử dụng vLLM.

---

## 6. Môi trường đánh giá chuẩn hóa

- **Hạ tầng phần cứng**: NVIDIA H200 GPU
- **Hệ điều hành**: Ubuntu 24.04 LTS
- **GPU Driver**: NVIDIA driver 590.x (hỗ trợ CUDA 13.x)

---

## 7. Rule & Anti-Cheating

**Nguyên tắc cốt lõi**: Giải pháp phải tối ưu hệ thống phục vụ trung thực, sẵn sàng triển khai cho người dùng thực tế. Mọi thủ thuật đánh lừa hệ thống đo lường hoặc chỉ hoạt động trên tập workload chấm thi đều bị xem là vi phạm nghiêm trọng.

### 7.1. Không gian tối ưu

**❌ Hành vi nghiêm cấm:**
- **Pre-bake / Hardcode**: Tính sẵn đáp án thay vì suy luận thực tại thời điểm phục vụ.
- **Dual-path**: Rẽ nhánh hành vi xử lý giữa lúc đo độ trễ và lúc kiểm tra chất lượng.
- **Gaming metrics**: Đệm rỗng, cắt ngắn chuỗi sinh trái phép để né cổng hậu kiểm.
- **Can thiệp hạ tầng**: Gọi mạng ngoài, sửa tokenizer/weights, làm bẩn tài nguyên.
- **Bất trung thực quy trình**: Tráo image sau khi nộp, lộ dữ liệu.

### 7.2. Quy trình Hậu kiểm
- Điểm số tự động trên hệ thống chưa phải là kết quả chốt cuối cùng.
- Ban tổ chức định kỳ hoặc đột xuất rà soát thủ công image, cấu hình, log và luồng serving.
- Bài nộp phát hiện gian lận sẽ bị hủy (void) kết quả hoặc điều chỉnh xếp hạng trực tiếp.
- Mọi quyết định xử lý đều được thông báo minh bạch qua email kèm lý do tóm tắt.

### 7.3. Phân định vùng sát điểm
Đối với các đội nằm trong vùng nhiễu đo lường ($\le 1–3$ điểm), thứ hạng sẽ được phân định tuần tự theo:
1. Mức độ suy giảm độ chính xác thấp hơn.
2. Chỉ số độ trễ p95 TTFT thấp hơn.
3. Tốc độ sinh văn bản cao hơn.
4. Thời điểm nộp bài hợp lệ sớm hơn.

### 7.4. Re-grade & Chế tài xử lý
- **Chấm lại**: Ban tổ chức có quyền chạy độc lập nhiều lần trên đúng bản Docker image đã chốt để lấy điểm trung vị. Các đội trong top đầu sẽ được ưu tiên rà soát.
- **Xử lý vi phạm**: Tùy mức độ, cá nhân hoặc đội thi có thể bị thu hồi điểm hoặc loại hoàn toàn khỏi giải đấu.
- **Quyền khiếu nại**: Hệ thống tiếp nhận phản hồi trong vòng 24 giờ kể từ thời điểm nhận email thông báo hoặc công bố kết quả hạng mục thi.
