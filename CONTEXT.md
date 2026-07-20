# Báo cáo Nghiên cứu Kỹ thuật Chuyên sâu: Chiến lược Kiến trúc và Tối ưu hóa LLM Inference trên NVIDIA H200

## 1. Phân tích Bài toán và Giới hạn Không gian Thiết kế

Sự phát triển bùng nổ của các mô hình ngôn ngữ lớn (LLM) đã chuyển dịch trọng tâm của ngành công nghiệp trí tuệ nhân tạo từ việc huấn luyện (training) sang bài toán phục vụ suy luận (inference) ở quy mô sản xuất. Cuộc thi Viettel AI Race 2026 (Đề 3: LLM Inference Optimization) tái hiện chính xác những thách thức khốc liệt nhất tại các trung tâm dữ liệu hiện đại: làm thế nào để tối đa hóa thông lượng xử lý (throughput) và giảm thiểu độ trễ (latency), trong khi phải duy trì độ chính xác (accuracy) của mô hình trên một nguồn tài nguyên phần cứng hữu hạn.

Bài toán đặt ra ba ràng buộc cốt lõi mang tính tương hỗ nhưng cũng thường xuyên mâu thuẫn lẫn nhau:
1. **Effective Request Score (ERS)**: Đo lường hiệu năng của hệ thống thông qua việc nội suy liên tục hai chỉ số độ trễ: Time-To-First-Token (TTFT) và Time-Per-Output-Token (TPOT).
2. **Cổng Chất lượng (Accuracy Gate)**: Đóng vai trò là cơ chế hậu kiểm nghiêm ngặt, sử dụng bộ dữ liệu GPQA Diamond để đối chiếu độ chính xác của hệ thống tối ưu so me với mô hình gốc (baseline) chạy trên định dạng BF16.
3. **Luồng dữ liệu đánh giá (Workload Trace)**: Được mô phỏng theo môi trường sản xuất thực tế với các chuỗi hội thoại đa lượt (multi-turn), có chứa thời gian "suy nghĩ" (think time) của người dùng, giới hạn ngữ cảnh, và đặc biệt là không thể sử dụng các thủ thuật gian lận như hardcode hay tính toán trước (pre-bake).

Hạ tầng đích của cuộc thi là máy chủ trang bị GPU NVIDIA H200 hoạt động trên hệ điều hành Ubuntu 24.04 LTS cùng NVIDIA Driver 590.x. Kiến trúc Hopper của H200 mang lại những lợi thế phần cứng vượt trội, bao gồm bộ nhớ HBM3e lên đến 141GB và băng thông bộ nhớ đạt mức 4.8 TB/s. Những thông số này đặc biệt quan trọng bởi vì giai đoạn giải mã (decode phase) trong quá trình suy luận LLM hoàn toàn bị giới hạn bởi băng thông bộ nhớ (memory-bandwidth bound) thay vì năng lực tính toán thuần túy (compute bound). Việc hiểu rõ kiến trúc phần cứng là nền tảng để lựa chọn phương pháp tối ưu phù hợp nhất.

---

## 2. Chiến lược Phân bổ Nguồn lực và Kiến trúc Hệ thống Phát triển

Khó khăn lớn nhất của các đội thi quy mô nhỏ không nằm ở việc thiếu các ý tưởng thuật toán, mà là sự chênh lệch sâu sắc giữa môi trường phát triển cục bộ và hạ tầng đánh giá. Với hiện trạng nguồn lực bao gồm 3 máy tính cá nhân (16GB và 32GB RAM), một lượng tín dụng (credit) trên nền tảng AMD Cloud, và quyền truy cập Firework API, việc thiết lập một đường ống tích hợp và phát triển liên tục (CI/CD) là yếu tố sống còn.

### 2.1 Kiến trúc Triển khai Đa Môi trường

Không một máy tính cá nhân nào với 16GB hoặc 32GB RAM có thể tải toàn bộ trọng số của các mô hình kích thước lớn (trên 30B tham số) ở định dạng không nén để kiểm thử hiệu năng thực tế. Do đó, quy trình phát triển phải được phân lớp thành ba môi trường riêng biệt:

- **Local Development (Phát triển Cục bộ)**: Tại đây, các máy 16GB và 32GB RAM được sử dụng để viết mã logic, xây dựng cấu trúc Docker, và thử nghiệm các mô hình thu nhỏ (proxy models) có kiến trúc tương đương (ví dụ: Llama-3-8B hoặc Qwen3-8B). Môi trường này cho phép kiểm thử tính đúng đắn của API (chuẩn OpenAI-compatible), kiểm tra luồng trả về (streaming response) và các kịch bản bắt lỗi (exception handling) mà không cần tiêu tốn tài nguyên đám mây.
- **Baseline & Proxy Testing (Đánh giá Căn bản)**: Thông qua Firework API. API này đóng vai trò là "Ground Truth" (Chân lý nền). Khi không có đủ phần cứng để chạy mô hình BF16 nguyên bản nhằm đo lường điểm GPQA Diamond, hệ thống sẽ gọi Firework API để lấy điểm số chuẩn. Điều này giúp thiết lập ngưỡng cơ sở để so sánh mức độ suy giảm chất lượng $\Delta$ của các phiên bản mô hình đã lượng tử hóa.
- **Staging (Kiểm thử Cận sản xuất)**: Trên AMD Cloud. Các nền tảng đám mây của AMD (như MI300X hoặc MI355X) sở hữu bộ nhớ thống nhất lên đến 192GB, cho phép tải toàn bộ mô hình lớn mà không cần chia nhỏ (sharding). Mặc dù hệ sinh thái ROCm của AMD có những điểm khác biệt so với CUDA của NVIDIA, phần mềm vLLM hiện nay đã hỗ trợ ROCm như một nền tảng hạng nhất (first-class platform) thông qua các Docker image chính thức. Việc sử dụng môi trường này cho phép chạy các bài kiểm thử `lm-evaluation-harness` toàn diện và đánh giá hiệu quả của các cấu hình lập lịch.

### 2.2 Phân công Nhiệm vụ Chuyên biệt

Để tối đa hóa hiệu suất của nhóm 3 người (gồm 2 Kỹ sư AI và 1 Kỹ sư DevOps), công việc cần được mô-đun hóa chặt chẽ:

| Vai trò | Trách nhiệm Cốt lõi | Công cụ & Môi trường Đặc thù |
| :--- | :--- | :--- |
| **Kỹ sư AI 1**<br>*(Core Model & Quantization)* | Chịu trách nhiệm nén mô hình. Thực thi các thuật toán lượng tử hóa W8A8, phân tích suy giảm chất lượng, và hiệu chỉnh bộ nhớ đệm KV Cache. | `llm-compressor`, Local PC (32GB RAM), Python |
| **Kỹ sư AI 2**<br>*(Serving & Scheduling)* | Tối ưu hóa các thuật toán lập lịch. Triển khai Speculative Decoding (EAGLE-3), Chunked Prefill, và Prefix Caching để cân bằng TTFT/TPOT. | AMD Cloud, vLLM Configuration, Phân tích Log |
| **Kỹ sư DevOps**<br>*(System Arch & Eval)* | Xây dựng pipeline CI/CD. Đóng gói Dockerfile chuẩn cho H200. Tự động hóa quá trình chạy lm-eval và giả lập ERS Trace. Kiểm soát Anti-Cheating. | Docker, `lm-evaluation-harness`, Bash, Git |

Kỹ sư DevOps phải đảm bảo rằng mã nguồn và cấu hình được phát triển trên ROCm (AMD Cloud) có khả năng chuyển đổi liền mạch sang CUDA (NVIDIA H200). Điều này đòi hỏi việc sử dụng các tham số cấu hình linh hoạt thông qua biến môi trường (Environment Variables) trong Dockerfile, thay vì hardcode trực tiếp các kiến trúc chỉ hỗ trợ riêng cho một nền tảng.

---

## 3. Quyết định Chiến lược: Động cơ Suy luận (Inference Engine)

Trong không gian tối ưu hóa LLM, Động cơ Suy luận (Inference Engine) là trái tim của hệ thống. Quyết định lựa chọn engine sẽ định hình toàn bộ các kỹ thuật tối ưu hóa có thể áp dụng sau đó. Ba giải pháp hàng đầu hiện nay trong ngành công nghiệp bao gồm TensorRT-LLM của NVIDIA, SGLang, và vLLM.

- **TensorRT-LLM**: Là bộ công cụ phát triển phần mềm (SDK) chính thức của NVIDIA. Nó mang lại thông lượng tối đa tuyệt đối ở mức tải cao (hơn 100 yêu cầu đồng thời) và được tối ưu hóa sâu sắc cho kiến trúc Hopper. Tuy nhiên, chi phí học tập và triển khai của TensorRT-LLM là rất dốc. Quá trình biên dịch trước (Ahead-Of-Time compilation) phức tạp, đòi hỏi thời gian tinh chỉnh đồ thị tính toán có thể lên tới nhiều tuần. Đối với một nhóm nhỏ, rủi ro gặp lỗi biên dịch (compilation errors) hoặc lỗi nhân CUDA tùy chỉnh là quá lớn, có thể làm hỏng toàn bộ tiến độ cuộc thi.
- **SGLang**: Nổi lên như một ứng cử viên mạnh mẽ với cơ chế RadixAttention, đặc biệt vượt trội trong các kịch bản hội thoại đa lượt và khi triển khai các mô hình có cấu trúc Multi-Head Latent Attention (MLA) như DeepSeek. Dù vậy, độ phủ mô hình của SGLang hẹp hơn và việc cài đặt các thư viện phụ thuộc như FlashInfer đôi khi gặp vấn đề về tính tương thích.
- **vLLM**: Phân tích kiến trúc chỉ ra rằng vLLM là sự lựa chọn an toàn và hiệu quả nhất. Khoảng cách về thông lượng giữa một kịch bản FastAPI cơ bản và vLLM có thể lên tới 10 đến 24 lần. vLLM cung cấp các tính năng "miễn phí" ngay khi xuất xưởng (out-of-the-box), bao gồm lập lịch theo từng vòng lặp (iteration-level scheduling), PagedAttention, bộ nhớ đệm tiền tố tự động (Automatic Prefix Caching - APC), và hỗ trợ lượng tử hóa toàn diện. Quan trọng hơn, cộng đồng mã nguồn mở khổng lồ của vLLM đảm bảo khả năng gỡ lỗi nhanh chóng, một yếu tố then chốt khi nhóm phải làm việc dưới áp lực thời gian của cuộc thi.

---

## 4. Tối ưu Hóa Toán học: Lượng tử hóa W8A8 trên H200

Khi mô hình xử lý chuỗi văn bản dài, quá trình giải mã (decode) bị giới hạn vật lý bởi tốc độ mà dữ liệu trọng số (weights) có thể được kéo từ bộ nhớ HBM vào các đơn vị tính toán (Streaming Multiprocessors). Lượng tử hóa (Quantization) giải quyết nút thắt này bằng cách giảm độ phân giải của dữ liệu, từ đó giảm áp lực lên băng thông bộ nhớ và tăng tốc độ truyền tải.

### 4.1 Điểm Giao thoa Hoàn hảo của Định dạng FP8

Trong lịch sử, lượng tử hóa INT8 hoặc 4-bit (như AWQ, GPTQ) thường yêu cầu các phép tính giải lượng tử hóa (dequantization) phức tạp bằng phần mềm, dẫn đến hiện tượng trễ (overhead). Kiến trúc Hopper của GPU H200 đánh dấu một bước ngoặt khi tích hợp các Tensor Cores hỗ trợ phần cứng nguyên bản (native hardware support) cho định dạng dấu phẩy động 8-bit (FP8), cụ thể là định dạng `e4m3`.

Sự hỗ trợ nguyên bản này có nghĩa là GPU không cần thực hiện giả lập phần mềm; nó có thể xử lý toán hạng FP8 trực tiếp ở tốc độ tối đa. Dữ liệu thực nghiệm cho thấy lượng tử hóa FP8 W8A8 (Trọng số 8-bit, Kích hoạt 8-bit) trên H200 mang lại mức tăng thông lượng từ 1.4 đến 1.7 lần so với BF16 trong quá trình giải mã. Về mặt chất lượng, thử nghiệm đa mô hình trên các bài kiểm tra suy luận phức tạp (như MMLU-Pro hay GPQA) chứng minh rằng mức suy giảm chất lượng của FP8 chỉ dao động từ -0.3 đến -0.5 điểm. Đây là một con số nằm trong biên độ nhiễu thống kê và hoàn toàn an toàn để vượt qua Accuracy Gate của Đề thi, trái ngược với lượng tử hóa 4-bit AWQ có thể làm giảm tới 1.6 điểm và đối mặt với rủi ro bị phạt bởi hàm $f(\Delta)$.

### 4.2 Triển khai Nén Động và Tĩnh bằng LLM-Compressor

Hệ sinh thái vLLM quy chuẩn hóa việc nén mô hình thông qua thư viện `llm-compressor`. Đối với nhóm 3 người, việc này được Kỹ sư AI 1 đảm nhận. Có hai cách tiếp cận để thực thi nén FP8: lượng tử hóa động trực tuyến (Online Dynamic Quantization) và lượng tử hóa tĩnh ngoại tuyến (Offline Static Quantization).

- **Lượng tử hóa động trực tuyến**: Có thể được kích hoạt chỉ bằng một tham số cờ `--quantization="fp8"` khi khởi động vLLM. Phương pháp này ép engine tự động chuyển đổi mô hình trên bộ nhớ RAM của GPU mà không cần dữ liệu hiệu chuẩn (calibration data). Mặc dù thuận tiện cho việc thử nghiệm cục bộ, nó tạo ra độ trễ lớn khi khởi động (cold start) và tiêu thụ tài nguyên tính toán không cần thiết trong môi trường sản xuất.
- **Lượng tử hóa tĩnh ngoại tuyến**: Là phương pháp bắt buộc cho tệp nộp bài (submission) cuối cùng. `llm-compressor` sử dụng chiến lược áp dụng các hệ số tỷ lệ (scaling factors) tĩnh theo từng kênh (per-channel) cho trọng số, kết hợp với các hệ số tỷ lệ động theo từng token (per-token) cho hàm kích hoạt. Lớp đầu ra cuối cùng (như `lm_head`) phải được giữ nguyên ở định dạng BF16 để bảo toàn độ chính xác của phân phối xác suất.

Toàn bộ kịch bản Python để chuyển đổi mô hình được thực hiện như sau:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Khởi tạo mô hình BF16 gốc (cần chạy trên AMD Cloud do yêu cầu RAM)
model_id = "path/to/original/baseline/model"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Cấu hình công thức nén W8A8 FP8 động
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"]  # Quan trọng: Bảo toàn lớp dự đoán token
)

# Thực thi thuật toán và lưu trữ
oneshot(model=model, recipe=recipe)
save_directory = "/output/FP8_Model_vLLM"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
```

Kết quả của đoạn mã này là một mô hình đã được nén theo định dạng `compressed-tensors`, có thể được vLLM tải trực tiếp vào bộ nhớ một cách tức thì với độ trễ khởi động bằng không.

---

## 5. Quản trị Không gian Trạng thái: Tối ưu Hóa KV Cache

Trong các luồng công việc ngôn ngữ lớn, sự tính toán chú ý (attention mechanism) yêu cầu hệ thống phải lưu trữ lịch sử của tất cả các token trước đó dưới dạng các vector Key và Value (KV Cache). Kích thước của KV Cache tăng theo cấp số nhân dựa trên độ dài ngữ cảnh và kích thước lô (batch size). Sự cạn kiệt bộ nhớ VRAM do KV Cache phình to là nguyên nhân chính dẫn đến lỗi Out-Of-Memory (OOM) trong môi trường sản xuất.

Toán học đằng sau bộ nhớ KV Cache được tính bằng công thức:

$$2 \times L \times H \times D \times S \times B$$

*Trong đó: $L$ là số lớp, $H$ là số đầu attention, $D$ là số chiều, $S$ là độ dài chuỗi, và $B$ là số byte (2 byte cho FP16).*

Nếu không có chiến lược tối ưu, chỉ vài trăm yêu cầu đồng thời có thể chiếm đoạt hàng trăm Gigabyte bộ nhớ.

### 5.1 Giải phẫu Thuật toán PagedAttention

Các hệ thống suy luận ngây thơ phân bổ bộ nhớ KV Cache dưới dạng các mảng tuyến tính liền kề (contiguous arrays) dựa trên độ dài tối đa tiềm năng của một yêu cầu. Điều này dẫn đến hiện tượng phân mảnh nội bộ (internal fragmentation) nghiêm trọng, lãng phí từ 60% đến 80% dung lượng VRAM.

vLLM giải quyết triệt để bài toán này bằng thuật toán PagedAttention, một kỹ thuật mượn ý tưởng từ cơ chế phân trang bộ nhớ ảo (virtual memory paging) của hệ điều hành. PagedAttention chia cắt KV Cache thành các khối (block) cố định, mặc định chứa 16 token. Các khối này được lưu trữ không liền kề trong bộ nhớ vật lý và được ánh xạ (mapped) tới luồng yêu cầu thông qua bảng khối (block tables). Cơ chế này cho phép cấp phát bộ nhớ theo nhu cầu (on-demand), ép tỷ lệ lãng phí bộ nhớ xuống dưới 4%, mở khóa khả năng tăng kích thước lô phục vụ lên gấp 2 đến 4 lần.

Tham số `--gpu-memory-utilization` điều khiển phần trăm VRAM khả dụng được trao cho trình quản lý PagedAttention sau khi đã tải trọng số mô hình. Mức mặc định là `0.90` (chiếm 90% VRAM còn lại). Trong kịch bản thi đấu trên H200 (nơi có sẵn 141GB VRAM), kỹ sư DevOps nên tinh chỉnh tham số này lên mức `0.93` để cực đại hóa không gian cho KV Cache, nhưng vẫn phải duy trì 7% headroom (khoảng trống) để đề phòng các đỉnh đột biến (spikes) trong quá trình cấp phát bộ nhớ kích hoạt (activation memory) khi xử lý các chuỗi prefill cực lớn.

### 5.2 Tối ưu Hội thoại Đa lượt qua Automatic Prefix Caching (APC)

Workload Trace của cuộc thi có tính chất đa lượt (multi-turn), trong đó lượt kế tiếp sẽ gửi kèm toàn bộ lịch sử hội thoại trước đó. Nếu tính toán lại (recompute) toàn bộ chuỗi lịch sử này từ đầu, hệ thống sẽ lãng phí tài nguyên khổng lồ.

Automatic Prefix Caching (APC) là giải pháp hoàn hảo cho đặc tả này. Bằng cách kích hoạt tham số `--enable-prefix-caching`, vLLM sẽ tính toán mã băm (hash) cho mỗi khối KV Cache. Khi một yêu cầu mới có đoạn đầu văn bản (tiền tố) trùng khớp với lịch sử đã lưu, vLLM sẽ trỏ trực tiếp đến các khối vật lý trong bộ nhớ thay vì tính toán lại, biến chi phí prefill của đoạn tiền tố đó về gần bằng không.

Từ vLLM phiên bản 0.11, thuật toán băm mặc định là `sha256`. Tuy nhiên, để tối ưu hóa độ trễ tính toán ở mức vi mô cho chỉ số TTFT, nhóm nên cấu hình `--prefix-caching-hash-algo xxhash`. `xxhash` là thuật toán băm phi mật mã (non-cryptographic), cung cấp tốc độ băm nhanh hơn đáng kể so với `sha256`, đặc biệt hiệu quả trên các máy chủ phải xử lý lưu lượng băm liên tục.

### 5.3 Giải quyết Lỗi Lượng tử hóa KV Cache FP8 trên Kiến trúc Hopper

Để khai thác triệt để băng thông bộ nhớ của H200, việc lượng tử hóa KV Cache từ 16-bit xuống 8-bit (thông qua cờ `--kv-cache-dtype fp8`) là một bước đi bắt buộc. Quá trình này nhân đôi số lượng token có thể lưu trữ và giảm hệ số góc của ITL (Inter-Token Latency).

Tuy nhiên, đây là cạm bẫy kỹ thuật lớn nhất đối với hạ tầng Hopper. Các thử nghiệm từ vLLM phát hiện ra rằng, hạt nhân (kernel) Flash Attention 3 chạy trên Tensor Cores FP8 của Hopper mắc lỗi mất độ chính xác tích lũy (accumulation precision loss). Khi độ dài ngữ cảnh tăng, sai số làm tròn trong các thanh ghi nội bộ cộng dồn, dẫn đến việc độ chính xác trong tác vụ tìm kiếm dài (như Needle-in-a-haystack) sụp đổ từ 91% xuống mức 13%. Việc dính lỗi này chắc chắn sẽ khiến bài nộp bị Cổng Accuracy Gate đánh trượt.

Khắc phục vấn đề này yêu cầu Kỹ sư AI 2 phải đảm bảo hệ thống vLLM được cài đặt phiên bản mới nhất, có tích hợp bản vá "Tích lũy hai cấp" (Two-level accumulation) cho Flash Attention. Cơ chế này buộc các kết quả tích lũy một phần phải được ghi vào các thanh ghi FP32 vật lý thực thụ, giúp khôi phục độ chính xác lên mức 89%, đảm bảo tính an toàn cho hệ thống hậu kiểm.

---

## 6. Làm phẳng Đường cong Độ trễ: Chunked Prefill và Lập lịch Liên tục

Hệ thống chấm điểm ERS không đánh giá toàn cục, mà đánh giá tính liên tục. Một yêu cầu có thể bị phạt nặng nếu TTFT hoặc TPOT vượt quá ngưỡng giới hạn (Ceiling). Sự gián đoạn trong phục vụ là nguyên nhân chính gây mất điểm.

### 6.1 Cơ chế Lập lịch Lô Liên tục (Continuous Batching)

Theo mô hình truyền thống (Static Batching), hệ thống đợi một lô (batch) các yêu cầu hoàn thành toàn bộ quá trình giải mã trước khi tiếp nhận lô mới. Điều này gây lãng phí tới 60% tài nguyên GPU khi các câu trả lời ngắn kết thúc trước. 

vLLM áp dụng Lập lịch Lô Liên tục (Continuous hay Iteration-level Batching), trong đó các yêu cầu mới được tiêm trực tiếp vào GPU ở cấp độ từng bước token (token step), và các yêu cầu hoàn thành sẽ nhả tài nguyên ngay lập tức. Cơ chế này kết hợp với PagedAttention tạo ra mức tăng thông lượng lên tới 24 lần ở mức tải cao.

### 6.2 Hiện tượng Tắc nghẽn Đầu hàng (Head-of-Line Blocking) và Giải pháp Chunked Prefill

Continuous Batching nảy sinh một tác dụng phụ nghiêm trọng. Giai đoạn tiền xử lý (prefill phase) để tính toán ma trận Attention cho một chuỗi prompt dài (ví dụ: 32.000 token) tiêu tốn lượng tính toán khổng lồ và mất từ 200 đến 400 mili-giây. Trong quá trình này, GPU dồn toàn lực cho việc tính toán (compute-bound) và đóng băng hoàn toàn việc sinh token cho các yêu cầu khác đang ở giai đoạn giải mã. Hậu quả là TPOT của các yêu cầu đang giải mã sẽ tăng vọt, tạo ra các đỉnh nhiễu (spikes) trong phân phối độ trễ và phá hủy điểm số ERS.

Giải pháp cấu trúc cho vấn đề này là Chunked Prefill (Chia nhỏ tiền xử lý). Bằng cách kích hoạt tham số `--enable-chunked-prefill`, vLLM chia nhỏ quá trình prefill khổng lồ thành các phân đoạn (chunk) nhỏ hơn. Kỹ thuật này cho phép bộ lập lịch đan xen (interleave) các bước giải mã của các yêu cầu khác vào giữa các phân đoạn prefill của yêu cầu mới. Việc này dung hòa hoàn hảo xung đột giữa TTFT và TPOT, giữ cho luồng sinh token luôn mượt mà.

Tham số quan trọng đi kèm là `--max-num-batched-tokens`. Tham số này đặt ngân sách (budget) tổng số token được phép xử lý trong một chu kỳ lập lịch. Nếu ưu tiên thông lượng (throughput), có thể đặt ở mức 16384 hoặc 32768. Tuy nhiên, đối với cuộc thi nơi ERS cực kỳ nhạy cảm với độ trễ vi mô, thiết lập mức thấp như `2048` sẽ ép bộ lập lịch chia nhỏ prefill nhiều hơn, ưu tiên sự công bằng (fairness) và làm mịn độ trễ TPOT.

> [!WARNING]
> **Lưu ý cho môi trường Staging:** Có ghi nhận lỗi xung đột giữa Chunked Prefill và ROCm trên các nền tảng AMD MI300X, gây lỗi tràn bộ nhớ (OOM) khi khởi tạo. Kỹ sư DevOps cần kiểm tra kỹ và có thể phải tạm tắt tính năng này bằng cờ `--no-enable-chunked-prefill` (dù điều này không được khuyến khích trên v1) khi chạy thử nghiệm trên AMD Cloud, nhưng chắc chắn phải bật lại khi đóng gói cho kiến trúc CUDA H200.

---

## 7. Bức phá Nút thắt Băng thông: Đầu cơ Giải mã (Speculative Decoding)

Ngay cả khi đã tối ưu hóa PagedAttention và Chunked Prefill, giai đoạn giải mã vẫn mang bản chất tuần tự (sinh từng token một), khiến các vi xử lý của H200 nhàn rỗi (underutilized) trong khi chờ dữ liệu truyền từ bộ nhớ.

Đầu cơ Giải mã (Speculative Decoding) phá vỡ nút thắt này thông qua mô hình "dự đoán rồi kiểm chứng" (draft-then-verify). Một mô hình nháp (draft model) nhỏ, tiêu thụ ít băng thông, sẽ dự đoán trước $K$ token. Mô hình chính (target model) sẽ nhận mảng $K$ token này và kiểm chứng toàn bộ chúng trong một lần quét song song duy nhất (một forward pass). Toán học chứng minh rằng, xác suất chấp nhận chuỗi dự đoán được bảo toàn thông qua kỹ thuật lấy mẫu từ chối (rejection sampling), đảm bảo phân phối đầu ra hoàn toàn không bị biến dạng (Lossless Algorithmic Guarantee). Sự bảo toàn phân phối này cực kỳ quan trọng vì nó đáp ứng hoàn hảo quy tắc không làm suy giảm độ chính xác của Cổng Accuracy Gate.

### 7.1 Sự tiến hóa của EAGLE-3

Mô hình đầu cơ ngây thơ sử dụng một mô hình hoàn toàn khác để dự đoán thường gặp vấn đề chênh lệch miền dữ liệu (domain mismatch) và tỷ lệ chấp nhận (acceptance rate) rất thấp. Thay vào đó, hệ thống vLLM hỗ trợ tích hợp EAGLE-3, một kỹ thuật đính kèm trực tiếp một "đầu nháp" (draft head) siêu nhẹ – chỉ bằng 2% đến 5% kích thước mô hình chính – lên trên mô hình nền tảng.

EAGLE-3 vượt trội nhờ kỹ thuật Dung hợp Đặc trưng Ba Tầng (Tri-Layer Feature Fusion), kết hợp ngữ cảnh cú pháp (tầng sớm), cấu trúc ngữ nghĩa (tầng giữa), và phân phối xác suất (tầng muộn) để dự đoán token. Cơ chế này nâng cao đáng kể tỷ lệ chấp nhận của các token đầu cơ, đem lại mức tăng tốc từ 2 đến 6 lần trên lý thuyết.

Hơn nữa, báo cáo đề xuất kích hoạt biến thể P-EAGLE (Parallel EAGLE) thông qua thuộc tính `"parallel_drafting": true` trong cấu hình. Thuật toán phân vùng chuỗi của P-EAGLE chia chuỗi vị trí thành các phần liên tiếp, tính toán mô hình nháp song song nhưng vẫn duy trì đúng sự phụ thuộc chú ý (attention dependencies), triệt tiêu hoàn toàn độ trễ tích lũy khi mô hình nháp phải tự sinh tuần tự.

### 7.2 Điểm Ngọt (Sweet Spot) của Ngân sách Token

Một sai lầm phổ biến là dự đoán quá nhiều token (ví dụ: $K = 5$ hoặc $7$). Khi $K$ càng lớn, thời gian dự đoán của mô hình nháp càng cao, và chi phí loại bỏ các token bị từ chối càng lớn, làm phản tác dụng và gây hại cho TTFT.

Thử nghiệm chuyên sâu trên máy chủ đa GPU cho thấy thiết lập `num_speculative_tokens` bằng **2** hoặc **3** tạo ra hiệu suất quang học nhất. Ở mức này, thông lượng sinh token (output throughput) tăng hơn 20%, trong khi độ trễ toàn trình giảm từ 12% đến 20%, và ITL cải thiện 12.4%. Điều này biến EAGLE-3 trở thành một đòn bẩy ERS vô cùng mạnh mẽ mà không tốn chi phí chất lượng.

---

## 8. Tuân thủ Chống Gian lận (Anti-Cheating) và Đánh giá Cổng Chất lượng

Điều lệ cuộc thi Viettel AI Race 2026 mang tinh thần của một môi trường sản xuất thực thụ. Điểm ERS tự động trên Leaderboard online chỉ là bề nổi; 5 bài nộp tốt nhất của đội sẽ được Ban Tổ chức chọn thủ công để rà soát hành vi phục vụ (serving behavior) và chạy qua bộ kiểm định GPQA Diamond.

### 8.1 Giới hạn của Các Kỹ thuật Tối ưu Hợp lệ

Quy tắc cốt lõi là không được phép rẽ nhánh hành vi (Dual-path): hệ thống không được nhận diện khi nào bị đo ERS để chạy nhanh (và sinh chuỗi kém chất lượng), và khi nào bị đo GPQA để chạy chậm (và sinh chuỗi chính xác).

Chiến lược kiến trúc được mô tả trong báo cáo này hoàn toàn hợp lệ và trung thực. Lượng tử hóa W8A8 FP8 áp dụng suy luận đồng nhất trên mọi luồng dữ liệu. PagedAttention, Prefix Caching và Chunked Prefill là các kỹ thuật tối ưu hóa cấp hệ thống (System-Level Optimization), chỉ làm thay đổi cách VRAM được quản lý và lịch trình luồng dữ liệu chứ không can thiệp vào toán học của mô hình. Tương tự, Speculative Decoding với chứng minh Lossless (không mất mát) đảm bảo phân phối xác suất hoàn toàn minh bạch và nhất quán. Nhóm không được sử dụng bất kỳ chiêu trò đệm rỗng (dummy padding) hay gọi API bên ngoài để trích xuất đáp án.

### 8.2 Giả lập Môi trường LM-Evaluation-Harness

Để tránh bị phạt bởi hàm tuyến tính từng khúc $f(\Delta)$, Kỹ sư DevOps phải đo lường độ chính xác cục bộ (trên AMD Cloud) trước khi nộp bài. Đề bài sử dụng `lm-evaluation-harness` của EleutherAI.

Quá trình giả lập yêu cầu khởi động một tiến trình vLLM cục bộ, sau đó trỏ công cụ đánh giá vào API của tiến trình này. Một chi tiết vi mô nhưng mang tính sống còn đối với các mô hình lượng tử hóa là định dạng mồi (Prompt Framing). Các mô hình lượng tử FP8 rất nhạy cảm với việc mất token bắt đầu câu (bos token). Khi chạy `lm_eval`, phải bắt buộc thêm cờ `add_bos_token=True` trong tham số `model_args` để ngăn chặn sự sụp đổ chất lượng đột ngột. Đồng thời, `temperature` phải được thiết lập cứng bằng `0.0` để loại bỏ yếu tố ngẫu nhiên (stochastic) khi so sánh với baseline.

Cấu trúc lệnh giả lập quá trình hậu kiểm của Ban Tổ chức:

```bash
lm_eval --model local-chat-completions \
  --tasks gpqa_diamond \
  --model_args "model=/path/to/FP8_Model,max_length=32768,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=32,tokenized_requests=False,add_bos_token=True" \
  --num_fewshot 0 \
  --batch_size auto
```

Điểm số này sẽ được so sánh với điểm Ground Truth lấy từ Firework API để tính toán trước mức độ phạt $\Delta$.

---

## 9. Đóng gói Kiến trúc và Mã Lệnh Triển khai (Production Deployment)

Sản phẩm bàn giao (Deliverable) cuối cùng là một ảnh container (Docker Image) tĩnh, tự chứa toàn bộ trọng số, cấu hình và mã khởi động. Kỹ sư DevOps cần viết một Dockerfile dựa trên nền tảng `nvidia/cuda:13.x-base-ubuntu24.04`, cài đặt phiên bản vLLM tương thích, và sao chép các tệp trọng số nén FP8 tĩnh vào bên trong container. Tuyệt đối không để ứng dụng tải dữ liệu từ internet khi khởi chạy để tránh vi phạm quy tắc "Can thiệp hạ tầng".

Lệnh khởi động (Entrypoint) của container là linh hồn của toàn bộ hệ thống, tích hợp mọi chiến lược đã phân tích:

```bash
# Lệnh Entrypoint Khởi động vLLM Server trên H200
vllm serve /model_weights/LLM-FP8-W8A8 \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.93 \
    --max-model-len 32768 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --prefix-caching-hash-algo xxhash \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048 \
    --speculative-config '{"method": "eagle3", "model": "/model_weights/EAGLE3-Draft", "num_speculative_tokens": 3, "parallel_drafting": true}'
```

Sự kết hợp của các cấu hình này tạo ra một vòng tròn tối ưu hóa khép kín:
- **Lượng tử hóa FP8 W8A8** ép nhỏ kích thước mô hình
- **Paged Attention và FP8 KV Cache** mở rộng dung lượng bộ nhớ
- **APC** loại bỏ chi phí prefill dư thừa của hội thoại đa lượt
- **Chunked Prefill** bảo vệ độ trễ liên tục
- **P-EAGLE-3** khuếch đại thông lượng giải mã

Cấu trúc hệ thống vững chắc này đáp ứng mọi tiêu chí khắc nghiệt của Viettel AI Race 2026, cung cấp cho nhóm một lộ trình khả thi để cạnh tranh sòng phẳng ở vị trí dẫn đầu bảng xếp hạng.