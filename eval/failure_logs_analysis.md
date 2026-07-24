# 📑 TỔNG HỢP LOG LỖI LỊCH SỬ & NGUYÊN NHÂN THẤT BẠI
## Viettel AI Race 2026 — Challenge 3 · Team Develarper

> **Mục đích:** Lưu trữ toàn bộ nhật ký lỗi (error logs) thực tế từ hệ thống BTC trong suốt quá trình thử nghiệm của cá nhân (**Long Quân Tôn**) và đồng đội (**Yoshio**), phân tích nguyên nhân kỹ thuật chi tiết để tránh lặp lại sai lầm.

---

## 1. Danh Sách Các Lỗi Crash Container & Thất Bại Benchmark

### ❌ Lỗi 1: Gán Giá Trị Cho Cờ Boolean (`--enable-chunked-prefill=true`)
* **Lần xuất hiện:** Nhánh Yoshio.
* **Mã lỗi:** Exit Code 2.
* **Trích đoạn Log từ BTC:**
  ```text
  spawn contestant container: wait for pod ready: contestant pod failed: container "inference" exited 2 (Error)
  api_server.py: error: argument --enable-chunked-prefill/--no-enable-chunked-prefill: ignored explicit argument 'true'
  ```
* **Nguyên nhân kỹ thuật:**
  * Trong vLLM CLI parser (`argparse`), cờ `--enable-chunked-prefill` là cờ Boolean dạng `store_true` (chỉ khai báo tên cờ, không truyền giá trị đằng sau).
  * Việc truyền `--enable-chunked-prefill=true` hoặc `--enable-chunked-prefill true` khiến Python `argparse` báo lỗi cú pháp `ignored explicit argument 'true'` và hủy chạy container.
* **Giải pháp khắc phục:**
  * Khai báo cờ độc lập không chứa giá trị gán: `- --enable-chunked-prefill`.

---

### ❌ Lỗi 2: Cờ CLI Không Tồn Tại (`--disable-log-requests`)
* **Lần xuất hiện:** Nhánh QuanTon (Commit `5b2a93a`).
* **Mã lỗi:** Exit Code 2.
* **Trích đoạn Log từ BTC:**
  ```text
  spawn contestant container: wait for pod ready: contestant pod failed: container "inference" exited 2 (Error)
  api_server.py: error: unrecognized arguments: --disable-log-requests
  ```
* **Nguyên nhân kỹ thuật:**
  * vLLM API Server (`api_server.py`) không hỗ trợ tham số CLI tên `--disable-log-requests` (chỉ hỗ trợ `--enable-log-requests` / `--no-enable-log-requests`).
  * Python `argparse` lập tức dừng tiến trình với báo lỗi `unrecognized arguments`.
* **Giải pháp khắc phục:**
  * Loại bỏ hoàn toàn cờ `--disable-log-requests` ra khỏi file `docker-compose.yml`.

---

### ❌ Lỗi 3: Truyền Sai Mamba Backend (`--mamba-backend=CUDA`)
* **Lần xuất hiện:** Nhánh Yoshio (Bài `#6`).
* **Mã lỗi:** Exit Code 1.
* **Trích đoạn Log từ BTC:**
  ```text
  spawn contestant container: wait for pod ready: contestant pod failed: container "inference" exited 1 (Error)
  (APIServer pid=1) File "/usr/local/lib/python3.12/dist-packages/vllm/config/mamba.py", line 20, in __getitem__
  (APIServer pid=1)     raise ValueError(
  (APIServer pid=1) ValueError: Unknown Mamba SSU backend: 'CUDA'. Valid options are: TRITON, FLASHINFER
  ```
* **Nguyên nhân kỹ thuật:**
  * Cấu hình vLLM Mamba config (`vllm/config/mamba.py`) chỉ định nghĩa 2 backend hợp lệ cho SSU là `TRITON` và `FLASHINFER`.
  * Truyền `--mamba-backend=CUDA` làm vLLM ném ngoại lệ `ValueError` và ngắt server.
* **Giải pháp khắc phục:**
  * Chỉ sử dụng cờ `- --mamba-backend=flashinfer` (kết hợp với image `p2-fi` cài sẵn `flashinfer-python`).

---

### ❌ Lỗi 4: Xung Đột Speculative Decoding N-Gram (`SpeculativeConfig ValidationError`)
* **Lần xuất hiện:** Nhánh Yoshio (Bài `#7`).
* **Mã lỗi:** Exit Code 1.
* **Trích đoạn Log từ BTC:**
  ```text
  spawn contestant container: wait for pod ready: contestant pod failed: container "inference" exited 1 (Error)
  (APIServer pid=1) File "/usr/local/lib/python3.12/dist-packages/vllm/engine/arg_utils.py", line 1751, in create_speculative_config
  (APIServer pid=1)     return SpeculativeConfig(**self.speculative_config)
  (APIServer pid=1) pydantic_core._pydantic_core.ValidationError: 1 validation error for SpeculativeConfig
  (APIServer pid=1) ngram_prompt_lookup_max
  (APIServer pid=1)   Unexpected keyword argument [type=unexpected_keyword_argument, input_value=4, input_type=int]
  ```
* **Nguyên nhân kỹ thuật:**
  * Speculative Decoding dạng N-gram chứa cấu hình từ khóa `ngram_prompt_lookup_max` không khớp với Pydantic dataclass schema trong vLLM v0.25.1.
* **Giải pháp khắc phục:**
  * Tuyệt đối không dùng Speculative Decoding cho mô hình Hybrid Mamba `LFM2.5-1.2B-Instruct`.

---

### ❌ Lỗi 5: Thất Bại Kiểm Tra Probe Chống Gian Lận BTC (`Long-Context Probe Failed 0%`)
* **Lần xuất hiện:** Nhánh QuanTon (Bài `X1`).
* **Mã lỗi:** Protocol Aborted (Bị hệ thống BTC ngắt kết nối).
* **Trích đoạn Log từ BTC:**
  ```text
  protocol aborted: long-context probe failed (0%) — truncation / dual-path likely
  ```
* **Nguyên nhân kỹ thuật:**
  * Việc áp dụng Speculative N-gram hoặc cắt xẻ prompt quá đà làm vLLM sinh đầu ra bị trảm (truncation) hoặc trả về 0 token khi bộ test kiểm tra long-context probe của BTC gửi request ngầm.
* **Giải pháp khắc phục:**
  * Tối ưu serving chuẩn chính ngạch vLLM (Prefix caching, Chunked prefill, FP8, FlashInfer Mamba), không dùng các kỹ thuật dual-path hay n-gram không tương thích.

---

## 2. Bảng Tóm Tắt Quy Tắc Vàng Kiểm Tra `docker-compose.yml`

| Cờ CLI | Cú Pháp Đúng (Đã Thẩm Định) | Cú Pháp Sai Gây Lỗi (Cấm Sử Dụng) |
|---|---|---|
| Chunked Prefill | `- --enable-chunked-prefill` | ❌ `--enable-chunked-prefill=true` (Lỗi 1) |
| Logging | *(Để mặc định, không khai báo cờ log)* | ❌ `--disable-log-requests` (Lỗi 2) |
| Mamba Backend | `- --mamba-backend=flashinfer` | ❌ `--mamba-backend=CUDA` (Lỗi 3) |
| Speculative | *(Không khai báo)* | ❌ `--speculative-config` (Lỗi 4 & 5) |
