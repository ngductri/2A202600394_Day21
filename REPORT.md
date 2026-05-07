# Báo cáo Lab 21: Fine-tuning LLM với LoRA/QLoRA

- **Mô hình:** `unsloth/Qwen2.5-3B-bnb-4bit` (Quantized 4-bit)
- **Tập dữ liệu:** `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`
- **GPU sử dụng:** NVIDIA Tesla T4 (16GB VRAM)
- *Link for OUTPUT_DIR*: https://drive.google.com/drive/folders/1rm7FYLEFVK9It1l6Vx2ezBv9zkryTWFy?usp=sharing

---

## 1. Cài đặt & Baseline
Quá trình fine-tuning được thực hiện bằng thư viện **Unsloth**, giúp tăng tốc đáng kể và tối ưu bộ nhớ khi huấn luyện các Mô hình Ngôn ngữ Lớn.

### Siêu tham số (Hyperparameters)
- **Mô hình gốc:** Qwen2.5-3B (4-bit)
- **Target Modules của LoRA:** `q_proj`, `v_proj`
- **Độ dài chuỗi tối đa:** 2048
- **Tốc độ học (Learning Rate):** 2e-4
- **Kích thước Batch:** 1 (Kích thước Batch hiệu dụng: 8 thông qua Gradient Accumulation)
- **Số Epoch:** 3 (khoảng 60-70 bước tùy theo phân tách dữ liệu)
- **Tối ưu hóa:** AdamW 8-bit

### Kết quả Baseline (Rank r=16)
- **Tham số có thể huấn luyện:** 3,686,400 (0.12% tổng số tham số)
- **Thời gian huấn luyện:** ~4.1 phút
- **Loss đánh giá:** 1.5161
- **Độ hỗn loạn (Perplexity):** 4.55

---

## 2. Kết quả thí nghiệm Rank

Em đã thử nghiệm với ba mức rank LoRA (rr) khác nhau để quan sát sự đánh đổi giữa hiệu năng mô hình (perplexity), mức sử dụng bộ nhớ (VRAM) và tốc độ huấn luyện.

| Rank (rr) | Alpha (α\alpha) | Tham số huấn luyện | Thời gian (phút) | VRAM đỉnh (GB) | Loss đánh giá | Perplexity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **8** | 16 | 1,843,200 | 4.16 | 7.84 | 1.5577 | 4.75 |
| **16** | 32 | 3,686,400 | 4.11 | 7.24 | 1.5161 | 4.55 |
| **64** | 128 | 14,745,600 | 4.15 | 8.62 | 1.4768 | 4.38 |

**Quan sát:**
- **Hiệu năng:** Việc tăng rank từ 8 lên 64 dẫn đến việc giảm liên tục loss đánh giá (từ 1.5577 xuống 1.4768) và perplexity (từ 4.75 xuống 4.38). Điều này cho thấy rank cao hơn cho phép các adapter LoRA học được các đặc điểm phức tạp hơn trong các chỉ dẫn tiếng Việt.
- **Bộ nhớ (VRAM):** Mức sử dụng VRAM tăng dần theo rank, đặc biệt rõ rệt ở mức r=64r=64 (8.62 GB so với 7.84 GB ở r=8r=8). Đáng ngạc nhiên là r=16r=16 cho thấy mức VRAM đỉnh thấp hơn trong lần chạy này, có thể do vấn đề phân mảnh bộ nhớ hoặc cơ chế quản lý bộ nhớ cụ thể tại thời điểm đó.
- **Thời gian:** Thời gian huấn luyện duy trì khá ổn định ở các mức rank (~4.1 phút), cho thấy đối với quy mô cụ thể này, nút thắt cổ chai về tính toán không nằm chủ yếu ở việc cập nhật trọng số adapter.

### 2.1. So sánh LoRA và DoRA (với rank r=16)
Em thực hiện so sánh giữa phương pháp LoRA truyền thống và DoRA (Weight-Decomposed Low-Rank Adaptation) để kiểm tra tính hiệu quả của việc phân tách độ lớn (magnitude) và hướng (direction) trong quá trình thích nghi.

| Phương pháp | Rank (rr) | Thời gian (phút) | VRAM đỉnh (GB) | Loss huấn luyện | Perplexity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LoRA** | 16 | 3.65 | 10.82 | 1.4708 | 4.55 |
| **DoRA** | 16 | 4.34 | 11.69 | 1.4699 | 4.55 |

**Nhận xét:**
- **Độ chính xác:** DoRA mang lại mức loss huấn luyện thấp hơn một chút so với LoRA (1.4699 so với 1.4708), cho thấy khả năng tối ưu hóa tốt hơn của kiến trúc này. Tuy nhiên, sự khác biệt về Perplexity cuối cùng là không đáng kể trong thí nghiệm này.
- **Chi phí tài nguyên:** DoRA tiêu tốn nhiều bộ nhớ VRAM hơn (~11.69 GB so với 10.82 GB) và thời gian huấn luyện lâu hơn (~4.34 phút so với 3.65 phút) do các phép tính phân tách trọng số phức tạp hơn.

---

## 3. Phân tích đường cong Loss

Đường cong loss trong quá trình huấn luyện (quan sát trong notebook) cho thấy sự sụt giảm đều đặn qua 60-70 bước.
- **Loss ban đầu:** Bắt đầu trong khoảng 1.6 - 1.8.
- **Sự hội tụ:** Loss giảm nhanh trong 20 bước đầu tiên và sau đó ổn định, đạt khoảng 1.3 - 1.4 vào cuối epoch thứ 3.
- **Độ ổn định:** Không quan sát thấy hiện tượng nhảy vọt hay dao động lớn, cho thấy tốc độ học (2e-4) và bộ tối ưu AdamW 8-bit đã được điều chỉnh tốt cho tác vụ này.
- **Rank cao hơn:** Với $r=64$, đường cong hội tụ về mức loss cuối cùng thấp hơn so với $r=8$, xác thực kết quả định lượng.

---

## 4. So sánh định tính

Dưới đây là so sánh các phản hồi giữa **Mô hình gốc (Base)** và **Mô hình đã Fine-tuned (r=16)**.

| Câu lệnh (Prompt) | Phản hồi của Mô hình gốc | Phản hồi của Mô hình Fine-tuned |
| :--- | :--- | :--- |
| **Giải thích machine learning...** | Tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu... | Là bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện dự đoán... |
| **Code Python Fibonacci** | Sử dụng hàm đệ quy hoặc vòng lặp... (Code đơn giản) | Cung cấp đoạn code có xử lý lỗi (ValueError cho số âm)... |
| **Tóm tắt LoRA vs QLoRA** | Là phương pháp cải thiện hiệu năng NLU... | Giải thích chi tiết hơn về regularization và độ ổn định... |

**Đánh giá:**
- Mô hình fine-tuned cung cấp các phản hồi có cấu trúc tốt hơn và chú trọng đến tính an toàn (ví dụ: kiểm tra đầu vào trong các tác vụ lập trình).
- Sự lưu loát trong tiếng Việt được duy trì, và các hướng dẫn được tuân thủ chính xác hơn so với mô hình gốc (vốn đôi khi đưa ra các câu trả lời chung chung).

---

## 5. Kết luận
Việc fine-tuning Qwen2.5-3B sử dụng LoRA/QLoRA trên GPU T4 mang lại hiệu quả rất cao. Mức rank **r=16** hoặc **r=32** dường như là "điểm tối ưu" cho tập dữ liệu này, mang lại sự cải thiện đáng kể trong khả năng tuân thủ chỉ dẫn với mức tiêu tốn bộ nhớ tối thiểu. Mặc dù **r=64** mang lại các chỉ số tốt nhất, nhưng sự cải thiện gia tăng về perplexity có thể không phải lúc nào cũng bù đắp được việc tăng mức sử dụng VRAM trong môi trường sản xuất với tài nguyên hạn chế.

---

## 6. Nhìn nhận cá nhân

### Những gì em đã học được:
1.  **Hiệu quả của Unsloth:** Em rất ấn tượng với tốc độ huấn luyện (dưới 5 phút cho 3 epoch). Sử dụng quantization 4-bit (QLoRA) là điều bắt buộc để có thể chạy các mô hình 3B+ trên GPU T4 16GB.
2.  **Tác động của Rank:** Em đã học được rằng việc tăng rank LoRA không chỉ đơn thuần là "có thêm tham số" mà còn là cung cấp cho mô hình nhiều "mức độ tự do" hơn để thích nghi với một ngôn ngữ hoặc lĩnh vực cụ thể (trong trường hợp này là tiếng Việt).
3.  **Sự đánh đổi:** Em đã có kinh nghiệm thực tế trong việc quản lý bộ nhớ GPU. Quy trình `del trainer; gc.collect(); torch.cuda.empty_cache()` là "cứu cánh" khi chạy nhiều thí nghiệm trong một phiên để tránh lỗi OOM.
4.  **Các chỉ số đánh giá:** Perplexity là một chỉ số toán học hữu ích, nhưng kiểm thử định tính mới là nơi bạn thực sự thấy được mô hình có "cảm giác" tốt hơn khi tuân thủ các chỉ dẫn hay không.

### Thách thức:
- Ban đầu, em gặp vấn đề về bộ nhớ khi cố gắng đánh giá mô hình với kích thước batch lớn. Việc triển khai hàm `safe_evaluate` với `per_device_eval_batch_size=1` đã giải quyết được vấn đề này.
- Hiểu được mối quan hệ giữa `r` và `alpha` (đặt $\alpha = 2r$) là chìa khóa để duy trì sự ổn định trong huấn luyện.
