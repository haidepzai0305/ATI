# Project 3 – DataSEC 2026: Phân loại âm thanh môi trường dưới nhiễu âm học

Tổng hợp từ đề bài `docs/03-datasec-environmental-audio.pdf`. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh như trong đề.

## 1. Dataset và bài toán

DataSEC là bộ benchmark gồm các sự kiện âm thanh môi trường đã được tách riêng, công bố trên *Scientific Data* năm 2025 [1], dùng để huấn luyện và đánh giá mô hình Sound Event Classification (SEC).

| Thuộc tính | Giá trị |
|---|---|
| Số clip | 4.292 file `.wav` đơn kênh (monophonic) |
| Tổng thời lượng | khoảng 18 giờ 26 phút |
| Số lớp | 22 macro-class, 28 subclass |
| Tần số lấy mẫu | 44,1 kHz |
| Định dạng | đơn kênh, linear PCM không nén |

- **SEC khác SED.** Trong SEC, mỗi bản ghi chứa đúng một sự kiện âm thanh, và mô hình chỉ cần dự đoán lớp của nó. SED (Sound Event Detection) phải xác định cả lớp lẫn thời điểm bắt đầu và kết thúc của sự kiện trong luồng âm thanh liên tục, nhiều nguồn.
- **Nhiệm vụ:** phân loại 22 lớp trên log-Mel spectrogram. Không cần xây dựng hệ thống SED.
- **Giới hạn tính toán:** mọi thí nghiệm bắt buộc phải chạy được trên Google Colab bản miễn phí, với một GPU NVIDIA T4 16 GB VRAM.
- **Cố định phiên bản dataset:** chỉ dùng DOI `10.5281/zenodo.17033970`.

| Tài nguyên | Đường dẫn |
|---|---|
| Bài báo gốc | https://doi.org/10.1038/s41597-025-05991-w |
| DataSEC (bản dùng cho môn học) | https://zenodo.org/records/17033970 |
| DOI của DataSEC | https://doi.org/10.5281/zenodo.17033970 |

## 2. Giữa kỳ

### A. Phân tích dataset

Cần báo cáo:
- tổng số clip và phân bố theo 22 macro-class;
- thống kê thời lượng theo từng lớp: min, max, mean, median;
- dạng sóng miền thời gian và spectrogram tần số tuyến tính của ít nhất 3 lớp âm thanh tương phản nhau;
- biểu diễn log-Mel spectrogram 2D;
- đánh giá mức cân bằng giữa các lớp;
- các lớp có đường bao phổ chồng lấn hoặc hay bị nhầm lẫn với nhau.

Ngoài ra cần giải thích khác biệt toán học và vật lý giữa ba biểu diễn:
1. **dạng sóng thô:** dao động áp suất âm theo thời gian;
2. **spectrogram tần số tuyến tính:** bình phương độ lớn của Short-Time Fourier Transform (STFT);
3. **log-Mel spectrogram:** trục tần số được ánh xạ sang thang Mel phi tuyến của thính giác người bằng bộ lọc tam giác, sau đó nén dải động bằng logarit.

### B. Pipeline trích xuất đặc trưng lúc chạy

Pipeline phải tái lập được hoàn toàn và **không được tạo hay lưu sẵn ảnh spectrogram xuống ổ đĩa**:

```
file âm thanh thô (.wav)
  -> đọc và resample
  -> cắt / đệm về độ dài cố định (4–5 s)
  -> tính log-Mel spectrogram (64 hoặc 128 Mel bin)
  -> đầu vào mạng nơ-ron
```

Tham số mặc định đề xuất: độ dài 4–5 s; 64 hoặc 128 Mel bin; resample về 16 kHz hoặc 22,05 kHz.

### C. Ba mô hình bắt buộc

1. **CNN nhỏ** trên log-Mel spectrogram 2D.
2. **MobileNetV3-Small** [6] với trọng số pretrained ImageNet. Kênh spectrogram duy nhất được nhân bản ra 3 kênh RGB.
3. **CRNN** [7]: CNN trích đặc trưng, tiếp theo là GRU [8] mã hóa chuỗi theo thời gian.

### D. Metric bắt buộc

- Top-1 Accuracy và Macro-F1;
- F1 của từng lớp;
- confusion matrix;
- số tham số của mô hình và thời gian huấn luyện thực tế (wall-clock);
- độ phức tạp tính toán, đo bằng MACs/FLOPs hoặc độ trễ suy luận (ms mỗi clip).

### E. Câu hỏi giữa kỳ

1. Lớp nào đạt độ chính xác nhận dạng cao nhất và thấp nhất?
2. Những lớp nào bị nhầm lẫn với nhau nghiêm trọng nhất?
3. Mô hình hóa thời gian bằng recurrent trong CRNN có cải thiện đo được so với CNN 2D thuần trên các âm thanh có thời lượng khác nhau không?
4. Kiến trúc nào có cân bằng hiệu năng – chi phí tính toán tốt nhất?

### F. Sản phẩm nộp giữa kỳ

- slide thuyết trình;
- Jupyter notebook hoặc mã Python chạy được;
- hình và bảng tóm tắt phân tích dataset;
- bảng kết quả baseline;
- confusion matrix và phân tích định tính các lỗi trên âm thanh;
- câu hỏi nghiên cứu đề xuất cho cuối kỳ;
- README của dự án.

## 3. Cuối kỳ

### Câu hỏi nghiên cứu bắt buộc (RQ1)

> How robust are environmental sound classifiers when the target sound is perturbed by realistic acoustic noise, temporal distortions, and spectral corruptions?

Tạm dịch: các mô hình phân loại âm thanh môi trường bền vững đến mức nào khi âm thanh mục tiêu bị nhiễu âm học thực tế, méo thời gian và hỏng phổ?

Ràng buộc:
- dùng lại tập test sạch đã chốt ở giữa kỳ;
- giữ nguyên toàn bộ nhãn của tập test;
- chỉ áp dụng nhiễu có kiểm soát lúc đánh giá (test time).

### A. Các loại nhiễu bắt buộc

Chọn **ít nhất 3** trong các loại dưới đây, mỗi loại thử ở **3 mức độ**:

| Loại nhiễu | Ví dụ mức độ trong đề |
|---|---|
| Nhiễu nền cộng thêm ở các mức Signal-to-Noise Ratio (SNR) | +20 dB, +10 dB, 0 dB |
| Thay đổi gain / âm lượng | −12 dB, −6 dB, +6 dB |
| Dịch thời gian vòng (circular time shift) | — |
| Che thời gian (đặt về 0 các đoạn thời gian ngẫu nhiên) | — |
| Che tần số (đặt về 0 một số dải phổ) | — |
| Trộn âm thanh nền môi trường có kiểm soát (tiếng đường phố, tiếng mưa) | — |

### B. So sánh bắt buộc

Đánh giá **ít nhất 3 kiến trúc**. Với mỗi mô hình, báo cáo:

```
F1_clean,  F1_corrupted,  ΔF1 = F1_clean − F1_corrupted
```

Vẽ đường suy giảm hiệu năng theo mức độ nhiễu cho từng loại nhiễu.

### C. Phân tích lỗi bắt buộc

- mức suy giảm theo từng lớp dưới mỗi loại nhiễu;
- thay đổi của confusion matrix giữa tập test sạch và tập test có nhiễu;
- ví dụ âm thanh cụ thể mà mô hình chuyển từ đúng sang sai;
- thảo luận định tính, gắn đặc tính âm học (cấu trúc hài, xung tức thời, năng lượng dải rộng) với các kiểu lỗi quan sát được.

### D. Sản phẩm nộp cuối kỳ

- báo cáo kỹ thuật theo kiểu bài hội nghị;
- notebook hoặc mã Python tái lập được;
- file chỉ số split cố định;
- file config ghi tham số trích xuất đặc trưng;
- xác suất dự đoán đã lưu cho mỗi mô hình và điều kiện, dạng CSV với các cột:
  `sample_id, true_class, pred_class, probabilities, duration, model, split, perturbation, severity`;
- hình và bảng so sánh sinh ra bằng code;
- README hướng dẫn tái lập.

## 4. Phần mở rộng (tùy chọn)

Chỉ làm sau khi RQ1 đã thực hiện và kiểm chứng đầy đủ.

| Extension | Câu hỏi | Nội dung |
|---|---|---|
| 1 – SpecAugment | RQ2 | Dùng SpecAugment [9] khi huấn luyện (che thời gian và tần số ngẫu nhiên trực tiếp trên spectrogram) có làm giảm mức sụt hiệu năng từ sạch sang có nhiễu không [2], [5]. |
| 2 – Phân loại phân cấp | RQ3 | Tận dụng taxonomy 2 tầng (22 macro-class, 28 subclass) để so sánh phân loại phẳng 22 lớp với mô hình phân cấp hai giai đoạn hoặc đa nhiệm. |
| 3 – Pareto độ chính xác – hiệu suất | RQ4 | Đặt Macro-F1 cạnh số tham số, độ trễ suy luận, thời gian huấn luyện và VRAM GPU đỉnh; xác định các kiến trúc Pareto-efficient trên cả dữ liệu sạch và có nhiễu. |
| 4 – Từ DataSEC sang DataSED | RQ5 (paper track) | Đánh giá mô hình huấn luyện trên DataSEC trên bản ghi liên tục thực tế của DataSED [1], [10], dùng ranh giới sự kiện ground-truth. |

Extension 4 dùng ba giao thức cắt đoạn:
1. đúng ranh giới thời gian ground-truth của sự kiện;
2. đoạn sự kiện cộng thêm ±1 giây ngữ cảnh xung quanh;
3. cửa sổ cố định 4–5 s đặt giữa sự kiện mục tiêu.

Đây là **phân loại sự kiện với ranh giới thời gian cho sẵn (oracle)**, không phải phát hiện sự kiện đầu-cuối. Câu hỏi khoa học cốt lõi là: biểu diễn học từ các sự kiện tách riêng, đã tuyển chọn, có tổng quát hóa được sang cùng các lớp đó khi chúng nằm trong cảnh âm thanh liên tục, không kiểm soát hay không [3], [4]. DOI của DataSED: `10.5281/zenodo.15346092` [10].

## 5. Quy tắc tái lập

- Cố định đúng phiên bản dataset (Zenodo DOI `10.5281/zenodo.17033970`).
- Cố định ID của các phần train / validation / test.
- Không bao giờ chỉnh tham số data augmentation dựa trên kết quả trên tập test.
- Lưu phân phối xác suất liên tục cho mọi mô hình và mọi điều kiện thí nghiệm.
- Ghi lại tần số lấy mẫu, tham số cửa sổ STFT, random seed và phiên bản các thư viện Python.
- Bài nộp theo paper track phải chạy thí nghiệm với ít nhất 3 random seed khác nhau.

## 6. Tài liệu tham khảo trong đề

| # | Tài liệu | DOI |
|---|---|---|
| [1] | Fredianelli et al., "Environmental Noise Dataset for Sound Event Classification and Detection," *Scientific Data*, 2025 | 10.1038/s41597-025-05991-w |
| [2] | Akter et al., "A hybrid CNN-LSTM model for environmental sound classification…," *Digital Signal Processing*, 2025 | 10.1016/j.dsp.2025.105234 |
| [3] | Chen & Peng, "Environmental sound classification using two-stream deep neural network with interactive time-frequency attention," *Applied Acoustics*, 2025 | 10.1016/j.apacoust.2025.110794 |
| [4] | Tonami et al., "Trainingless Adaptation of Pretrained Models for Environmental Sound Classification," ICASSP 2025 | 10.1109/icassp49660.2025.10889784 |
| [5] | Xu et al., "Improving one-dimensional-based environmental sound classification models with progressive projection knowledge distillation…," *Eng. Appl. of AI*, 2025 | 10.1016/j.engappai.2025.110622 |
| [6] | Howard et al., "Searching for MobileNetV3," ICCV 2019 | 10.1109/ICCV.2019.00140 |
| [7] | Choi et al., "Convolutional recurrent neural networks for music classification," ICASSP 2017 | 10.1109/ICASSP.2017.7952585 |
| [8] | Cho et al., "Learning phrase representations using RNN encoder-decoder…," EMNLP 2014 | 10.3115/v1/D14-1179 |
| [9] | Park et al., "SpecAugment: A simple data augmentation method for automatic speech recognition," Interspeech 2019 | 10.21437/Interspeech.2019-2680 |
| [10] | Fredianelli et al., "DataSED: Curated environmental sound dataset for sound event detection," Zenodo, 2025 | 10.5281/zenodo.15346092 |

## 7. Ghi chú khi triển khai

Phần này là nhận xét khi đọc đề, không có trong đề bài.

- **ΔF1 dùng F1 nào.** Phần cuối kỳ chỉ ghi "F1", trong khi metric chính ở giữa kỳ là Macro-F1. Nên ghi rõ trong báo cáo là đang dùng Macro-F1.
- **Nhiễu gain có thể vô tác dụng.** Nếu pipeline chuẩn hóa biên độ hoặc chuẩn hóa spectrogram theo từng clip, việc tăng giảm gain sẽ bị triệt tiêu trước khi vào mô hình. Cần ghi rõ pipeline có chuẩn hóa hay không, và khi đó kết quả của nhiễu gain có ý nghĩa gì.
- **"Đặt về 0" trên log-Mel cần định nghĩa rõ.** Đặt năng lượng về 0 trước khi lấy log khác với đặt giá trị log-Mel về 0. Nên chọn một cách và ghi vào file config.
- **RQ2 trùng với nhiễu che.** SpecAugment che thời gian và tần số lúc huấn luyện, nên mô hình có SpecAugment gần như được "luyện trước" cho nhiễu che thời gian và che tần số lúc test. Nếu làm RQ2, nên tách kết quả trên loại nhiễu đã gặp khi huấn luyện (che) với loại chưa gặp (nhiễu cộng, trộn nền, gain).
- **Dataset trong máy.** Thư mục `DATASEC/` có 22 thư mục con, khớp với số macro-class. Chưa kiểm tra bản này có trùng với phiên bản Zenodo đã chốt hay không.
