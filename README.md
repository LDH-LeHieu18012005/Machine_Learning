# 🧠 Stroke Prediction Project - Machine Learning

Dự án sử dụng các thuật toán Học máy (Machine Learning) để dự đoán nguy cơ đột quỵ dựa trên dữ liệu sức khỏe của bệnh nhân.

---

## 📝 1. Giới thiệu đề tài
Theo **Tổ chức Y tế Thế giới (WHO)**, đột quỵ là nguyên nhân gây tử vong đứng thứ ba trên toàn cầu. Dự án này nhằm xây dựng một mô hình hỗ trợ chẩn đoán sớm, giúp giảm thiểu tỷ lệ tử vong và tàn tật.

* **Mục tiêu:** Phát triển mô hình dự đoán chính xác khả năng bị đột quỵ.
* **Thách thức:** Xử lý tập dữ liệu mất cân bằng nghiêm trọng (Imbalanced Data).

---

## 📊 2. Chi tiết dữ liệu (Dataset)
* **Tổng số bản ghi:** 43,400
* **Số lượng thuộc tính:** 12
* **Tỷ lệ nhãn bệnh:** 1.8% (Dữ liệu mất cân bằng)
* **Nguồn:** [Kaggle Dataset](https://www.kaggle.com/datasets/shashwatwork/cerebral-stroke-predictionimbalaced-dataset)

### Danh sách các thuộc tính:
| Thuộc tính | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- |
| `gender` | Object | Giới tính bệnh nhân |
| `age` | Float | Tuổi |
| `hypertension` | Int | Cao huyết áp (0/1) |
| `heart_disease` | Int | Bệnh tim mạch (0/1) |
| `avg_glucose_level`| Float | Mức đường huyết trung bình |
| `bmi` | Float | Chỉ số khối cơ thể |
| `smoking_status` | Object | Tình trạng hút thuốc |

---

## ⚙️ 3. Pipeline dự án
Dự án được triển khai qua các bước chuẩn hóa sau:

1.  **Tiền xử lý (Preprocessing):**
    * Xử lý giá trị thiếu (Missing values) cho `smoking_status`.
    * Mã hóa biến phân loại (One-hot Encoding).
    * Cân bằng dữ liệu bằng phương pháp **SMOTE / Sampling**.
2.  **Huấn luyện (Training):** Chia tập dữ liệu 75% Train / 25% Test.
3.  **Đánh giá (Evaluation):** Sử dụng Confusion Matrix, Recall, F1-score để đo lường hiệu quả.

---

## 🤖 4. Các mô hình sử dụng
Chúng tôi thử nghiệm và so sánh 3 thuật toán phổ biến:
* **Logistic Regression (LR):** Hiệu quả trong việc tính toán xác suất cơ bản.
* **Random Forest (RF):** Khả năng xử lý tốt các mối quan hệ phi tuyến.
* **Support Vector Machine (SVM):** Tìm kiếm ranh giới phân tách (Hyperplane) tối ưu.

---

## 📈 5. Kết quả
Mô hình tập trung tối ưu hóa chỉ số **Recall** để không bỏ sót các trường hợp bệnh thực tế.
* **Recall:** `90.31%`
* **F1-score:** `14.19`

> Kết quả trực quan tại: `reports`

---

## 📂 6. Cấu trúc thư mục
```text
Machine_Learning/
├── app/              # Mã nguồn huấn luyện & xử lý dữ liệu
├── demo/             # File chạy demo nhanh & Notebooks
├── data/             # Tập dữ liệu CSV
├── reports/          # Báo cáo Word & Hình ảnh kết quả
├── slides/           # Slide thuyết trình (PDF)
├── requirements.txt  # Danh sách thư viện cần cài đặt
└── README.md         # Tài liệu hướng dẫn
## 📂 7. Tác giả
Họ tên: Lê Dương Hiếu

Mã SV: 12423011

Lớp: 124231
