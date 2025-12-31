# DỰ ĐOÁN ĐỘT QUỴ DỰA TRÊN DỮ LIỆU BỆNH NHÂN BẰNG MACHINE LEARNING

## 1. GIỚI THIỆU ĐỀ TÀI
Dự án tập trung vào việc ứng dụng Machine Learning để dự đoán nguy cơ đột quỵ dựa trên dữ liệu y tế của bệnh nhân. Theo Tổ chức Y tế Thế giới (WHO), đột quỵ là một trong những nguyên nhân gây tử vong hàng đầu trên toàn cầu. Việc phát hiện sớm nguy cơ đột quỵ có thể giúp giảm đáng kể tỷ lệ tử vong và các biến chứng nguy hiểm.

- Mục tiêu: Xây dựng mô hình phân loại nhị phân dự đoán nguy cơ đột quỵ.
- Định hướng: Ưu tiên Recall cao để hạn chế bỏ sót các ca bệnh.
- Thách thức: Dữ liệu mất cân bằng nghiêm trọng, nhãn đột quỵ chỉ chiếm khoảng 1.8%.

## 2. CHI TIẾT DỮ LIỆU (DATASET)
- Số lượng bản ghi: 43,400
- Số cột: 12
- Tỷ lệ nhãn đột quỵ: 1.8%
- Nguồn dữ liệu: https://www.kaggle.com/datasets/shashwatwork/cerebral-stroke-predictionimbalaced-dataset

### Mô tả các thuộc tính
- id (int64): Mã định danh bệnh nhân
- gender (object): Giới tính
- age (float64): Tuổi
- hypertension (int64): Cao huyết áp
- heart_disease (int64): Bệnh tim
- ever_married (object): Tình trạng hôn nhân
- work_type (object): Loại công việc
- Residence_type (object): Khu vực sinh sống
- avg_glucose_level (float64): Mức đường huyết trung bình
- bmi (float64): Chỉ số BMI
- smoking_status (object): Tình trạng hút thuốc
- stroke (int64): Nhãn đột quỵ (0: Không, 1: Có)

## 3. QUY TRÌNH THỰC HIỆN (PIPELINE)
Toàn bộ pipeline được triển khai trong notebook: demo/demo.ipynb.

3.1. Chuẩn bị dữ liệu

Đọc file dữ liệu: data/Dotquy.csv

Target: stroke

Làm sạch nhãn:

Ép kiểu numeric (to_numeric) cho các cột số.

Thay inf / -inf → NaN.

Loại bỏ dòng có target hoặc feature không hợp lệ.

3.2. Chia train/test

Chia dữ liệu: 75% train, 25% test (train_test_split(test_size=0.25, stratify=y, random_state=42))

Giữ tỷ lệ lớp để xử lý imbalance.

3.3. Tiền xử lý (Preprocess) bằng ColumnTransformer

Numeric columns: age, avg_glucose_level, bmi

SimpleImputer(strategy="median")

Minmax()

Categorical columns: gender, ever_married, work_type, Residence_type, smoking_status

SimpleImputer(strategy="most_frequent")

OneHotEncoder(handle_unknown="ignore")

3.4. Huấn luyện (Train) + xử lý mất cân bằng (Imbalance)

Mỗi model được gắn vào pipeline: preprocess → (tùy chọn) RandomOverSampler → model

Các model hỗ trợ class_weight: LogisticRegression, SVM, RandomForestClassifier

Ưu tiên class_weight="balanced" để xử lý imbalance.

3.5. Đánh giá (Evaluate)

Đánh giá trên TRAIN với StratifiedKFold(n_splits=5) để giảm bias và tránh leakage.

Metrics sử dụng:

Confusion Matrix

Accuracy, Precision, Recall, F1-score


Threshold dự đoán:

Baseline: chọn threshold tối ưu F1 dựa trên Precision–Recall curve

Sau tuning: chọn threshold ưu tiên Recall với ràng buộc Precision tối thiểu (mặc định 0.15)

3.6. Tuning (RandomizedSearchCV)

Sau khi chạy baseline, notebook tuning một số model mạnh:

Random Forest (RF)

Support Vector Machine (SVM)

Logistic Regression (LR)

Chọn tham số để tối ưu metric cho dữ liệu mất cân bằng.

3.7. Inference


Demo inference chạy qua: demo/app.py

### Kết quả tiêu biểu
- Recall: 90.31%
- F1-score: 14.19%

## 4. CẤU TRÚC THƯ MỤC

<h2>Cấu trúc thư mục dự án</h2>

<pre>
Machine_Learning/
├── app/                 # Mã nguồn chính
├── demo/                # Demo inference & notebook
├── data/                # Dữ liệu
├── reports/             # Báo cáo & hình ảnh
├── slides/              # Slide thuyết trình
├── requirements.txt     # Thư viện Python
├── .gitignore           # File loại trừ
└── README.md            # Tài liệu dự án
</pre>



## 5. HƯỚNG DẪN CÀI ĐẶT & THỨ TỰ THỰC THI

### Bước 1: Cài đặt môi trường
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

### Bước 2: Chạy dự án theo đúng thứ tự
python app/Cerebral_stroke_prediction.py
- Đọc dữ liệu
- Tiền xử lý + SMOTE
- Huấn luyện mô hình
- Xuất kết quả vào thư mục reports/

python demo/demo_inference.py
- Dự đoán nguy cơ đột quỵ cho bệnh nhân mới

jupyter notebook demo/app.py
- (Tùy chọn) Trực quan hóa dữ liệu và giải thích chi tiết
## 6. KẾT QUẢ VÀ PHÂN TÍCH

### 6.1 Bảng kết quả chi tiết
Dưới đây là so sánh hiệu suất của các mô hình dựa trên các phương pháp lấy mẫu (Sampling) khác nhau:

| Mô hình | Sampling | Accuracy | Prec (0) | Prec (1) | Recall (0) | Recall (1) | F1 (0) | F1 (1) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Under | 0.7869 | 0.9965 | 0.0809 | 0.7850 | 0.8724 | 0.8782 | 0.1481 |
| **Logistic Regression** | Over | 0.7825 | 0.9966 | 0.0798 | 0.7805 | 0.8776 | 0.8754 | 0.1463 |
| **SVM** | **Under** | **0.7682** | **0.9973** | **0.0770** | **0.7653** | **0.9031** | **0.8660** | **0.1419** |
| **SVM** | Over | 0.7754 | 0.9964 | 0.0770 | 0.7733 | 0.8724 | 0.8708 | 0.1416 |
| **Random Forest** | Under | 0.7953 | 0.9971 | 0.0856 | 0.7932 | 0.8929 | 0.8835 | 0.1562 |
| **Random Forest** | Over | 0.8885 | 0.9912 | 0.1146 | 0.8940 | 0.6327 | 0.9401 | 0.1941 |

### 6.2 Phân tích kết quả
* **Mô hình tối ưu:** **SVM (Under-sampling)** được chọn là mô hình tốt nhất cho bài toán này vì đạt chỉ số **Recall (1) lên tới 90.31%**.
* **Tầm quan trọng của Recall:** Trong chẩn đoán y khoa, việc "bỏ sót" một bệnh nhân có nguy cơ đột quỵ (False Negative) nguy hiểm hơn nhiều so với việc chẩn đoán nhầm người khỏe mạnh (False Positive). Do đó, chỉ số Recall cho nhãn bệnh (1) là ưu tiên hàng đầu.
* **Đánh giá F1-score:** Mặc dù F1-score ở nhãn (1) thấp (0.1419) do dữ liệu cực kỳ mất cân bằng, nhưng mô hình vẫn đảm bảo được khả năng sàng lọc rất tốt (Recall > 90%).
## 7. THÔNG TIN TÁC GIẢ
- Họ tên: Lê Dương Hiếu
- Mã sinh viên: 12423011
- Lớp: 124231

## GHI CHÚ
Dự án phục vụ mục đích học tập và nghiên cứu, không thay thế cho chẩn đoán y khoa chuyên môn.
