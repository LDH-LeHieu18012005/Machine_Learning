import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

st.title("Stroke Prediction")

# --- Chọn model từ 2 đường dẫn (bạn có thể chỉnh đường dẫn nếu cần) ---
model_options = {
    "svm_under2.joblib (model 1)": r"svm_under2.joblib",
    "svm_under3.joblib (model 2)": r"svm_under3.joblib"
}

selected_label = st.selectbox("Chọn mô-đun SVM", list(model_options.keys()))
model_path = model_options[selected_label]

st.write("Model path:", model_path)

# Thử load model — nhưng thực hiện khi người dùng ấn Predict để tránh lỗi khi khởi tạo
# ----------------------
# Nhập thông tin người dùng
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=120, value=45)
hypertension = st.selectbox("Hypertension", [0, 1], index=0)
heart_disease = st.selectbox("Heart Disease", [0, 1], index=0)
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", value=100.0, format="%.2f")
bmi = st.number_input("BMI", value=25.0, format="%.2f")
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Never_worked"])
smoking_status = st.selectbox("Smoking Status", ["never smoked", "smokes", "formerly smoked"])

# Tạo dataframe đầu vào trùng cột với mô hình
input_data = pd.DataFrame({
    'gender': [1 if gender == 'Male' else 0],
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'ever_married': [1 if ever_married == 'Yes' else 0],
    'Residence_type': [1 if Residence_type == 'Urban' else 0],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'work_type_Never_worked': [1 if work_type == 'Never_worked' else 0],
    'work_type_Private': [1 if work_type == 'Private' else 0],
    'work_type_Self-employed': [1 if work_type == 'Self-employed' else 0],
    'smoking_status_never smoked': [1 if smoking_status == 'never smoked' else 0],
    'smoking_status_smokes': [1 if smoking_status == 'smokes' else 0]
})

# Chuẩn hóa các cột số trực tiếp trong code
scaler = MinMaxScaler()
# Giả sử min-max dựa trên dataset chuẩn, ví dụ:
scaler.fit(pd.DataFrame({
    'age': [0, 120],
    'avg_glucose_level': [55, 300],  # phạm vi glucose hợp lý
    'bmi': [10, 60]                  # phạm vi BMI hợp lý
}))

columns_to_scale = ['age', 'avg_glucose_level', 'bmi']
input_data[columns_to_scale] = scaler.transform(input_data[columns_to_scale])

# Nút dự đoán
if st.button("Predict Stroke"):
    # Kiểm tra file model tồn tại không
    if not os.path.exists(model_path):
        st.error(f"Không tìm thấy file mô-đun tại: {model_path}")
    else:
        try:
            model = joblib.load(model_path)
        except Exception as e:
            st.error(f"Lỗi khi load mô-đun: {e}")
        else:
            try:
                prediction = model.predict(input_data)
            except Exception as e:
                st.error(f"Lỗi khi dự đoán (predict): {e}")
            else:
                # Lấy xác suất nếu có
                probability = None
                try:
                    if hasattr(model, "predict_proba"):
                        probability = model.predict_proba(input_data)[0][1]
                    else:
                        # Một số SVM không có predict_proba; nếu có decision_function thì chuyển sang khoảng 0..1 bằng sigmoid đơn giản
                        if hasattr(model, "decision_function"):
                            score = model.decision_function(input_data)[0]
                            # Chuyển đổi tỉ lệ score -> probability qua logistic sigmoid (chỉ để tham khảo)
                            import math
                            probability = 1 / (1 + math.exp(-score))
                except Exception:
                    probability = None  # nếu lỗi khi tính probability thì bỏ qua

                st.write("Prediction:", "Stroke" if int(prediction[0]) == 1 else "No Stroke")
                if probability is not None:
                    st.write("Probability (approx.):", round(float(probability) * 100, 2), "%")
                else:
                    st.info("Mô hình không cung cấp xác suất (predict_proba) hoặc không thể tính xác suất.")
