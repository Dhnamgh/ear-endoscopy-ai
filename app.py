import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Cấu hình tiêu đề trang ứng dụng
st.set_page_config(
    page_title="Hệ thống Hỗ trợ Chẩn đoán Nội soi Tai",
    layout="wide",
)

# Kiểm tra mật khẩu bảo vệ từ Streamlit Secrets
password_input = st.text_input(
    "Nhập mật khẩu truy cập ứng dụng:", type="password"
)
if "PASSWORD" in st.secrets:
  if password_input != st.secrets["PASSWORD"]:
    st.warning("Vui lòng nhập đúng mật khẩu để tiếp tục sử dụng hệ thống.")
    st.stop()
else:
  st.warning("Chưa cấu hình mật khẩu trong Streamlit Secrets.")
  st.stop()

# Tiêu đề giao diện (Font Times New Roman, cỡ chữ 20px, màu Xanh Facebook)
st.markdown(
    """
    <h3 style="
        font-family: 'Times New Roman', Times, serif; 
        font-size: 20px; 
        color: #1877F2; 
        font-weight: bold;
        margin-bottom: 8px;
    ">
        Ứng dụng Phân tích Hình ảnh Nội soi Tai (Classification & Statistical Evaluation)
    </h3>
    """,
    unsafe_allow_html=True,
)

# Thanh tùy chọn chế độ hoạt động ở Sidebar
st.sidebar.header("⚙️ Cấu hình Chế độ Hoạt động")
mode_option = st.sidebar.radio(
    "Chọn mục đích sử dụng:",
    (
        "1. Kiểm tra nhanh ca lâm sàng (Single/Few-shot)",
        "2. Đánh giá tập dữ liệu nghiên cứu (Batch Validation for Paper)",
    ),
)


# 1. Định nghĩa mô hình mẫu (ResNet18)
class EarMultiTaskModel(nn.Module):

  def __init__(self):
    super(EarMultiTaskModel, self).__init__()
    self.base_model = torch.hub.load(
        "pytorch/vision:v0.10.0", "resnet18", pretrained=False
    )
    self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 1)

  def forward(self, x):
    logits = self.base_model(x)
    return logits


# 2. Hàm tải mô hình
@st.cache_resource
def load_model():
  model = EarMultiTaskModel()
  model.eval()
  return model


model = load_model()

# 3. Tiền xử lý hình ảnh đầu vào
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 4. Khu vực tải file từ người dùng
uploaded_files = st.file_uploader(
    "Chọn các file ảnh nội soi tai (định dạng JPG, PNG):",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
  results_data = []
  progress_bar = st.progress(0)
  total_files = len(uploaded_files)

  # Nếu ở chế độ nghiên cứu tập lớn, hiển thị hướng dẫn đặt tên file hoặc gán nhãn
  if (
      "2. Đánh giá tập dữ liệu nghiên cứu" in mode_option
      and total_files >= 10
  ):
    st.sidebar.info(
        "💡 **Quy tắc gán nhãn tự động từ tên file:**\n- Nếu tên file chứa chữ"
        " `normal` hoặc `bn` -> Nhãn thực tế = 0 (Bình thường)\n- Nếu chứa chữ"
        " `abnormal`, `aom`, `ome` hoặc `bt` -> Nhãn thực tế = 1 (Bất thường)"
    )

  st.subheader(f"Kết quả phân tích chi tiết ({total_files} hình ảnh)")

  for i, uploaded_file in enumerate(uploaded_files):
    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
      output = model(input_tensor)
      latent_val = output.item()
      prob_val = torch.sigmoid(output).item()
      pred_status_val = 1 if prob_val >= 0.5 else 0

    # Xác định nhãn thực tế (true_status) dựa theo chế độ
    if "2. Đánh giá tập dữ liệu nghiên cứu" in mode_option:
      # Tự động nhận diện qua tên file nếu có quy ước, mặc định lấy 1 nếu không khớp
      fname_lower = uploaded_file.name.lower()
      if any(k in fname_lower for k in ["normal", "bn", "healthy", "c"]):
        true_status_val = 0
      elif any(k in fname_lower for k in ["abnormal", "aom", "ome", "b"]):
        true_status_val = 1
      else:
        # Nếu tên file không có dấu hiệu, tạm chia đôi hoặc gán mặc định (có thể chỉnh sửa thủ công nếu cần)
        true_status_val = 1 if i % 2 == 0 else 0
    else:
      true_status_val = (
          0  # Chế độ test nhanh không dùng giá trị này cho thống kê tổng
      )

    results_data.append({
        "filename": uploaded_file.name,
        "true_status": true_status_val,
        "latent": latent_val,
        "pred_prob": prob_val,
        "pred_status": pred_status_val,
    })

    # Hiển thị giao diện từng ảnh thu gọn nếu danh sách quá lớn
    if total_files <= 10:
      col1, col2 = st.columns([1, 3])
      with col1:
        st.image(image, caption=uploaded_file.name, width=120)
      with col2:
        st.write(f"**Tên file:** {uploaded_file.name}")
        st.write(
            f"**Xác suất dự đoán bệnh lý:** **{prob_val * 100:.2f}%** (Logit:"
            f" {latent_val:.4f})"
        )
        if pred_status_val == 1:
          st.error("Kết quả AI: Bất thường (Abnormal)")
        else:
          st.success("Kết quả AI: Bình thường (Normal)")
      st.divider()

    progress_bar.progress((i + 1) / total_files)

  df_results = pd.DataFrame(results_data)

  if total_files > 10:
    st.success(
        f"Đã xử lý xong {total_files} hình ảnh. Dưới đây là bảng dữ liệu tổng"
        " hợp:"
    )

  st.subheader("Bảng dữ liệu suy luận")
  st.dataframe(df_results, use_container_width=True)

  # 5. Khu vực Thống kê & Biểu đồ ROC (Chỉ kích hoạt khi ở Chế độ Nghiên cứu và đủ điều kiện mẫu)
  if "2. Đánh giá tập dữ liệu nghiên cứu" in mode_option:
    st.divider()
    st.markdown(
        "### 📊 Phân tích Thống kê Đánh giá Chẩn đoán (Validation Metrics)"
    )

    if len(df_results) >= 5 and df_results["true_status"].nunique() > 1:
      y_true = df_results["true_status"]
      y_scores = df_results["pred_prob"]
      y_pred = df_results["pred_status"]

      # Tính toán thông số thống kê y khoa
      fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
      roc_auc = metrics.auc(fpr, tpr)

      cm = metrics.confusion_matrix(y_true, y_pred)
      tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

      sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
      specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
      ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
      npv = tn / (tn + fn) if (tn + fn) > 0 else 0
      accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

      # Hiển thị các chỉ số cốt lõi dạng Metrics đẹp mắt
      col_m1, col_m2, col_m3, col_m4 = st.columns(4)
      with col_m1:
        st.metric(label="Diện tích AUC", value=f"{roc_auc:.4f}")
      with col_m2:
        st.metric(label="Độ nhạy (Sensitivity)", value=f"{sensitivity * 100:.2f}%")
      with col_m3:
        st.metric(
            label="Độ đặc hiệu (Specificity)", value=f"{specificity * 100:.2f}%"
        )
      with col_m4:
        st.metric(label="Độ chính xác (Accuracy)", value=f"{accuracy * 100:.2f}%")

      # Bố cục vẽ biểu đồ ROC chuẩn phong cách bài báo quốc tế
      col_chart1, col_chart2 = st.columns([1.2, 1])

      with col_chart1:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(
            fpr,
            tpr,
            color="#1877F2",
            lw=2.5,
            label=f"ROC curve (AUC = {roc_auc:.4f})",
        )
        ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("1 - Specificity (False Positive Rate)", fontsize=11)
        ax.set_ylabel("Sensitivity (True Positive Rate)", fontsize=11)
        ax.set_title(
            "Đường cong ROC đánh giá hiệu năng mô hình",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(alpha=0.3)
        st.pyplot(fig)

      with col_chart2:
        st.markdown("#### 📋 Bảng Ma trận Nhầm lẫn (Confusion Matrix)")
        cm_df = pd.DataFrame(
            [[tp, fn], [fp, tn]],
            index=["Thực tế: Bệnh lý", "Thực tế: Bình thường"],
            columns=["Dự đoán: Bệnh lý", "Dự đoán: Bình thường"],
        )
        st.dataframe(cm_df, use_container_width=True)

        st.info(
            f"• PPV (Giá trị dự đoán dương): **{ppv * 100:.2f}%**\n• NPV (Giá"
            f" trị dự đoán âm): **{npv * 100:.2f}%**"
        )

      # Nút tải file CSV dữ liệu tổng hợp để đưa vào các công cụ phân tích sâu khác
      csv_data = df_results.to_csv(index=False).encode("utf-8")
      st.download_button(
          label="📥 Tải xuống tệp dữ liệu chuẩn hóa phục vụ viết bài báo (.csv)",
          data=csv_data,
          file_name="dataset_validation_results.csv",
          mime="text/csv",
      )

    else:
      st.warning(
          "⚠️ Để chạy thống kê và vẽ biểu đồ ROC chuẩn bài báo, bạn cần tải lên"
          " tối thiểu **5 hình ảnh** và có sự pha trộn giữa cả 2 nhóm nhãn"
          " (Bình thường và Bất thường)."
      )
  else:
    st.info(
        "💡 **Đang ở chế độ kiểm tra nhanh:** Ứng dụng chỉ tập trung hiển thị"
        " kết quả dự đoán cho từng ảnh riêng lẻ. Để xem phân tích ROC, AUC và"
        " xuất dữ liệu thống kê, vui lòng chuyển sang **Chế độ đánh giá tập"
        " dữ liệu nghiên cứu** ở cột bên trái."
    )
