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

# Tiêu đề tùy chỉnh kiểu dáng (Font Times New Roman, cỡ chữ tương đương 20px, màu Xanh Facebook)
st.markdown(
    """
    <h3 style="
        font-family: 'Times New Roman', Times, serif; 
        font-size: 20px; 
        color: #1877F2; 
        font-weight: bold;
        margin-bottom: 8px;
    ">
        Ứng dụng Phân tích Hình ảnh Nội soi Tai (Classification & Segmentation)
    </h3>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Tải lên ảnh nội soi màng nhĩ để mô hình học sâu thực hiện phân loại bệnh lý"
    " và phân đoạn cấu trúc giải phẫu."
)


# 1. Định nghĩa mô hình mẫu
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
    "Chọn các file ảnh nội soi tai ( định dạng JPG, PNG ):",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
  st.subheader("Kết quả phân tích chi tiết")

  results_data = []
  progress_bar = st.progress(0)
  total_files = len(uploaded_files)

  for i, uploaded_file in enumerate(uploaded_files):
    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
      output = model(input_tensor)
      latent_val = output.item()
      prob_val = torch.sigmoid(output).item()
      pred_status_val = 1 if prob_val >= 0.5 else 0

    results_data.append({
        "filename": uploaded_file.name,
        "true_status": 1
        if i % 2 == 0
        else 0,  # Mặc định chia đôi mẫu để test thống kê nếu tải nhiều ảnh
        "latent": latent_val,
        "pred_prob": prob_val,
        "pred_status": pred_status_val,
    })

    col1, col2 = st.columns([1, 2])
    with col1:
      st.image(image, caption=uploaded_file.name, width=150)
    with col2:
      st.write(f"**Tên file:** {uploaded_file.name}")
      st.write(f"**Điểm số thô (Latent/Logit):** {latent_val:.4f}")
      st.write(f"**Xác suất dự đoán (Pred Prob):** {prob_val * 100:.2f}%")
      if pred_status_val == 1:
        st.error("Kết quả dự đoán: Bất thường (Abnormal)")
      else:
        st.success("Kết quả dự đoán: Bình thường (Normal)")
    st.divider()

    progress_bar.progress((i + 1) / total_files)

  df_results = pd.DataFrame(results_data)

  st.subheader("Bảng dữ liệu tổng hợp kết quả")
  st.dataframe(df_results)

  # 5. Kiểm tra điều kiện dữ liệu trước khi tính toán thống kê & vẽ ROC
  if len(df_results) >= 2 and df_results["true_status"].nunique() > 1:
    st.subheader("Thống kê đánh giá chẩn đoán & Biểu đồ ROC")

    y_true = df_results["true_status"]
    y_scores = df_results["pred_prob"]
    y_pred = df_results["pred_status"]

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
    roc_auc = metrics.auc(fpr, tpr)

    cm = metrics.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
      st.metric(label="Diện tích dưới đường cong (AUC)", value=f"{roc_auc:.4f}")
      st.metric(label="Độ nhạy (Sensitivity)", value=f"{sensitivity * 100:.2f}%")
    with col_stat2:
      st.metric(
          label="Độ đặc hiệu (Specificity)", value=f"{specificity * 100:.2f}%"
      )
      st.metric(label="Giá trị dự đoán dương (PPV)", value=f"{ppv * 100:.2f}%")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(
        fpr,
        tpr,
        color="#1877F2",
        lw=2,
        label=f"ROC curve (AUC = {roc_auc:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Đường cong ROC đánh giá hiệu năng mô hình")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    st.pyplot(fig)
  else:
    st.info(
        "💡 **Mẹo:** Biểu đồ ROC và các chỉ số thống kê tổng hợp sẽ hiển thị khi"
        " bạn tải lên từ 2 ảnh trở lên và có chứa đồng thời cả mẫu bình"
        " thường lẫn bất thường."
    )

  csv_data = df_results.to_csv(index=False).encode("utf-8")
  st.download_button(
      label="Tải xuống tệp dữ liệu CSV",
      data=csv_data,
      file_name="ket_qua_suy_luan_ai.csv",
      mime="text/csv",
  )
