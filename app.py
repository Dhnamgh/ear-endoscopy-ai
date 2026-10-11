import io
import os
import numpy as np
import pandas as pd
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

# Tiêu đề tùy chỉnh kiểu dáng (Font Times New Roman, cỡ chữ 15pt tương đương 20px, màu Xanh Facebook)
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


# 1. Định nghĩa mô hình mẫu (thay thế bằng kiến trúc thực tế của bạn)
class EarMultiTaskModel(nn.Module):

  def __init__(self):
    super(EarMultiTaskModel, self).__init__()
    self.base_model = torch.hub.load(
        "pytorch/vision:v0.10.0", "resnet18", pretrained=False
    )
    # Nhánh phân loại (Classification head)
    self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 1)

  def forward(self, x):
    logits = self.base_model(x)
    return logits


# 2. Hàm tải mô hình (được cache để tối ưu tốc độ)
@st.cache_resource
def load_model():
  model = EarMultiTaskModel()
  # model.load_state_dict(torch.load("path_to_weights.pth", map_location=torch.device('cpu')))
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
  st.subheader("Kết quả phân tích hàng loạt")

  results_data = []
  progress_bar = st.progress(0)
  total_files = len(uploaded_files)

  for i, uploaded_file in enumerate(uploaded_files):
    image = Image.open(uploaded_file).convert("RGB")

    # Xử lý tensor
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
      output = model(input_tensor)
      latent_val = output.item()  # Điểm số thô (logit)
      prob_val = torch.sigmoid(output).item()  # Xác suất (0 đến 1)
      pred_status_val = 1 if prob_val >= 0.5 else 0

    results_data.append({
        "filename": uploaded_file.name,
        "true_status": 1,  # Gán mặc định hoặc cập nhật theo nhãn thực tế nếu có
        "latent": latent_val,
        "pred_prob": prob_val,
        "pred_status": pred_status_val,
    })

    # Hiển thị trực quan từng ảnh và kết quả
    col1, col2 = st.columns([1, 2])
    with col1:
      st.image(image, caption=uploaded_file.name, width=150)
    with col2:
      st.write(f"**Tên file:** {uploaded_file.name}")
      st.write(f"**Điểm số thô (Latent/Logit):** {latent_val:.4f}")
      st.write(f"**Xác suất dự đoán (Pred Prob):** {prob_val * 100:.2f}%")
      if pred_status_val == 1:
        st.error("Kết quả: Bất thường (Abnormal)")
      else:
        st.success("Kết quả: Bình thường (Normal)")
    st.divider()

    progress_bar.progress((i + 1) / total_files)

  # 5. Tổng hợp thành DataFrame và cung cấp nút tải file CSV mang sang Stata
  df_results = pd.DataFrame(results_data)

  st.subheader("Bảng dữ liệu tổng hợp sẵn sàng phân tích thống kê")
  st.dataframe(df_results)

  csv_data = df_results.to_csv(index=False).encode("utf-8")

  st.download_button(
      label="Tải xuống tệp CSV để phân tích trên Stata",
      data=csv_data,
      file_name="ket_qua_suy_luan_ai.csv",
      mime="text/csv",
  )
