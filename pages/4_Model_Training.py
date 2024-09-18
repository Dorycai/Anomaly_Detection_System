import streamlit as st
import time
import numpy as np

st.set_page_config(page_title="模型训练", page_icon="🔧")

st.markdown("# 模型训练")

# 分类模型列表
models = [
    "SVC", "Bayes", "MLP", "QDA", "AdaBoost", "Decision Tree",
    "Random Forest", "kNN", "Rocket", "ConvNet", "ResNet",
    "Inception Time", "SiT-conv", "SiT-linear", "SiT-stem", "SiT-stem-ReLU"
]

# 左侧参数选择
st.sidebar.markdown("## 参数选择")

# 模型选择
selected_model = st.sidebar.selectbox("选择模型", options=models)

# Batch size 选择
batch_size = st.sidebar.slider("选择 Batch Size", 16, 512, 64)

# Epochs 选择
epochs = st.sidebar.slider("选择 Epochs", 1, 100, 20)

# Eval 选择
eval_option = st.sidebar.radio("选择是否进行 Eval", ("是", "否"))

# output_dim 选择
output_dim = st.sidebar.selectbox("选择 Output Dimension", options=[32, 64, 128, 256, 512])

# lambda_CL 选择
lambda_CL = st.sidebar.slider("选择 Lambda_CL", 0.0, 1.0, 0.5)

# Temperature 选择
temperature = st.sidebar.slider("选择 Temperature", 0.1, 10.0, 1.0)

# 开始训练按钮
start_training = st.sidebar.button("开始训练")

# 右侧显示训练过程
st.markdown("### 实时训练监控")

# 模拟训练过程中的 loss 和 accuracy 更新
if start_training:
    st.write(f"正在使用模型 {selected_model} 进行训练...")

    # 创建一个区域来显示 loss 和 accuracy
    progress_bar = st.progress(0)
    loss_chart = st.empty()  # 用于显示实时 loss
    acc_chart = st.empty()  # 用于显示实时 accuracy

    # 模拟训练过程，后期接train的后端算法
    for epoch in range(1, epochs + 1):
        loss = np.random.rand() * 0.1  # 模拟 loss 下降
        accuracy = np.random.rand() * 100  # 模拟 accuracy 增加

        # 更新进度条和 loss/accuracy 显示
        progress_bar.progress(epoch / epochs)
        loss_chart.write(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")
        acc_chart.write(f"Epoch {epoch}/{epochs} - Accuracy: {accuracy:.2f}%")

        time.sleep(0.5)  # 模拟每个 epoch 的训练时间

    st.success(f"训练完成！模型 {selected_model} 的训练结束。")
