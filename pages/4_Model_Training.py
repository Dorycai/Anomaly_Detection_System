import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time

# 设置页面标题
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

# 创建两个列区域
col1, col2 = st.columns(2)

# 创建显示 loss 和 accuracy 曲线的区域
loss_placeholder = col1.empty()
acc_placeholder = col2.empty()

# 初始化 loss 和 accuracy 曲线数据
loss_values = []
accuracy_values = []

# 创建显示 loss 和 accuracy 数值的区域
loss_chart = col1.empty()
acc_chart = col2.empty()

# 模拟训练过程
if start_training:
    st.write(f"正在使用模型 {selected_model} 进行训练...")

    # 训练过程
    for epoch in range(1, epochs + 1):
        loss = np.random.rand() * 0.1  # 模拟 loss 下降
        accuracy = np.random.rand() * 100  # 模拟 accuracy 增加

        # 更新 loss 和 accuracy 曲线数据
        loss_values.append(loss)
        accuracy_values.append(accuracy)

        # 绘制 loss 曲线
        plt.figure(figsize=(6, 4))
        plt.cla()  # 清除上一图像
        plt.plot(range(1, len(loss_values) + 1), loss_values, color='blue', label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        loss_placeholder.pyplot(plt)  # 在左侧动态显示 loss 曲线

        # 绘制 accuracy 曲线
        plt.figure(figsize=(6, 4))
        plt.cla()  # 清除上一图像
        plt.plot(range(1, len(accuracy_values) + 1), accuracy_values, color='green', label='Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training Accuracy Curve')
        plt.legend()
        acc_placeholder.pyplot(plt)  # 在右侧动态显示 accuracy 曲线

        # 更新 loss 和 accuracy 信息显示
        loss_chart.write(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")
        acc_chart.write(f"Epoch {epoch}/{epochs} - Accuracy: {accuracy:.2f}%")

        time.sleep(0.5)  # 模拟每个 epoch 的训练时间

    st.success(f"训练完成！模型 {selected_model} 的训练结束。")
