import streamlit as st

# 设置页面标题
st.set_page_config(page_title="异常检测方法", page_icon="🔍")

# 页面标题
st.markdown("# 异常检测方法")

# 十二种异常检测方法
algorithms = {
    "Isolation Forest": "IForest 构建了基于随机空间分割的二叉树。子序列越接近根节点，越有可能是异常。",
    "IForest1": "IForest1 是每个点（单独）作为输入进行处理。",
    "Local Outlier Factor (LOF)": "LOF 计算邻域密度与局部密度的比率，检测异常点。",
    "Matrix Profile (MP)": "MP 通过最大最近邻距离检测异常子序列。",
    "NormA": "NormA 使用聚类方法识别正常模式，并通过统计标准计算子序列的加权距离。",
    "PCA": "PCA 将数据投影到低维超平面，偏离该平面的点被视为异常点。",
    "Autoencoder (AE)": "AE 将数据投影到低维空间并重构，重构误差较大的点被视为异常点。",
    "LSTM-AD": "LSTM-AD 通过LSTM网络预测当前子序列的下一个值，预测误差用于识别异常。",
    "Polynomial Fit (POLY)": "POLY 拟合多项式模型，使用预测误差检测异常。",
    "CNN": "CNN 构建了当前和之前子序列之间的相关性，异常得分为预测偏差。",
    "OCSVM": "OCSVM 是一种支持向量方法，拟合正常训练数据集并找到正常数据的边界。",
    "HBOS": "HBOS 为时间序列构建直方图，异常得分为直方图高度的倒数。"
}

# 用户选择算法
selected_algorithm = st.selectbox("选择异常检测算法", options=list(algorithms.keys()))

# 显示算法描述
st.write(f"**算法描述**: {algorithms[selected_algorithm]}")

# 参数选择部分
if selected_algorithm == "Isolation Forest":
    contamination = st.slider("污染率 (contamination)", 0.01, 0.5, 0.1)
    n_estimators = st.number_input("树的数量 (n_estimators)", min_value=100, max_value=1000, value=100)
    st.write(f"已选择：污染率={contamination}, 树的数量={n_estimators}")
elif selected_algorithm == "LSTM-AD":
    learning_rate = st.slider("学习率 (learning rate)", 0.0001, 0.01, 0.001)
    hidden_layers = st.number_input("隐藏层数量 (hidden layers)", min_value=1, max_value=10, value=2)
    st.write(f"已选择：学习率={learning_rate}, 隐藏层数量={hidden_layers}")
# 先看看这里设计合不合理，在考虑增加其他方法的参数调整

# 选择“我的数据集”
st.write("### 选择数据集")
uploaded_datasets = ["我的数据集 1", "我的数据集 2", "我的数据集 3"]  # 与之前上传的数据集保持一致
selected_dataset = st.selectbox("选择一个数据集", options=uploaded_datasets)

# “开始检测”按钮
if st.button("开始检测"):
    st.write(f"正在使用 {selected_algorithm} 对 {selected_dataset} 进行异常检测...")
    # 在这里调用后端异常检测逻辑
