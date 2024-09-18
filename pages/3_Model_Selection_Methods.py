import streamlit as st

# 设置页面标题
st.set_page_config(page_title="模型选择方法", page_icon="🤖")

# 页面标题
st.markdown("# 模型选择方法")

# 模型选择说明
st.write("选择一个模型以查看其简要介绍。")

# 模型列表及其描述
models = {
    "SVC": "SVC : 将实例映射到空间中的点，以最大化类之间的间隔。",
    "Bayes": "Bayes : 使用贝叶斯定理，通过各类的后验概率对点进行分类。",
    "MLP": "MLP : 由多层互联神经元组成的多层感知器。",
    "QDA": "QDA : 适用于分类问题的判别分析算法。",
    "AdaBoost": "AdaBoost : 使用弱分类器提升模型性能的元算法。",
    "Decision Tree": "Decision Tree : 根据特征将数据点分裂到不同的叶子节点。",
    "Random Forest": "Random Forest : 使用随机样本和特征的决策树集合。",
    "kNN": "kNN : 选择k个最近邻中的最常见类别作为分类结果。",
    "Rocket": "Rocket : 通过卷积核转换时间序列，生成特征以训练线性分类器。",
    "ConvNet": "ConvNet : 使用卷积层从输入数据中学习空间特征。",
    "ResNet": "ResNet : 使用残差连接的卷积神经网络。",
    "Inception Time": "Inception Time : 使用不同大小卷积核的ResNet组合。",
    "SiT-conv": "SiT-conv : 带有卷积层输入的transformer架构。",
    "SiT-linear": "SiT-linear : 将时间序列划分为不重叠的片段并线性投影到嵌入空间的transformer架构。",
    "SiT-stem": "SiT-stem : 使用卷积层进行输入处理并逐渐增加维度的transformer架构。",
    "SiT-stem-ReLU": "SiT-stem-ReLU : 类似于SiT-stem，但使用Scaled ReLU激活函数。",
}

# 用户选择模型
selected_model = st.selectbox("选择模型", options=list(models.keys()))

# 显示模型描述
st.write(f"**模型描述**: {models[selected_model]}")

