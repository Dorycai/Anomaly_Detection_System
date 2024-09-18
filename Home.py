import streamlit as st

# 设置页面标题和图标
st.set_page_config(page_title="时序异常检测系统", page_icon="📊")

# 页面欢迎语
st.write("# 欢迎使用时序异常检测系统 👋")

# 侧边栏提示
st.sidebar.success("选择一个功能以开始操作。")

# 主要功能模块介绍
st.markdown("""
    本系统包含以下模块：
    - 📊 **数据集管理**：上传、预览和管理数据集。
    - 🔍 **异常检测方法**：了解并选择适用的异常检测算法。
    - 🧠 **模型选择方法**：为您的时序数据选择最佳模型。
    - 🏋️ **模型训练**：通过自定义参数训练模型，并实时查看训练进度。
    - 🔮 **推理与预测**：使用已训练好的模型进行数据预测。
    - 📚 **知识融合**：通过融合外部知识增强模型能力。

    请从左侧边栏选择相应功能。
""")
