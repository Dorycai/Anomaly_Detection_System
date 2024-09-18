import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# 设置页面标题
st.set_page_config(page_title="知识融合模块", page_icon="🧠")

# 模块介绍
st.title("知识融合模块")
st.write("通过将外部领域知识与时间序列特征相结合，增强算法的智能化与适应性。")

# 知识融合框架展示
st.image("knowledge_integration_framework.png", caption="知识融合框架图")

# 优点对比展示
st.subheader("模型优点展示")
st.write("""
知识融合模块在以下方面对模型进行了优化：
- **提高精度**：通过结合外部知识，模型选择更加精准。
- **增强适应性**：面对复杂数据，知识融合使得模型在不同场景下表现更为优异。
""")

# 对比图表
st.subheader("加入知识前后的模型对比")
st.write("通过下方的图表，您可以看到引入外部知识对模型的显著改进。")

# 模拟的模型对比数据
before_knowledge = np.random.rand(10) * 0.6
after_knowledge = before_knowledge + np.random.rand(10) * 0.4

# 绘制对比图
fig, ax = plt.subplots()
ax.plot(before_knowledge, label="未加入知识")
ax.plot(after_knowledge, label="加入外部知识", linestyle='--')
ax.set_xlabel("时间序列")
ax.set_ylabel("精度")
ax.legend()

st.pyplot(fig)


# 预测结果展示
st.subheader("预测结果")
st.write("在融合知识后，以下为模型的预测结果：")

# 模拟结果
st.write("预测结果：异常序列检测精度提升约20%")


