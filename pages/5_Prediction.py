import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

st.set_page_config(page_title="推理与预测", page_icon="🔮")

st.markdown("# 推理与预测")

st.write("## 选择模型")

# 12个异常检测算法的列表
algorithms = [
    "IForest", "IForest1", "LOF", "MP", "NormA", "PCA", "AE",
    "LSTM-AD", "POLY", "CNN", "OCSVM", "HBOS"
]

# 预训练模型下拉选择框
pretrained_models = ["model1", "model2", "model3"]  # 预训练模型的名称列表
selected_pretrained_models = st.multiselect("选择预训练模型（可多选）", options=pretrained_models)

# 用户训练的模型下拉选择框，命名格式为“model_参数1_参数2等（后续继续完善）”
trained_models = ["model_64_0.5", "model_128_0.3", "model_256_0.1"]  # 示例模型
selected_trained_models = st.multiselect("选择训练模型（可多选）", options=trained_models)

# 推理数据集选择（从之前的"我的数据集"中选择，不用重新上传）
st.write("## 选择推理数据集")
# 假设已经有在“我的数据集”上传的数据，可以从列表中选择
my_datasets = ["我的数据集1", "我的数据集2", "我的数据集3"]  # 模拟数据集
selected_dataset = st.selectbox("选择用于推理的数据集", options=my_datasets)

# 开始推理按钮
if st.button("开始预测"):
    if not selected_pretrained_models and not selected_trained_models:
        st.warning("请至少选择一个模型进行推理。")
    else:
        st.write(f"正在使用模型 `{selected_pretrained_models + selected_trained_models}` 对 `{selected_dataset}` 进行预测...")

        # 模拟多个模型的推理过程，每个模型对每个序列选择一个最佳异常检测算法
        predictions = {
            model: np.random.choice(algorithms, size=10)  # 模拟每个模型的预测结果为12个算法中的一个
            for model in selected_pretrained_models + selected_trained_models
        }

        # 显示各个模型的预测结果
        st.write("## 各模型预测的最佳异常检测算法")
        for model, preds in predictions.items():
            st.write(f"模型 `{model}` 预测结果：")
            st.write(pd.DataFrame({
                "输入数据序列": [f"序列{i}" for i in range(1, 11)],
                "预测的最佳异常检测算法": preds
            }))

        # 多数投票逻辑：对每条输入数据根据所有模型的结果进行投票，选择最多的异常检测算法
        final_predictions = []
        for i in range(10):  # 对每个数据序列进行投票
            votes = [predictions[model][i] for model in predictions]
            final_result = Counter(votes).most_common(1)[0][0]  # 选择票数最多的异常检测算法
            final_predictions.append(final_result)

        # 显示每个序列的最终投票结果
        st.write("## 每个序列的最终投票结果")
        final_df = pd.DataFrame({
            "输入数据序列": [f"序列{i}" for i in range(1, 11)],
            "最终选择的最佳异常检测算法": final_predictions
        })
        st.dataframe(final_df)

        # 统计整个数据集中投票最多的异常检测算法
        overall_best_algorithm = Counter(final_predictions).most_common(1)[0][0]

        # 显示整个数据集的最优异常检测算法
        st.write(f"## 整个数据集的最佳异常检测算法为： **{overall_best_algorithm}**")
