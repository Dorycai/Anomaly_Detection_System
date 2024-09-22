import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

# 设置页面标题
st.set_page_config(page_title="模型推理与预测", page_icon="🔮")

# 页面标题
st.markdown("# 模型推理与预测")

# 模型列表整合（包含预训练模型和用户训练模型）
all_models = [
    {"id": "pretrained_1", "name": "预训练模型1", "type": "SVC", "hyperparameters": "C=1.0, kernel=linear",
     "evaluation_metrics": "Accuracy: 0.85"},
    {"id": "pretrained_2", "name": "预训练模型2", "type": "Bayes", "hyperparameters": "alpha=1.0",
     "evaluation_metrics": "Accuracy: 0.78"},
    {"id": "trained_1", "name": "用户训练模型1", "type": "Random Forest", "hyperparameters": "n_estimators=100",
     "evaluation_metrics": "Accuracy: 0.92"},
    {"id": "trained_2", "name": "用户训练模型2", "type": "MLP", "hyperparameters": "hidden_layer_sizes=(100,)",
     "evaluation_metrics": "Accuracy: 0.88"},
    # 添加其他预训练和用户训练的模型信息
]

# 将模型列表转换为 DataFrame
model_df = pd.DataFrame(all_models)

# 显示模型信息表
st.write("### 模型选择列表")
st.dataframe(model_df, height=300)

# 用户选择模型
selected_model_id = st.selectbox("选择模型 ID", options=model_df["id"])

# 显示选定模型的详细信息
selected_model_info = model_df[model_df["id"] == selected_model_id].iloc[0]
st.write("### 您选择的模型信息")
st.write(f"**模型名称**: {selected_model_info['name']}")
st.write(f"**模型类型**: {selected_model_info['type']}")
st.write(f"**超参数配置**: {selected_model_info['hyperparameters']}")
st.write(f"**评估指标**: {selected_model_info['evaluation_metrics']}")

# 推理数据集选择（与之前“我的数据集”一致）
st.write("## 选择推理数据集")
my_datasets = ["我的数据集1", "我的数据集2", "我的数据集3"]  # 示例数据集
selected_dataset = st.selectbox("选择用于推理的数据集", options=my_datasets)

# 开始推理按钮
if st.button("开始预测"):
    st.write(f"正在使用模型 `{selected_model_info['name']}` 对 `{selected_dataset}` 进行预测...")

    # 模拟推理过程，每个模型对每个序列选择一个最佳异常检测算法
    algorithms = [
        "IForest", "IForest1", "LOF", "MP", "NormA", "PCA", "AE",
        "LSTM-AD", "POLY", "CNN", "OCSVM", "HBOS"
    ]

    # 预测结果（模拟），随机生成每个序列的预测结果
    predictions = {
        selected_model_info['name']: np.random.choice(algorithms, size=10)  # 模拟每个模型的预测结果
    }

    # 显示模型的预测结果
    st.write("## 模型预测的最佳异常检测算法")
    st.write(f"模型 `{selected_model_info['name']}` 预测结果：")
    st.write(pd.DataFrame({
        "输入数据序列": [f"序列{i}" for i in range(1, 11)],
        "预测的最佳异常检测算法": predictions[selected_model_info['name']]
    }))

    # 多数投票逻辑：对每条输入数据根据模型结果进行投票，选择最多的异常检测算法
    final_predictions = []
    for i in range(10):  # 对每个数据序列进行投票
        votes = [predictions[selected_model_info['name']][i]]
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
