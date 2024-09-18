import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import numpy as np

# 设置页面标题
st.set_page_config(page_title="数据集管理", page_icon="📊")

# 页面标题
st.markdown("# 数据集管理")

# 页面选项卡选择
tab = st.radio("选择数据集页面", ["训练数据集", "我的数据集"])

### 训练数据集页面 ###
if tab == "训练数据集":
    st.write("选择预设数据集并查看其统计信息。")

    # 预设的16个数据集及其简介
    datasets = {
        'Dodgers': 'Dodgers : 球赛结束后异常交通流量的时间序列数据（1个时间序列）',
        'ECG': 'ECG : 标准的心电图数据集（52个时间序列）',
        'IOPS': 'IOPS : 机器性能指标数据集（58个时间序列）',
        'KDD21': 'KDD21 : 来自SIGKDD 2021发布的综合数据集（250个时间序列）',
        'MGAB': 'MGAB : Mackey-Glass时间序列，含有复杂的异常（10个时间序列）',
        'NAB': 'NAB : 包含现实世界网络相关和人工时间序列（58个时间序列）',
        'SensorScope': 'SensorScope : 环境数据（23个时间序列）',
        'YAHOO': 'YAHOO : 基于Yahoo生产系统的时间序列（367个时间序列）',
        'Daphnet': 'Daphnet : 帕金森病患者佩戴的加速度传感器数据（45个时间序列）',
        'GHL': 'GHL : 燃油加热环路的遥测数据（126个时间序列）',
        'Genesis': 'Genesis : 可携带式拣选演示仪的时间序列数据（6个时间序列）',
        'MITDB': 'MITDB : 移动心电图记录数据（32个时间序列）',
        'OPPORTUNITY': 'OPPORTUNITY : 人类活动识别的运动传感器数据（465个时间序列）',
        'Occupancy': 'Occupancy : 房间的温度、湿度、光线、CO2水平（10个时间序列）',
        'SMD': 'SMD : 服务器机器遥测数据（281个时间序列）',
        'SVDB': 'SVDB : 心电图记录数据（115个时间序列）',
    }

    # 用户选择数据集
    dataset_name = st.selectbox("选择一个数据集", options=list(datasets.keys()))

    # 显示选择的数据集简介
    st.write(f"**{dataset_name}**: {datasets[dataset_name]}")

    # 模拟加载每个数据集的统计信息
    # 统计数据：文件名, period_length, ratio, nb_anomaly, average_anom_length, median_anom_length, std_anom_length
    data_info = pd.DataFrame({
        '文件名': [f'{dataset_name}_subseq_{i}' for i in range(1, 6)],
        'period_length': np.random.randint(20, 100, 5),
        'ratio': np.random.rand(5),
        'nb_anomaly': np.random.randint(1, 10, 5),
        'average_anom_length': np.random.rand(5) * 10,
        'median_anom_length': np.random.rand(5) * 5,
        'std_anom_length': np.random.rand(5) * 2
    })

    # 显示数据集的统计信息
    st.write("该数据集的子数据集统计信息：")
    st.dataframe(data_info)

### 我的数据集页面 ###
elif tab == "我的数据集":
    st.write("上传您的自定义数据集，系统将自动处理并进行统计分析。")

    # 用户上传数据集
    uploaded_file = st.file_uploader("上传您的时间序列数据集（CSV或Excel）", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        # 读取上传的文件
        if uploaded_file.name.endswith('.csv'):
            my_data = pd.read_csv(uploaded_file)
        else:
            my_data = pd.read_excel(uploaded_file)

        st.write("数据集预览：")
        st.dataframe(my_data.head())  # 展示数据集的前几行

        # 假设时间列在数据集中存在，选择时间列
        time_column = st.selectbox("选择时间列", options=my_data.columns)

        # 统计数据集的描述性信息
        st.write("数据集统计信息：")
        st.write(my_data.describe())

        # 生成折线图
        st.write("时间序列折线图：")
        plt.figure(figsize=(10, 6))
        plt.plot(my_data[time_column])
        plt.title(f'{uploaded_file.name} - 时间序列折线图')
        plt.xlabel('时间')
        plt.ylabel('值')
        st.pyplot(plt)
