import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


# 从pyplot导入MultipleLocator类，这个类用于设置刻度间隔


def draw(city, h_temp_arr, l_temp_arr):
    for i in range(31):  # 把气温的数据从str 转化为 int
        h_temp_arr[i] = int(h_temp_arr[i][:-1])
        l_temp_arr[i] = int(l_temp_arr[i][:-1])

    date = []
    for i in range(1, 32):  # 设定日期数组
        date.append(i)

    # 正式画图
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 用来设置字体样式
    plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
    plt.style.use('fivethirtyeight')  # 设置主题
    fig = plt.figure(figsize=(15, 8))
    plt.title(city + ' 三月份最高温度/最低温度折线图')
    plt.xlabel('日期', fontsize=14)
    plt.ylabel('温度/℃', fontsize=14)
    plt.xlim(0.5, 31.5)
    plt.plot(date, h_temp_arr, color='deeppink', linewidth=1.5, linestyle='--', label='每日最高气温', marker='o')
    plt.plot(date, l_temp_arr, color='darkblue', linewidth=1.5, linestyle=':', label='每日最低气温', marker='+')
    plt.legend()  # 显示 x、y 轴说明
    plt.grid()  # 显示网格线
    # plt.show()
    plt.savefig(city + " 三月份气温折线图.png")


df = pd.read_csv('data.csv', encoding='gbk')
h_temp_df = pd.DataFrame(df, columns=['city', 'bWendu'])  # 筛选出每日最高气温
l_temp_df = pd.DataFrame(df, columns=['city', 'yWendu'])  # 筛选出每日最低气温
cities = ['杭州', '宁波', '温州', '嘉兴', '湖州', '绍兴', '金华', '衢州', '舟山', '台州', '丽水']
for city in cities:
    h_temp_arr = np.array(h_temp_df.loc[h_temp_df['city'] == city])
    l_temp_arr = np.array(l_temp_df.loc[l_temp_df['city'] == city])
    draw(city, h_temp_arr[:, 1], l_temp_arr[:, 1])
