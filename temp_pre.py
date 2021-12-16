import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch
from torch import nn
from matplotlib import pyplot as plt


def linear_regression(x, y, test_x, test_y, D_inputs=49, D_outputs=7, epoch=10001, temp='none', lr=0.0001):
    # 构建模型
    model = nn.Sequential(nn.Linear(D_inputs, D_outputs))
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    print(temp)
    # 训练模型
    for t in range(epoch):
        # train_set
        model.zero_grad()  # 梯度清零
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()

        # test_set
        y_pred_test = model(test_x)
        test_loss = loss_fn(y_pred_test, test_y)

        # 参数更新
        with torch.no_grad():  # 参数更新方式
            for parm in model.parameters():
                parm -= lr * parm.grad
        if t % 100 == 0:
            print('iter: {}\ttrain_loss: {}\ttest_loss: {}'.format(t, loss, test_loss))

    out = model(x[666, :])
    out_pred = model(test_x[-1, :])
    print(out)
    print(y[666])
    print('pred:{}'.format(out_pred))
    torch.save(model.state_dict(), '{}.pt'.format(temp))
    return


def pred(pred_x_set):
    # 构建模型
    model_high = nn.Sequential(nn.Linear(D_inputs, D_outputs))
    model_low = nn.Sequential(nn.Linear(D_inputs, D_outputs))
    model_high.load_state_dict(torch.load('high.pt'))
    model_low.load_state_dict(torch.load('low.pt'))
    day_h = {}  # 收集每天的平均温度（最高/最低）
    day_l = {}  # 收集每天的平均温度（最高/最低）
    for i in range(7):
        name1 = '4月{}号最高温度'.format(i + 1)
        name2 = '4月{}号最低温度'.format(i + 1)
        day_h[name1] = 0
        day_l[name2] = 0
        for city in cities:
            day_h[name1] += model_high(pred_x_set[city]).detach().numpy()[i]
            day_l[name2] += model_low(pred_x_set[city]).detach().numpy()[i]
        day_h[name1] /= 11
        day_h[name1] = int(day_h[name1] + 0.5)
        day_l[name2] /= 11
        day_l[name2] = int(day_l[name2] + 0.5)

    print(day_h)
    print(day_l)

    x = list(day_h.keys())
    for i in range(len(x)):
        x[i] = x[i][:-4]
    y_h = day_h.values()
    y_l = day_l.values()
    # 画一个折线图
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 用来设置字体样式
    plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
    plt.style.use('fivethirtyeight')  # 设置主题
    fig = plt.figure(figsize=(15, 9))
    plt.title('四月一号到七号最高温度/最低温度折线图')
    plt.xlabel('日期', fontsize=14)
    plt.xticks(rotation=45)  # 使x轴坐标文本反转45°
    plt.ylabel('温度/℃', fontsize=14)
    plt.ylim(15, 30)
    # plt.xlim(0.5, 31.5)
    plt.plot(x, y_h, color='deeppink', linewidth=1.5, linestyle='--', label='每日最高气温', marker='o')
    plt.plot(x, y_l, color='darkblue', linewidth=1.5, linestyle=':', label='每日最低气温', marker='+')
    plt.legend()  # 显示 x、y 轴说明
    plt.grid()  # 显示网格线
    # plt.show()
    plt.savefig('四月份气温预测图')


def data_process(x):
    # 具体的天气数据预处理函数，把温度、天气、风向等数据转化为数字
    for i in range(0, len(x[:, 0])):
        # 对最高温度进行处理
        x[:, 0][i] = float(x[:, 0][i][:-1])
        # 对最低温度进行处理
        x[:, 1][i] = float(x[:, 1][i][:-1])
        # 对天气情况进行处理
        judge = 0
        times = 0
        if '暴雨' in x[:, 2][i]:
            judge += 0
            times += 1
        if '大雨' in x[:, 2][i]:
            judge += 1
            times += 1
        if '雷阵雨' in x[:, 2][i]:
            judge += 2
            times += 1
        if '中雨' in x[:, 2][i]:
            judge += 3
            times += 1
        if '小雨' in x[:, 2][i]:
            judge += 4
            times += 1
        if '小雪' in x[:, 2][i]:
            judge += 5
            times += 1
        if '阵雪' in x[:, 2][i]:
            judge += 6
            times += 1
        if '雨夹雪' in x[:, 2][i]:
            judge += 7
            times += 1
        if '雾' or '霾' in x[:, 2][i]:
            judge += 8
            times += 1
        if '阴' in x[:, 2][i]:
            judge += 9
            times += 1
        if '多云' in x[:, 2][i]:
            judge += 10
            times += 1
        if '晴' in x[:, 2][i]:
            judge += 10
            times += 1

        x[:, 2][i] = judge / times
        # 对风向进行处理
        judge = 0
        if '北风' in x[:, 3][i]:
            judge = 0
        if '东北风' in x[:, 3][i]:
            judge = 0.5
        if '东风' in x[:, 3][i]:
            judge = 1
        if '东南风' in x[:, 3][i]:
            judge = 1.5
        if '南风' in x[:, 3][i]:
            judge = 2
        if '西南风' in x[:, 3][i]:
            judge = 2.5
        if '西风' in x[:, 3][i]:
            judge = 3
        if '西北风' in x[:, 3][i]:
            judge = 3.5
        x[:, 3][i] = judge
        # 对风力进行处理
        judge = 0
        if '微' in x[:, 4][i]:
            judge = 0
        if '1级' in x[:, 4][i]:
            judge = 1
        if '2级' in x[:, 4][i]:
            judge = 2
        if '3级' in x[:, 4][i]:
            judge = 3
        if '4级' in x[:, 4][i]:
            judge = 4
        if '5级' in x[:, 4][i]:
            judge = 5
        if '6级' in x[:, 4][i]:
            judge = 6
        if '7级' in x[:, 4][i]:
            judge = 7
        if '8级' in x[:, 4][i]:
            judge = 8
        x[:, 4][i] = float(judge)
        # 对空气指数进行处理
        x[:, 5][i] = float(x[:, 5][i])
        # 对空气质量进行处理
        judge = 0
        if '优' in x[:, 6][i]:
            judge = 0
        if '良' in x[:, 6][i]:
            judge = 1
        if '轻度' in x[:, 6][i]:
            judge = 2
        if '中度' in x[:, 6][i]:
            judge = 3
        x[:, 6][i] = float(judge)
    x = x.astype(float)  # x 处理前是 pandas 的 object 类型

    # 实现数据的标准化
    '''
    sklearn.preprocessing.scale() 实现数据标准化
    参数解释：
        X : {array-like, sparse matrix}
        要标准化的数据，numpy的array类数据。

        axis : int (0 by default)
        处理哪个维度，0表示处理横向的数据（行）， 1表示处理纵向的数据（列），默认为0

        with_mean : boolean, True by default
        是否中心化。

        with_std : boolean, True by default
        是否标准化。

        copy : boolean, optional, default True
        是否复制。
    '''
    x_scale = preprocessing.scale(x, axis=0, with_mean=True, with_std=True, copy=True)

    # 实现数据的分类处理
    '''
    一共31*12个数据，我的做法是：取1-7天的天气数据，预测8-14天的温度信息
    依此类推，取2-8天的数据，预测9-15天的温度
    易得，我可以组成18组数据作为我的 train set
    为了方便 torch.from_numpy()运算（只支持2维数组），把1-7天的7×7的矩阵压扁成1×49的矩阵
    x_data 为353批次，每批次包含七天天气数据的test集，即353*49的矩阵
    y_h_data和y_l_data为353批次，每批次包含七天的温度数据的val集，即353*7矩阵
    fianl_data为3月25-31的数据，用来预测4月1号到7号的数据
    '''
    x_data = []
    for i in range(0, x_scale.shape[0] - 13):
        x_data.append(np.append([], x_scale[i:i + 7, :]))
    x_data = np.array(x_data)

    y_h_data = []
    for i in range(7, x_scale.shape[0] - 6):
        y_h_data.append(np.append([], x[i:i + 7, 0]))
    y_h_data = np.array(y_h_data)

    y_l_data = []
    for i in range(7, x_scale.shape[0] - 6):
        y_l_data.append(np.append([], x[i:i + 7, 1]))
    y_l_data = np.array(y_l_data)

    final_data = np.array([np.append([], x_scale[x_scale.shape[0] - 7:x_scale.shape[0], :])])
    final_data = final_data[0]  # 2021年3月25-31号数据

    # 把数据从 numpy 格式转化为 Tensor
    x_data = torch.from_numpy(x_data)
    y_h_data = torch.from_numpy(y_h_data)
    y_l_data = torch.from_numpy(y_l_data)
    final_data = torch.from_numpy(final_data)

    # 设定tensor格式
    x_data = torch.tensor(x_data, dtype=torch.float32)
    y_h_data = torch.tensor(y_h_data, dtype=torch.float32)
    y_l_data = torch.tensor(y_l_data, dtype=torch.float32)
    final_data = torch.tensor(final_data, dtype=torch.float32)

    return x_data, y_h_data, y_l_data, final_data


# 超参
D_inputs = 49  # 七天的数据都作为一个样本，产生7*7=49个特征参数
D_outputs = 7

learning_rate = 0.001
num_epoches = 8001

global cities
cities = ['杭州', '宁波', '温州', '嘉兴', '湖州', '绍兴', '金华', '衢州', '舟山', '台州', '丽水']

# train_set
train_df = pd.read_csv('data2019-2020.csv', encoding='gbk').fillna(str(0))
train_x_set = torch.Tensor()
train_y_h_set = torch.Tensor()
train_y_l_set = torch.Tensor()

# test_set
test_df = pd.read_csv('data.csv', encoding='gbk').fillna(str(0))
test_x_set = torch.Tensor()
test_y_h_set = torch.Tensor()
test_y_l_set = torch.Tensor()

# pred_test
pred_x_set = {}

for city in cities:
    # train_set
    # 读取数据，并把前两列数据裁剪掉
    train_data = np.array(train_df.loc[train_df['city'] == city])[:, 2:-1]
    # 具体的天气数据预处理函数，把温度、天气、风向等数据转化为数字
    [x, y_h, y_l, _] = data_process(train_data)
    train_x_set = torch.cat([train_x_set, x], dim=0)
    train_y_h_set = torch.cat([train_y_h_set, y_h], dim=0)
    train_y_l_set = torch.cat([train_y_l_set, y_l], dim=0)

    # test_set and pred_set
    test_data = np.array(test_df.loc[test_df['city'] == city])[:, 2:-1]
    [x, y_h, y_l, final_x] = data_process(test_data)
    test_x_set = torch.cat([test_x_set, x], dim=0)
    test_y_h_set = torch.cat([test_y_h_set, y_h], dim=0)
    test_y_l_set = torch.cat([test_y_l_set, y_l], dim=0)
    pred_x_set[city] = final_x

print(train_x_set.shape)
print(train_y_h_set.shape)
linear_regression(train_x_set, train_y_h_set, test_x_set, test_y_h_set, epoch=num_epoches, temp='high',
                  lr=learning_rate)
linear_regression(train_x_set, train_y_l_set, test_x_set, test_y_l_set, epoch=5801, temp='low',
                  lr=learning_rate)

pred(pred_x_set)
