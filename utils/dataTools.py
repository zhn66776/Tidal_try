import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

# 定义读取时间序列并进行MMS归一化的dataReader
# class mydataReader:
#     def __init__(self,filename) -> None:
#         data_csv = pd.read_csv(filename, usecols=['time','Water_Level_LAT'],
#         index_col='time',parse_dates=['time'])
#         # ##############################################################
#         # Tips:
#         # 数据预处理(归一化有利于梯度下降)-MMS标准化
#         # 如果不进行归一化，那么由于特征向量中不同特征的取值相差较大，
#         # 会导致目标函数变“扁”。这样在进行梯度下降的时候，
#         # 梯度的方向就会偏离最小值的方向，走很多弯路，即训练时间过长。
#         # ##############################################################
#         data_csv = data_csv.dropna()
#         max_value = data_csv["Water_Level_LAT"].max()
#         min_value = data_csv["Water_Level_LAT"].min()
#         scalar = max_value - min_value
#         with open("./log/scalar.txt",mode='w+',encoding='utf-8') as f:
#             f.write("max_value = {0} \nmin_value = {1} \nscalar = {2}".format(max_value,min_value,scalar))
#         f.close()

#         data_csv["Water_Level_LAT"] = data_csv["Water_Level_LAT"].map(lambda x: (x-min_value)/scalar)

#         self.data_csv = data_csv

#         # 改变值类型
#         dataset = data_csv.values
#         self.dataset = dataset.astype('float32')

#     def split(self,lookback,trainSet_ratio = 0.7,valSet_ratio = 0.1):
#         # 数据集创建
#         dataX, dataY = [], []
#         # print(self.dataset.shape) (19316, 1) data_csv.values会多出一个维度
#         for i in tqdm(range(len(self.dataset) - lookback)):
#             a = self.dataset[i:(i + lookback)]
#             dataX.append(a)
#             dataY.append(self.dataset[i + lookback])
#         data_X,data_Y = np.array(dataX), np.array(dataY)
#         # 去掉那个多的维度
#         data_X = data_X.squeeze()

#         # print(data_X.shape) (19306, 10)
#         # print(data_Y.shape) (19306, 1)

#         # 数据划分
#         train_size = int(len(data_X) * trainSet_ratio)
#         val_size = int(len(data_X) * valSet_ratio)
#         test_size = len(data_X) - train_size - val_size

#         train_X = data_X[:train_size]
#         train_Y = data_Y[:train_size]
#         val_X = data_X[train_size:train_size+val_size]
#         val_Y = data_Y[train_size:train_size+val_size]
#         test_X = data_X[train_size+val_size:]
#         test_Y = data_Y[train_size+val_size:]

#         print("测试集大小为{}".format(test_size))
#         return (train_X,train_Y) , (val_X,val_Y) , (test_X,test_Y)

#     def getSeries(self):
#         """
#         返回原始序列
#         """
#         return self.data_csv

# # 定义一个子类叫 custom_dataset，继承与 Dataset
# class custom_dataset(Dataset):
#     def __init__(self,data_X,data_Y):
#         """
#         :parameters:
#         data_X: 构造好的X矩阵
#         data_Y: 构造好的Y标签
#         """
#         self.X = torch.tensor(data_X, dtype=torch.float32)
#         self.Y = torch.tensor(data_Y, dtype=torch.float32)

#     def __getitem__(self, index):
#         return self.X[index], self.Y[index]

#     def __len__(self):
#         return len(self.X)

class mydataReader:
    def __init__(self, filename):
        data_csv = pd.read_csv(filename, usecols=['time', 'Water_Level_LAT'],
                               index_col='time', parse_dates=['time'])
        # 数据预处理（归一化）
        data_csv = data_csv.dropna()
        max_value = data_csv["Water_Level_LAT"].max()
        min_value = data_csv["Water_Level_LAT"].min()
        scalar = max_value - min_value
        with open("./log/scalar.txt", mode='w+', encoding='utf-8') as f:
            f.write("max_value = {0} \nmin_value = {1} \nscalar = {2}".format(max_value, min_value, scalar))
        data_csv["Water_Level_LAT"] = data_csv["Water_Level_LAT"].map(lambda x: (x - min_value) / scalar)

        self.data_csv = data_csv
        # 转换为numpy数组
        dataset = data_csv.values
        self.dataset = dataset.astype('float32')

    def split(self, lookback, n_steps, trainSet_ratio=0.7, valSet_ratio=0.1):
        # 创建数据集，Y包含未来n_steps个数据点
        dataX, dataY = [], []
        for i in tqdm(range(len(self.dataset) - lookback - n_steps + 1)):
            a = self.dataset[i:(i + lookback)]
            dataX.append(a)
            dataY.append(self.dataset[(i + lookback):(i + lookback + n_steps)])
        data_X, data_Y = np.array(dataX), np.array(dataY)
        data_X = data_X.squeeze()

        # 数据划分
        total_samples = len(data_X)
        train_size = int(total_samples * trainSet_ratio)
        val_size = int(total_samples * valSet_ratio)
        test_size = total_samples - train_size - val_size

        train_X = data_X[:train_size]
        train_Y = data_Y[:train_size]
        val_X = data_X[train_size:train_size + val_size]
        val_Y = data_Y[train_size:train_size + val_size]
        test_X = data_X[train_size + val_size:]
        test_Y = data_Y[train_size + val_size:]

        print("测试集大小为{}".format(test_size))
        return (train_X, train_Y), (val_X, val_Y), (test_X, test_Y)

    def getSeries(self):
        """
        返回原始序列
        """
        return self.data_csv


# 定义一个子类叫 custom_dataset，继承与 Dataset
class custom_dataset(Dataset):
    def __init__(self, data_X, data_Y):
        self.data_X = torch.tensor(data_X, dtype=torch.float32)
        self.data_Y = torch.tensor(data_Y, dtype=torch.float32)

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self, idx):
        return self.data_X[idx], self.data_Y[idx]


"""    废置代码
# 定义读取时间序列并进行MMS归一化的dataReader
def mydataReader(filename):
    data_csv = pd.read_csv(filename, usecols=['time','Water_Level_LAT'],
    index_col='time',parse_dates=['time'])

    # 数据预处理(归一化有利于梯度下降)
    # 如果不进行归一化，那么由于特征向量中不同特征的取值相差较大，
    # 会导致目标函数变“扁”。这样在进行梯度下降的时候，
    # 梯度的方向就会偏离最小值的方向，走很多弯路，即训练时间过长。
    data_csv = data_csv.dropna()
    max_value = data_csv["Water_Level_LAT"].max()
    min_value = data_csv["Water_Level_LAT"].min()
    scalar = max_value - min_value
    data_csv["Water_Level_LAT"] = data_csv["Water_Level_LAT"].map(lambda x: (x-min_value)/scalar)
    
    dataset = data_csv.values
    dataset = dataset.astype('float32')
    return dataset

# supervised-prediction
def create_dataset(dataset, look_back=10):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

# 数据划分
def mydataSplit(data_X,data_Y,trainSet_ratio = 0.7):
    train_size = int(len(data_X) * trainSet_ratio)
    test_size = len(data_X) - train_size
    train_X = data_X[:train_size]
    train_Y = data_Y[:train_size]
    test_X = data_X[train_size:]
    test_Y = data_Y[train_size:]

    print("测试集大小为{}".format(test_size))
    return (train_X,train_Y), (test_X,test_Y)
"""

