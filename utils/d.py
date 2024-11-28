import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

class mydataReader:
    def __init__(self, filename) -> None:
        # 加载数据，使用两个特征
        data_csv = pd.read_csv(
            filename, usecols=['time', 'Water_Level_LAT', 'utide'],
            index_col='time', parse_dates=['time']
        )
        data_csv = data_csv.dropna()

        # 对两个特征分别归一化
        max_values = data_csv.max()
        min_values = data_csv.min()
        scalars = max_values - min_values

        with open("./log/scalar.txt", mode='w+', encoding='utf-8') as f:
            f.write(f"max_values = {max_values.to_dict()} \nmin_values = {min_values.to_dict()} \nscalars = {scalars.to_dict()}")
        
        for col in ['Water_Level_LAT', 'utide']:
            data_csv[col] = data_csv[col].map(lambda x: (x - min_values[col]) / scalars[col])

        self.data_csv = data_csv
        dataset = data_csv.values
        self.dataset = dataset.astype('float32')

    def split(self, lookback, trainSet_ratio=0.7, valSet_ratio=0.1):
        dataX, dataY = [], []
        for i in range(len(self.dataset) - lookback):
            a = self.dataset[i:(i + lookback), :]
            dataX.append(a)
            dataY.append(self.dataset[i + lookback, 0])  # 目标是 'Water_Level_LAT'

        data_X, data_Y = np.array(dataX), np.array(dataY)
        train_size = int(len(data_X) * trainSet_ratio)
        val_size = int(len(data_X) * valSet_ratio)
        test_size = len(data_X) - train_size - val_size

        train_X = data_X[:train_size]
        train_Y = data_Y[:train_size]
        val_X = data_X[train_size:train_size + val_size]
        val_Y = data_Y[train_size:train_size + val_size]
        test_X = data_X[train_size + val_size:]
        test_Y = data_Y[train_size + val_size:]

        print(f"测试集大小为{test_size}")
        return (train_X, train_Y), (val_X, val_Y), (test_X, test_Y)

    def getSeries(self):
        return self.data_csv
class custom_dataset(Dataset):
    def __init__(self, data_X, data_Y):
        self.X = torch.tensor(data_X, dtype=torch.float32)
        self.Y = torch.tensor(data_Y, dtype=torch.float32)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)
