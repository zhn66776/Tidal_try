import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time  # helper libraries

# 定义文件路径
input_file = "StockPricesPredictionProject/can1998HA.csv"

# 转换数据集函数
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# 设置随机种子
np.random.seed(5)

# 加载数据集
df = read_csv(input_file, delimiter=',')
all_y = df['Water_Level_LAT'].values
dataset = all_y.reshape(-1, 1)

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# 分割数据集
train_size = int(len(dataset) * 0.9)  # 90% 训练
val_size = int(len(dataset) * 0.05)    # 5% 验证
test_size = len(dataset) - train_size - val_size  # 剩余 5% 测试
train, val, test = dataset[0:train_size, :], dataset[train_size:train_size + val_size, :], dataset[train_size + val_size:, :]

# 定义时间步长=======================================================================================         loookback          =======================================
look_back = 200
trainX, trainY = create_dataset(train, look_back)
valX, valY = create_dataset(val, look_back)
testX, testY = create_dataset(test, look_back)

# 重塑输入数据
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
valX = np.reshape(valX, (valX.shape[0], 1, valX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# 创建 LSTM 模型，添加 L2 正则化和学习率
learning_rate = 0.00001  # 设置学习率
weight_decay = 0.0001    # 设置 L2 正则化系数

model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back), kernel_regularizer=l2(weight_decay)))
model.add(Dropout(0.1))
model.add(Dense(1))

# 使用带有自定义学习率的 Adam 优化器
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='mse', optimizer=optimizer)

# 使用验证集训练模型并记录损失================================================================================         parameter         =============================================================
history = model.fit(trainX, trainY, validation_data=(valX, valY), epochs=200, batch_size=64, verbose=1)

# 绘制训练和验证损失
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error Loss')
plt.title('Train and Validation Loss over Epochs')
plt.legend()
plt.show()

# 滚动预测
rolling_predictions = []
current_input = testX[0]  # 使用测试集的第一个数据作为初始输入

for i in range(len(testY)):
    # 预测当前时间步
    prediction = model.predict(current_input.reshape(1, 1, look_back), verbose=0)
    rolling_predictions.append(prediction[0, 0])
    
    # 将当前预测加入到输入中，进行滚动预测
    current_input = np.roll(current_input, -1)  # 向左滚动一位
    current_input[0, -1] = prediction  # 添加预测值到输入的最后一位

# 逆归一化预测值和实际值
rolling_predictions = scaler.inverse_transform(np.array(rolling_predictions).reshape(-1, 1))
test_actual_data = scaler.inverse_transform(testY.reshape(-1, 1))

# 计算 R², RMSE 和 MAE
rmse = math.sqrt(mean_squared_error(test_actual_data, rolling_predictions))
mae = mean_absolute_error(test_actual_data, rolling_predictions)
r2 = r2_score(test_actual_data, rolling_predictions)

print(f'Test RMSE: {rmse:.2f}')
print(f'Test MAE: {mae:.2f}')
print(f'Test R²: {r2:.2f}')

# 绘制测试集的实际值和滚动预测值进行对比
plt.figure(figsize=(18, 10))
plt.plot(test_actual_data, label="Actual Test Data")  # 测试集实际数据
plt.plot(rolling_predictions, label="Rolling Predicted Test Data", linestyle="--")  # 滚动预测数据
plt.xlabel("Time Steps")
plt.ylabel("Water Level")
plt.title("Test Data: Actual vs Rolling Predicted")
plt.legend()
plt.show()

