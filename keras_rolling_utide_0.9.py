import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Input
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 设置随机种子
np.random.seed(5)

# 定义文件路径
input_file = "dataProcessed/ABE5Y1_HA_processed.csv"

# 加载数据集
df = read_csv(input_file, delimiter=',')

# 仅使用最后 10,000 个数据点
df = df.iloc[-10000:]

# 提取特征
dataset = df[['anomaly', 'utide']].values

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# 分割数据集
train_size = 8000
test_size = 1000
train = dataset[:train_size]
test = dataset[train_size:train_size + test_size]

# 定义时间步长
look_back = 30

# 创建数据集函数
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])  # 目标是下一个时间步的 anomaly
    return np.array(dataX), np.array(dataY)

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# 获取特征数量
num_features = trainX.shape[2]

# 创建 BiLSTM 模型
learning_rate = 0.001
weight_decay = 0.0001

input_layer = Input(shape=(look_back, num_features))
x = Bidirectional(LSTM(50, kernel_regularizer=l2(weight_decay)))(input_layer)
x = Dropout(0.3)(x)
output_layer = Dense(1)(x)

model = Model(inputs=input_layer, outputs=output_layer)

# 编译模型
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='mse', optimizer=optimizer)

# 定义动态学习率回调
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=10, min_lr=1e-6, verbose=1)

# 使用验证集训练模型
history = model.fit(trainX, trainY, validation_split=0.1, epochs=200, batch_size=64, verbose=1, callbacks=[reduce_lr])

# 绘制训练和验证损失
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error Loss')
plt.title('Train and Validation Loss over Epochs')
plt.legend()
plt.show()

# ===========================
# 测试集滚动预测
# ===========================

# 初始化输入序列，使用测试集的第一个输入序列
current_input = testX[0]  # 形状为 (look_back, num_features)

# 提取测试集的 utide 特征
utide_test = test[:, 1]  # 获取测试集所有的 utide 值

# 存储预测结果
rolling_predictions = []

# 滚动预测测试集长度的数据
for i in range(len(testY)):
    # 准备输入数据
    current_input_reshaped = current_input.reshape(1, look_back, num_features)
    
    # 预测下一个时间步的 anomaly，禁用进度输出
    prediction = model.predict(current_input_reshaped, verbose=0)
    
    # 存储预测结果
    rolling_predictions.append(prediction[0, 0])
    
    # 更新输入序列
    current_input = np.roll(current_input, -1, axis=0)
    
    # 更新输入序列的最后一行
    current_input[-1, 0] = prediction[0, 0]       # 使用预测的 anomaly 值
    current_input[-1, 1] = utide_test[look_back + i]  # 使用已知的 utide 值

# 将预测结果转换为数组
rolling_predictions = np.array(rolling_predictions)

# 逆归一化预测值和实际值
predictions_full = np.zeros((len(rolling_predictions), num_features))
predictions_full[:, 0] = rolling_predictions  # 填充预测的 anomaly 值
predictions_full[:, 1] = test[look_back:, 1]  # 填充实际的 utide 值（已知的）

testY_full = np.zeros((len(testY), num_features))
testY_full[:, 0] = testY  # 实际的 anomaly 值
testY_full[:, 1] = test[look_back:, 1]  # 实际的 utide 值

# 逆归一化
predictions_inverse = scaler.inverse_transform(predictions_full)[:, 0]
testY_inverse = scaler.inverse_transform(testY_full)[:, 0]
utide_inverse = scaler.inverse_transform(test[look_back:])[:, 1]

# 计算 R², RMSE 和 MAE
rmse = math.sqrt(mean_squared_error(testY_inverse, predictions_inverse))
mae = mean_absolute_error(testY_inverse, predictions_inverse)
r2 = r2_score(testY_inverse, predictions_inverse)

print(f'Test RMSE: {rmse:.2f}')
print(f'Test MAE: {mae:.2f}')
print(f'Test R²: {r2:.2f}')
aligned_utide = utide_inverse

# 绘制预测和实际值进行对比
plt.figure(figsize=(18, 10))
plt.plot(testY_inverse, label="Actual Test Data")  # 测试集实际数据
plt.plot(predictions_inverse, label="Predicted Test Data", linestyle="--")  # 预测数据
plt.plot(aligned_utide, label="Utide (Known Feature)", linestyle="-.", color='green', linewidth=1.5)

plt.xlabel("Time Steps")
plt.ylabel("Anomaly Value")
plt.title("Test Data: Actual vs Predicted (Rolling Forecast)")
plt.legend()
plt.show()
