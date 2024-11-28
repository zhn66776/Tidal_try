import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Input
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# 设置随机种子
np.random.seed(5)

# 定义文件路径
input_file = "dataProcessed/ABE5Y1_HA_processed.csv"

# 加载数据集
df = read_csv(input_file, delimiter=',')

# 确认数据集长度
total_length = len(df)
print(f"数据集总长度为：{total_length}")

# 定义要选取的连续数据点数量
num_points = 12000

# 确保数据集长度足够
if total_length >= num_points:
    max_start_index = total_length - num_points
    np.random.seed(int(time.time()))
    start_index = np.random.randint(0, max_start_index + 1)
    end_index = start_index + num_points - 1

    # 获取起始和结束时间
    start_time = df.iloc[start_index]['time']
    end_time = df.iloc[end_index]['time']

    # 选取连续的数据
    df = df.iloc[start_index:end_index + 1]

    print(f"选取的数据从索引 {start_index} 开始，到索引 {end_index} 结束。")
    print(f"对应的时间范围：从 {start_time} 到 {end_time}。")
else:
    print("数据集长度不足 12,000 行，无法选取。")

# 提取特征
# 仅使用 'anomaly' 和 'utide' 作为特征
feature_columns = ['anomaly', 'utide']
dataset = df[feature_columns].values

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)

# 分割数据集（时间序列分割）
train_size = 11000
test_size = 1000
train, test = dataset_scaled[:train_size], dataset_scaled[train_size:train_size + test_size]

print(f"训练集大小: {train.shape[0]}, 测试集大小: {test.shape[0]}")

# 定义时间步长
look_back = 30  # 增加look_back以捕捉更多历史信息

# 创建数据集函数（单步预测）
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])  # 目标是下一个时间步的 anomaly
    return np.array(dataX), np.array(dataY)

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

print(f"训练集X形状: {trainX.shape}, 训练集Y形状: {trainY.shape}")
print(f"测试集X形状: {testX.shape}, 测试集Y形状: {testY.shape}")

# 获取特征数量
num_features = trainX.shape[2]

# 创建 LSTM 模型
learning_rate = 0.001
weight_decay = 0.0001

input_layer = Input(shape=(look_back, num_features))
# x = Bidirectional(LSTM(250, return_sequences=True, kernel_regularizer=l2(weight_decay)))(input_layer)
# x = Bidirectional(LSTM(100, kernel_regularizer=l2(weight_decay)))(x)
# 添加更多的LSTM层
x = Bidirectional(LSTM(200, return_sequences=True, kernel_regularizer=l2(weight_decay)))(input_layer)
x = Bidirectional(LSTM(200, return_sequences=True, kernel_regularizer=l2(weight_decay)))(x)
x = Bidirectional(LSTM(50, kernel_regularizer=l2(weight_decay)))(x)
x = Dropout(0.4)(x)
output_layer = Dense(1)(x)

model = Model(inputs=input_layer, outputs=output_layer)

# 编译模型
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='mse', optimizer=optimizer)

# 定义回调
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=10, min_lr=1e-6, verbose=1)
#early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

def scheduler(epoch, lr):
    if epoch % 50 == 0 and epoch != 0:
        lr = lr * 0.5
    return lr

lr_scheduler = LearningRateScheduler(scheduler)

# 使用验证集训练模型
history = model.fit(
    trainX, trainY,
    validation_split=0.1,
    epochs=200,
    batch_size=512,
    verbose=1,
    callbacks=[reduce_lr, lr_scheduler]
)

# 绘制训练和验证损失
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('mse')
plt.title('loss function')
plt.legend()
plt.show()

# ===========================
# LSTM模型滚动预测
# ===========================

# 初始化输入序列，使用测试集的第一个输入序列
current_input = testX[0].copy()  # 形状为 (look_back, num_features)

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
# 填充所有特征以便逆归一化
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

aligned_utide = utide_inverse

# 计算误差
error_anomaly = testY_inverse - predictions_inverse
error_utide = testY_inverse - utide_inverse

# 在同一个图表中绘制两个误差
plt.figure(figsize=(18, 6))
plt.plot(error_anomaly, label='LSTM_error (Anomaly)', color='red')
plt.plot(error_utide, label='utide_error (Anomaly vs Utide)', color='blue')
plt.xlabel('time step')
plt.ylabel('error')
plt.title('error compare')
plt.legend()
plt.grid(True)
plt.show()

# 绘制预测和实际值进行对比
plt.figure(figsize=(18, 10))
plt.plot(testY_inverse, label="real (Anomaly)")  # 测试集实际数据
plt.plot(predictions_inverse, label="LSTM_predicted (Anomaly)", color='red', linestyle="--")  # LSTM预测数据
plt.plot(aligned_utide, label="(Utide)", color='blue', linestyle="--", linewidth=1.5)  # 谐波分析预测数据
plt.xlabel("time step")
plt.ylabel("Anomaly")
plt.title("real vs predicted vs utide")
plt.legend()
plt.show()

# ===========================
# 加权平均预测
# ===========================

# 确保谐波预测与LSTM预测对齐
# utide_inverse 已经与 rolling_predictions 对齐

# 定义加权系数
alpha = 0.5  # LSTM权重
beta = 0.5   # 谐波分析权重

# 加权平均预测
combined_predictions_inverse = alpha * predictions_inverse + beta * aligned_utide

# 计算加权平均的评估指标
rmse_combined = math.sqrt(mean_squared_error(testY_inverse, combined_predictions_inverse))
mae_combined = mean_absolute_error(testY_inverse, combined_predictions_inverse)
r2_combined = r2_score(testY_inverse, combined_predictions_inverse)

# 绘制加权平均预测和实际值进行对比
plt.figure(figsize=(18, 10))
plt.plot(testY_inverse, label="real (Anomaly)")
plt.plot(combined_predictions_inverse, label="combined (LSTM & Utide)", color='purple', linestyle="--")
plt.xlabel("time step")
plt.ylabel("Anomaly")
plt.title("combined (LSTM & Utide)")
plt.legend()
plt.show()

# 绘制加权平均预测与实际值的误差
error_combined = testY_inverse - combined_predictions_inverse

plt.figure(figsize=(18, 6))
plt.plot(error_combined, label='combined error', color='purple')
plt.xlabel('time step')
plt.ylabel('error')
plt.title('combined error')
plt.legend()
plt.grid(True)
plt.show()

# ===========================
# 计算并打印所有评估指标
# ===========================

print("各模型的误差指标：\n")

print("LSTM 模型的误差指标：")
print(f'Test RMSE: {rmse:.2f}')
print(f'Test MAE: {mae:.2f}')
print(f'Test R²: {r2:.2f}')

print("\n谐波分析的误差指标：")
harmonic_rmse = math.sqrt(mean_squared_error(testY_inverse, utide_inverse))
harmonic_mae = mean_absolute_error(testY_inverse, utide_inverse)
harmonic_r2 = r2_score(testY_inverse, utide_inverse)
print(f'Harmonic RMSE: {harmonic_rmse:.2f}')
print(f'Harmonic MAE: {harmonic_mae:.2f}')
print(f'Harmonic R²: {harmonic_r2:.2f}')

print("\n加权平均预测的误差指标：")
print(f'Combined RMSE: {rmse_combined:.2f}')
print(f'Combined MAE: {mae_combined:.2f}')
print(f'Combined R²: {r2_combined:.2f}')

# ===========================
# 绘制真实 Anomaly 与预测 Anomaly 的关系
# ===========================

# 真实 Anomaly 与 LSTM 预测 Anomaly 的关系
plt.figure(figsize=(8, 8))
plt.scatter(testY_inverse, predictions_inverse, alpha=0.5, label=f'LSTM R² = {r2:.2f}', color='red')
plt.plot([testY_inverse.min(), testY_inverse.max()], [testY_inverse.min(), testY_inverse.max()], 'r--', label='theo y=x')
plt.xlabel('real Anomaly')
plt.ylabel('prediceted Anomaly')
plt.title('real Anomaly vs LSTM predicted Anomaly 的关系')
plt.legend()
plt.grid(True)
plt.show()

# 真实 Anomaly 与谐波分析预测 Anomaly 的关系
plt.figure(figsize=(8, 8))
plt.scatter(testY_inverse, utide_inverse, alpha=0.5, label=f'Harmonic R² = {harmonic_r2:.2f}', color='blue')
plt.plot([testY_inverse.min(), testY_inverse.max()], [testY_inverse.min(), testY_inverse.max()], 'b--', label='theo y=x')
plt.xlabel('real Anomaly')
plt.ylabel('(Utide)')
plt.title('real Anomaly vs utide Anomaly')
plt.legend()
plt.grid(True)
plt.show()

# 真实 Anomaly 与加权平均预测 Anomaly 的关系
plt.figure(figsize=(8, 8))
plt.scatter(testY_inverse, combined_predictions_inverse, alpha=0.5, label=f'Combined R² = {r2_combined:.2f}', color='purple')
plt.plot([testY_inverse.min(), testY_inverse.max()], [testY_inverse.min(), testY_inverse.max()], 'g--', label='理想对角线 y=x')
plt.xlabel('Anomaly')
plt.ylabel('combined Anomaly')
plt.title('real Anomaly vs Anomaly')
plt.legend()
plt.grid(True)
plt.show()
