import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Input
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------------------
# 设置随机种子，确保结果可重复
# ---------------------------
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(3407)
np.random.seed(3407)
tf.random.set_seed(3407)
# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)

# ---------------------------
# 加载数据
# ---------------------------
input_file = "dataProcessed/ABE2018_HA1.csv"
df = pd.read_csv(input_file, delimiter=',')

total_train_points = 31964
look_back_points = 100
total_test_points = 2976
start_index = 0
future_steps_list = [25]

# 数据完整性检查
total_required = total_train_points + look_back_points + total_test_points
if len(df) < total_required + start_index:
    raise ValueError("Dataset too small.")

df_sampled = df.iloc[start_index:start_index + total_required].reset_index(drop=True)

# 数据预处理
anomaly = df_sampled['anomaly'].values.reshape(-1, 1)
utide_original = df_sampled['tide_h'].values.reshape(-1, 1)  # 用于后续比较baseline

scaler_anomaly_train = MinMaxScaler(feature_range=(0, 1))
scaler_anomaly_train.fit(anomaly)
anomaly_scaled = scaler_anomaly_train.transform(anomaly)

train_single = anomaly_scaled[:total_train_points]
test_single = anomaly_scaled[total_train_points: total_train_points + total_test_points + look_back_points]

# ---------------------------
# 数据生成函数
# ---------------------------
def create_dataset_single_multioutput(dataset, look_back, future_steps):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - future_steps + 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        future_values = [dataset[i + look_back + j, 0] if dataset.ndim > 1 else dataset[i + look_back + j] for j in range(future_steps)]
        dataY.append(future_values)
    return np.array(dataX), np.array(dataY)

def inverse_anomaly_array(arr_scaled):
    arr_scaled = arr_scaled.reshape(-1, 1)
    return scaler_anomaly_train.inverse_transform(arr_scaled).flatten()

# ---------------------------
# 模型构建函数
# ---------------------------
def build_bilstm_single(input_shape, future_steps):
    input_layer = Input(shape=input_shape)
    x = Bidirectional(LSTM(100, kernel_regularizer=l2(0.0001)))(input_layer)
    x = Dropout(0.3)(x)
    # x = Bidirectional(LSTM(100, return_sequences=True, kernel_regularizer=l2(0.0001)))(x)
    # x = Dropout(0.3)(x)
    # # 注意：这里原代码可能有逻辑问题，两次从 input_layer 输入LSTM。但用户要求不改模型结构，所以保持原样。
    # x = Bidirectional(LSTM(100, kernel_regularizer=l2(0.0001)))(x)
    # x = Dropout(0.3)(x)
    output_layer = Dense(future_steps)(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# ---------------------------
# 训练与预测
# ---------------------------
results_no_utide = {
    'future_steps': [],
    'Time_Step': [],
    'Actual_Values': [],
    'Predicted_Values': [],
    'Utide_Values': []
}

for future_steps in future_steps_list:
    trainX_single, trainY_single = create_dataset_single_multioutput(train_single, look_back_points, future_steps)
    testX_single, testY_single = create_dataset_single_multioutput(test_single, look_back_points, future_steps)

    # 数据划分
    trainX, valX, trainY, valY = train_test_split(trainX_single, trainY_single, test_size=0.1, random_state=3407)

    # 模型构建
    model = build_bilstm_single((look_back_points, 1), future_steps)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=1e-6)
    epochs = 80
    batch_size = 64

    print(f"Training BiLSTM_single multi-output (no utide) future_steps={future_steps}...")
    history = model.fit(
        trainX, trainY,
        validation_data=(valX, valY),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[reduce_lr]
    )
    print("Training completed.\n")

    # 滚动预测
    current_input = test_single[:look_back_points].copy()
    predictions_scaled = []
    actual_scaled = []
    predicted_points = 0
    while predicted_points < total_test_points:
        input_sequence = current_input.reshape(1, look_back_points, 1)
        future_pred_scaled = model.predict(input_sequence, verbose=0)[0]  # shape: (future_steps,)

        for step_i in range(future_steps):
            if predicted_points >= total_test_points:
                break
            predictions_scaled.append(future_pred_scaled[step_i])
            actual_scaled_val = test_single[look_back_points + predicted_points, 0]
            actual_scaled.append(actual_scaled_val)
            current_input = np.roll(current_input, -1)
            current_input[-1] = future_pred_scaled[step_i]
            predicted_points += 1

    # 数据反归一化
    predicted_anomaly = inverse_anomaly_array(np.array(predictions_scaled))
    actual_anomaly = inverse_anomaly_array(np.array(actual_scaled))
    actual_utide = df_sampled['tide_h'][total_train_points + look_back_points: total_train_points + look_back_points + total_test_points].values

    # 前1000步对比图和残差图
    residuals_predicted = actual_anomaly[:1000] - predicted_anomaly[:1000]
    residuals_utide = actual_anomaly[:1000] - actual_utide[:1000]

    # 对比图
    plt.figure(figsize=(12, 6))
    plt.plot(actual_anomaly[:1000], label='actual', color='black')
    plt.plot(predicted_anomaly[:1000], label='predicted', color='red', linestyle='--')
    plt.plot(actual_utide[:1000], label='utide', color='green', linestyle='--')
    plt.title(f'Future Steps: {future_steps}')
    plt.xlabel('Time Step')
    plt.ylabel('water level')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.legend()
    plt.grid(True)
    plt.show()

    # 残差图
    plt.figure(figsize=(12, 6))
    plt.plot(residuals_predicted, label='Residuals (Actual - Predicted)', color='blue')
    plt.plot(residuals_utide, label='Residuals (Actual - Utide)', color='green', linestyle='--')
    plt.title(f'Residuals (Future Steps: {future_steps})')
    plt.xlabel('Time Step')
    plt.ylabel('Residuals')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.legend()
    plt.grid(True)
    plt.show()

    # 每100步计算一次R²、MSE、MAE
    num_segments = 10
    segment_length = 100
    segment_labels = []
    r2_pred_list, mse_pred_list, mae_pred_list = [], [], []
    r2_utide_list, mse_utide_list, mae_utide_list = [], [], []

    for seg_i in range(num_segments):
        start = seg_i * segment_length
        end = start + segment_length

        seg_actual = actual_anomaly[start:end]
        seg_pred = predicted_anomaly[start:end]
        seg_utide_val = actual_utide[start:end]

        # 计算Predicted相对于Actual的指标
        r2_pred = r2_score(seg_actual, seg_pred)
        mse_pred = mean_squared_error(seg_actual, seg_pred)
        mae_pred = mean_absolute_error(seg_actual, seg_pred)

        # 计算Utide相对于Actual的指标
        r2_utide = r2_score(seg_actual, seg_utide_val)
        mse_utide = mean_squared_error(seg_actual, seg_utide_val)
        mae_utide = mean_absolute_error(seg_actual, seg_utide_val)

        r2_pred_list.append(r2_pred)
        mse_pred_list.append(mse_pred)
        mae_pred_list.append(mae_pred)

        r2_utide_list.append(r2_utide)
        mse_utide_list.append(mse_utide)
        mae_utide_list.append(mae_utide)

        segment_labels.append(f'{start}-{end}')

    x = np.arange(num_segments)

    # R²图
    plt.figure(figsize=(12,6))
    plt.plot(x, r2_pred_list, marker='o', color='red', label='Predicted R²')
    plt.plot(x, r2_utide_list, marker='o', color='green', label='Utide R²')
    plt.title(f'R² per 100-step segment (Future Steps: {future_steps})')
    plt.xticks(x, segment_labels, rotation=45)
    plt.ylabel('R²')
    plt.grid(True, linestyle='--', linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # MSE图
    plt.figure(figsize=(12,6))
    plt.plot(x, mse_pred_list, marker='o', color='red', label='Predicted MSE')
    plt.plot(x, mse_utide_list, marker='o', color='green', label='Utide MSE')
    plt.title(f'MSE per 100-step segment (Future Steps: {future_steps})')
    plt.xticks(x, segment_labels, rotation=45)
    plt.ylabel('MSE')
    plt.grid(True, linestyle='--', linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # MAE图
    plt.figure(figsize=(12,6))
    plt.plot(x, mae_pred_list, marker='o', color='red', label='Predicted MAE')
    plt.plot(x, mae_utide_list, marker='o', color='green', label='Utide MAE')
    plt.title(f'MAE per 100-step segment (Future Steps: {future_steps})')
    plt.xticks(x, segment_labels, rotation=45)
    plt.ylabel('MAE')
    plt.grid(True, linestyle='--', linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 保存结果
    for i in range(len(predicted_anomaly)):
        results_no_utide['future_steps'].append(future_steps)
        results_no_utide['Time_Step'].append(i)
        results_no_utide['Actual_Values'].append(actual_anomaly[i])
        results_no_utide['Predicted_Values'].append(predicted_anomaly[i])
        results_no_utide['Utide_Values'].append(actual_utide[i])

results_df_no_utide = pd.DataFrame(results_no_utide)
print("\nResults DataFrame no utide scenario:")
print(results_df_no_utide.head())
