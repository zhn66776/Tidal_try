import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Input
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# ---------------------------
# 设置随机种子确保结果可重复
# ---------------------------
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(3407)
np.random.seed(3407)
tf.random.set_seed(3407)

# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)

# ---------------------------
# 加载数据和预处理
# ---------------------------
input_file = "dataProcessed/ABE5Y1_HA_processed.csv"
df = pd.read_csv(input_file, delimiter=',')

total_train_points = 10000
look_back_points = 100
total_test_points = 3000
start_index = 105000
utide_factor = 0.1  # 加权因子
future_steps_list = [1, 5, 10, 15, 20, 25, 30, 35, 40]  # 多步预测的步数选项

total_required = total_train_points + look_back_points + total_test_points
if len(df) < total_required + start_index:
    raise ValueError("Dataset too small.")

df_sampled = df.iloc[start_index:start_index + total_required].reset_index(drop=True)
anomaly = df_sampled['anomaly'].values.reshape(-1, 1)
utide_original = df_sampled['utide'].values.reshape(-1, 1)
utide_adjusted = utide_original * utide_factor

scaler_anomaly_train = MinMaxScaler(feature_range=(0, 1))
scaler_utide_train = MinMaxScaler(feature_range=(0, 1))
scaler_anomaly_train.fit(anomaly)
scaler_utide_train.fit(utide_adjusted)

anomaly_scaled = scaler_anomaly_train.transform(anomaly)
utide_scaled = scaler_utide_train.transform(utide_adjusted)
dataset_scaled_multi = np.hstack((anomaly_scaled, utide_scaled))

train_multi = dataset_scaled_multi[:total_train_points]
test_multi = dataset_scaled_multi[total_train_points: total_train_points + total_test_points + look_back_points]

# ---------------------------
# 数据生成函数
# ---------------------------
def create_dataset_multi_multioutput(dataset, look_back, future_steps):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - future_steps + 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        future_values = [dataset[i + look_back + j, 0] for j in range(future_steps)]
        dataY.append(future_values)
    return np.array(dataX), np.array(dataY)

def inverse_anomaly_array(arr_scaled):
    arr_scaled = arr_scaled.reshape(-1, 1)
    return scaler_anomaly_train.inverse_transform(arr_scaled).flatten()

def inverse_utide_array_factor(arr_scaled):
    arr_scaled = arr_scaled.reshape(-1, 1)
    utide_factor_domain = scaler_utide_train.inverse_transform(arr_scaled).flatten()
    return utide_factor_domain / utide_factor

# ---------------------------
# 模型构建函数
# ---------------------------
def build_bilstm_multi(input_shape, future_steps):
    input_layer = Input(shape=input_shape)
    x = Bidirectional(LSTM(200, return_sequences=True, kernel_regularizer=l2(0.0001)))(input_layer)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(200, kernel_regularizer=l2(0.0001)))(x)
    x = Dropout(0.3)(x)
    output_layer = Dense(future_steps)(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# ---------------------------
# 训练与预测
# ---------------------------
check_steps = [10, 50, 100, 200, 300, 400, 500]  # 用于打印指定步骤的r2/mse/mae

results_with_utide = {
    'future_steps': [],
    'Time_Step': [],
    'Actual_Values': [],
    'Predicted_Values': [],
    'Utide_Values': []
}

for future_steps in future_steps_list:
    trainX_multi, trainY_multi = create_dataset_multi_multioutput(train_multi, look_back_points, future_steps)
    testX_multi, testY_multi = create_dataset_multi_multioutput(test_multi, look_back_points, future_steps)

    # 数据划分
    trainX, valX, trainY, valY = train_test_split(trainX_multi, trainY_multi, test_size=0.1, random_state=3407)

    # 构建模型
    model = build_bilstm_multi((look_back_points, 2), future_steps)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=1e-6)
    epochs = 80
    batch_size = 256

    print(f"Training BiLSTM_2feat multi-output (utide_factor={utide_factor}, future_steps={future_steps})...")
    history = model.fit(
        trainX, trainY,
        validation_data=(valX, valY),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[reduce_lr]
    )
    print("Training completed.\n")
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss Curve (Future Steps: {future_steps})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 滚动预测
    current_input = test_multi[:look_back_points].copy()
    predictions_scaled = []
    actual_scaled = []
    utide_scaled_arr = []

    predicted_points = 0
    while predicted_points < total_test_points:
        input_sequence = current_input.reshape(1, look_back_points, 2)
        future_pred_scaled = model.predict(input_sequence, verbose=0)[0]  # shape: (future_steps,)

        for step_i in range(future_steps):
            if predicted_points >= total_test_points:
                break
            predictions_scaled.append(future_pred_scaled[step_i])
            actual_scaled_val = test_multi[look_back_points + predicted_points, 0]
            utide_scaled_val = test_multi[look_back_points + predicted_points, 1]
            actual_scaled.append(actual_scaled_val)
            utide_scaled_arr.append(utide_scaled_val)

            # 更新输入序列
            current_input = np.roll(current_input, -1, axis=0)
            current_input[-1, 0] = future_pred_scaled[step_i]
            if predicted_points + 1 < total_test_points:
                current_input[-1, 1] = test_multi[look_back_points + predicted_points + 1, 1]
            else:
                current_input[-1, 1] = test_multi[look_back_points + predicted_points, 1]

            predicted_points += 1

    predicted_anomaly = inverse_anomaly_array(np.array(predictions_scaled))
    actual_anomaly = inverse_anomaly_array(np.array(actual_scaled))
    actual_utide = inverse_utide_array_factor(np.array(utide_scaled_arr))

    # 计算并打印指定步数的r2/mse/mae
    print(f"\nFuture Steps = {future_steps}:")
    for s in check_steps:
        if s <= len(predicted_anomaly):
            r2_mod = r2_score(actual_anomaly[:s], predicted_anomaly[:s])
            mse_mod = mean_squared_error(actual_anomaly[:s], predicted_anomaly[:s])
            mae_mod = mean_absolute_error(actual_anomaly[:s], predicted_anomaly[:s])

            r2_ut = r2_score(actual_anomaly[:s], actual_utide[:s])
            mse_ut = mean_squared_error(actual_anomaly[:s], actual_utide[:s])
            mae_ut = mean_absolute_error(actual_anomaly[:s], actual_utide[:s])

            print(f"At step {s}:")
            print(f"Model: MSE={mse_mod:.4f}, MAE={mae_mod:.4f}, R²={r2_mod:.4f}")
            print(f"Utide: MSE={mse_ut:.4f}, MAE={mae_ut:.4f}, R²={r2_ut:.4f}")

    for i in range(len(predicted_anomaly)):
        results_with_utide['future_steps'].append(future_steps)
        results_with_utide['Time_Step'].append(i)
        results_with_utide['Actual_Values'].append(actual_anomaly[i])
        results_with_utide['Predicted_Values'].append(predicted_anomaly[i])
        results_with_utide['Utide_Values'].append(actual_utide[i])

results_df_with_utide = pd.DataFrame(results_with_utide)
print("\nResults DataFrame with utide factor scenario:")
print(results_df_with_utide.head())
