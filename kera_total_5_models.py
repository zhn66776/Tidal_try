import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Model
from keras.layers import Dense, LSTM, GRU, Dropout, Bidirectional, Input, Conv1D, MaxPooling1D
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, Callback
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import tensorflow as tf

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ---------------------------
# 数据加载与准备
# ---------------------------
input_file = "dataProcessed/ABE5Y1_HA_processed.csv"  
# 使用同一份数据来训练单特征和双特征模型，以保证可比性

df = pd.read_csv(input_file, delimiter=',')
total_train_points = 10000
look_back_points = 200
total_test_points = 2000

total_required = total_train_points + look_back_points + total_test_points
if len(df) < total_required:
    raise ValueError(f"Dataset too small. Required: {total_required}, Current: {len(df)}")

max_start_index = len(df) - total_required
start_index = np.random.randint(0, max_start_index + 1)
print(f"Selected start index: {start_index}")

df_sampled = df.iloc[start_index:start_index + total_required].reset_index(drop=True)

anomaly = df_sampled['anomaly'].values.reshape(-1, 1)
utide = df_sampled['utide'].values.reshape(-1, 1)

scaler_anomaly = MinMaxScaler(feature_range=(0, 1))
anomaly_scaled = scaler_anomaly.fit_transform(anomaly)

scaler_utide = MinMaxScaler(feature_range=(0, 1))
utide_scaled = scaler_utide.fit_transform(utide)

# 单特征数据集（仅anomaly）
dataset_scaled_single = anomaly_scaled  
train_single = dataset_scaled_single[:total_train_points]
test_single = dataset_scaled_single[total_train_points : total_train_points + total_test_points + look_back_points]

# 双特征数据集（anomaly + utide）
dataset_scaled_multi = np.hstack((anomaly_scaled, utide_scaled))
train_multi = dataset_scaled_multi[:total_train_points]
test_multi = dataset_scaled_multi[total_train_points : total_train_points + total_test_points + look_back_points]

def create_dataset_single(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i+look_back])
    return np.array(dataX), np.array(dataY)

def create_dataset_multi(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i+look_back, 0]) # 预测anomaly
    return np.array(dataX), np.array(dataY)

# 创建单特征的train/test
trainX_single, trainY_single = create_dataset_single(train_single, look_back_points)
testX_single, testY_single = create_dataset_single(test_single, look_back_points)

# 创建双特征的train/test
trainX_multi, trainY_multi = create_dataset_multi(train_multi, look_back_points)
testX_multi, testY_multi = create_dataset_multi(test_multi, look_back_points)

print("Single-feature TrainX:", trainX_single.shape, "TrainY:", trainY_single.shape)
print("Single-feature TestX:", testX_single.shape, "TestY:", testY_single.shape)
print("Multi-feature TrainX:", trainX_multi.shape, "TrainY:", trainY_multi.shape)
print("Multi-feature TestX:", testX_multi.shape, "TestY:", testY_multi.shape)

num_features_single = 1
num_features_multi = 2

# ---------------------------------------
# 定义模型
# ---------------------------------------
def build_lstm_single(input_shape):
    input_layer = Input(shape=input_shape)
    x = LSTM(300, return_sequences=True, kernel_regularizer=l2(0.0001))(input_layer)
    x = Dropout(0.3)(x)
    x = LSTM(300, kernel_regularizer=l2(0.0001))(x)
    x = Dropout(0.3)(x)
    output_layer = Dense(1)(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

def build_bilstm_single(input_shape):
    input_layer = Input(shape=input_shape)
    x = Bidirectional(LSTM(300, return_sequences=True, kernel_regularizer=l2(0.0001)))(input_layer)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(300, kernel_regularizer=l2(0.0001)))(x)
    x = Dropout(0.3)(x)
    output_layer = Dense(1)(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

def build_gru_single(input_shape):
    input_layer = Input(shape=input_shape)
    x = Bidirectional(GRU(250, return_sequences=True, kernel_regularizer=l2(0.0001)))(input_layer)
    x = Dropout(0.3)(x)
    x = Bidirectional(GRU(250, kernel_regularizer=l2(0.0001)))(x)
    x = Dropout(0.3)(x)
    output_layer = Dense(1)(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

def build_bilstm_multi(input_shape):
    input_layer = Input(shape=input_shape)
    x = Bidirectional(LSTM(300, return_sequences=True, kernel_regularizer=l2(0.0001)))(input_layer)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(300, kernel_regularizer=l2(0.0001)))(x)
    x = Dropout(0.3)(x)
    output_layer = Dense(1)(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

def build_gru_multi(input_shape):
    input_layer = Input(shape=input_shape)
    x = Bidirectional(GRU(250, return_sequences=True, kernel_regularizer=l2(0.0001)))(input_layer)
    x = Dropout(0.4)(x)
    x = Bidirectional(GRU(250, kernel_regularizer=l2(0.0001)))(x)
    x = Dropout(0.4)(x)
    output_layer = Dense(1)(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# ---------------------------------------
# 定义滚动预测函数
# ---------------------------------------
def rolling_prediction_single_fixed(model, test_multi, scaler_anomaly, scaler_utide, look_back):
    total_tp = len(test_multi) - look_back
    current_input = test_multi[:look_back,0].reshape(-1,1)
    rolling_predictions = []
    actual_values = []
    utide_values = []
    r2_scores = []
    utide_r2_scores = []

    for i in range(total_tp):
        input_sequence = current_input.reshape(1, look_back, 1)
        prediction = model.predict(input_sequence, verbose=0)
        # 对预测值进行inverse_transform
        predicted_anomaly_scaled = prediction[0,0]
        predicted_anomaly = scaler_anomaly.inverse_transform([[predicted_anomaly_scaled]])[0,0]
        
        rolling_predictions.append(predicted_anomaly)

        actual_anomaly_scaled = test_multi[look_back + i, 0]
        actual_anomaly = scaler_anomaly.inverse_transform([[actual_anomaly_scaled]])[0, 0]
        actual_values.append(actual_anomaly)

        utide_scaled_val = test_multi[look_back + i, 1]
        utide_val = scaler_utide.inverse_transform([[utide_scaled_val]])[0, 0]
        utide_values.append(utide_val)

        if i >= 20:
            r2_scores.append(r2_score(actual_values, rolling_predictions))
            utide_r2_scores.append(r2_score(actual_values, utide_values))
        else:
            r2_scores.append(np.nan)
            utide_r2_scores.append(np.nan)

        # current_input用scaled值更新
        current_input = np.roll(current_input, -1, axis=0)
        current_input[-1,0] = predicted_anomaly_scaled

    return {
        'actual_values': np.array(actual_values),
        'predicted_values': np.array(rolling_predictions),
        'utide_values': np.array(utide_values),
        'r2_scores': np.array(r2_scores),
        'utide_r2_scores': np.array(utide_r2_scores)
    }

def rolling_prediction_multi(model, test, scaler_anomaly, scaler_utide, look_back):
    total_tp = len(test) - look_back
    current_input = test[:look_back].copy()
    rolling_predictions = []
    actual_values = []
    utide_values = []
    r2_scores = []
    utide_r2_scores = []

    for i in range(total_tp):
        input_sequence = current_input.reshape(1, look_back, 2)
        prediction = model.predict(input_sequence, verbose=0)
        predicted_anomaly_scaled = prediction[0, 0]

        actual_anomaly_scaled = test[look_back + i, 0]
        actual_anomaly = scaler_anomaly.inverse_transform([[actual_anomaly_scaled]])[0, 0]
        actual_values.append(actual_anomaly)

        utide_scaled_val = test[look_back + i, 1]
        utide_val = scaler_utide.inverse_transform([[utide_scaled_val]])[0, 0]
        utide_values.append(utide_val)

        predicted_anomaly = scaler_anomaly.inverse_transform([[predicted_anomaly_scaled]])[0,0]
        rolling_predictions.append(predicted_anomaly)

        current_input = np.roll(current_input, -1, axis=0)
        if i < total_tp - 1:
            next_utide_scaled = test[look_back + i + 1, 1]
        else:
            next_utide_scaled = test[look_back + i, 1]

        current_input[-1, 0] = predicted_anomaly_scaled
        current_input[-1, 1] = next_utide_scaled

        if i >= 20:
            r2_scores.append(r2_score(actual_values, rolling_predictions))
            utide_r2_scores.append(r2_score(actual_values, utide_values))
        else:
            r2_scores.append(np.nan)
            utide_r2_scores.append(np.nan)

    return {
        'actual_values': np.array(actual_values),
        'predicted_values': np.array(rolling_predictions),
        'utide_values': np.array(utide_values),
        'r2_scores': np.array(r2_scores),
        'utide_r2_scores': np.array(utide_r2_scores)
    }

# ---------------------------------------
# 定义训练和预测的模型集合
# 单特征模型： LSTM_single, BiLSTM_single, GRU_single
# 双特征模型： BiLSTM_2feat, GRU_2feat
# ---------------------------------------
single_feature_models = {
    'LSTM_single': build_lstm_single((look_back_points, num_features_single)),
    'BiLSTM_single': build_bilstm_single((look_back_points, num_features_single)),
    'GRU_single': build_gru_single((look_back_points, num_features_single))
}

multi_feature_models = {
    'BiLSTM_2feat': build_bilstm_multi((look_back_points, num_features_multi)),
    'GRU_2feat': build_gru_multi((look_back_points, num_features_multi))
}

all_models = {}
all_models.update(single_feature_models)
all_models.update(multi_feature_models)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=1e-6)

epochs = 150
batch_size = 256

histories = {}

# 训练所有模型并记录history
for name, model in all_models.items():
    print(f"Training {name}...")
    if '2feat' in name:
        history = model.fit(trainX_multi, trainY_multi, 
                            validation_split=0.1, 
                            epochs=epochs, 
                            batch_size=batch_size, 
                            verbose=1, 
                            callbacks=[reduce_lr])
    else:
        history = model.fit(trainX_single, trainY_single, 
                            validation_split=0.1, 
                            epochs=epochs, 
                            batch_size=batch_size, 
                            verbose=1, 
                            callbacks=[reduce_lr])
    histories[name] = history
    print(f"{name} training completed.\n")

# 绘制每个模型的loss曲线
plt.figure(figsize=(12,8))
for name, history in histories.items():
    plt.plot(history.history['loss'], label=f'{name} Train Loss')
    plt.plot(history.history['val_loss'], label=f'{name} Val Loss')
plt.title('Model Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 滚动预测
results = {}
for name, model in all_models.items():
    print(f"Performing rolling prediction with {name}...")
    if '2feat' in name:
        res = rolling_prediction_multi(model, test_multi, scaler_anomaly, scaler_utide, look_back_points)
    else:
        res = rolling_prediction_single_fixed(model, test_multi, scaler_anomaly, scaler_utide, look_back_points)
    results[name] = res
    print(f"{name} rolling prediction completed.\n")

# 实际与utide基准值
actual_values_full = scaler_anomaly.inverse_transform(test_multi[look_back_points:look_back_points + total_test_points, 0].reshape(-1,1)).flatten()
utide_values_full = scaler_utide.inverse_transform(test_multi[look_back_points:look_back_points + total_test_points, 1].reshape(-1,1)).flatten()
r2_utide = r2_score(actual_values_full, utide_values_full)

# 计算MSE、MAE对比
steps_to_plot = 100
x = np.arange(steps_to_plot)
model_names = list(results.keys())
width = 0.1

mse_dict = {}
mae_dict = {}

for name in model_names:
    actual = results[name]['actual_values'][:steps_to_plot]
    predicted = results[name]['predicted_values'][:steps_to_plot]
    mse = (actual - predicted)**2
    mae = np.abs(actual - predicted)
    mse_dict[name] = mse
    mae_dict[name] = mae

mse_utide = (actual_values_full[:steps_to_plot] - utide_values_full[:steps_to_plot])**2
mae_utide = np.abs(actual_values_full[:steps_to_plot] - utide_values_full[:steps_to_plot])

# 绘制MSE与MAE对比直方图
fig, axes = plt.subplots(1,2, figsize=(20,8))

# MSE对比
for idx, name in enumerate(model_names):
    axes[0].bar(x + idx*width, mse_dict[name], width, label=f'{name} MSE')
axes[0].bar(x + len(model_names)*width, mse_utide, width, label='Utide MSE', color='grey')
axes[0].set_title('MSE Comparison (First 100 Steps)')
axes[0].set_xlabel('Time Step')
axes[0].set_ylabel('MSE')
axes[0].legend()
axes[0].set_xticks(x)

for i in range(steps_to_plot):
    errors = [mse_dict[m][i] for m in model_names] + [mse_utide[i]]
    min_idx = np.argmin(errors)
    axes[0].text(i + min_idx*width, errors[min_idx] + 0.001, '*', ha='center', va='bottom', color='black')

# MAE对比
for idx, name in enumerate(model_names):
    axes[1].bar(x + idx*width, mae_dict[name], width, label=f'{name} MAE')
axes[1].bar(x + len(model_names)*width, mae_utide, width, label='Utide MAE', color='grey')
axes[1].set_title('MAE Comparison (First 100 Steps)')
axes[1].set_xlabel('Time Step')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].set_xticks(x)

for i in range(steps_to_plot):
    errors = [mae_dict[m][i] for m in model_names] + [mae_utide[i]]
    min_idx = np.argmin(errors)
    axes[1].text(i + min_idx*width, errors[min_idx] + 0.001, '*', ha='center', va='bottom', color='black')

plt.tight_layout()
plt.show()

# 打印R²何时低于Utide
start_index_plot = 0
n_steps_plot = 300
indices = np.arange(start_index_plot, min(start_index_plot + n_steps_plot, total_test_points))

for name in model_names:
    model_r2 = results[name]['r2_scores'][indices]
    falloff_indices = np.where(model_r2 < r2_utide)[0]
    first_falloff = falloff_indices[0] if falloff_indices.size > 0 else "Never"
    print(f"{name} R² falls below Utide at step: {first_falloff}")

# 输出预测对比图
plt.figure(figsize=(20,10))
plt.plot(actual_values_full[:steps_to_plot], label='Actual', color='black')
plt.plot(utide_values_full[:steps_to_plot], label='Utide', color='green', linestyle='--')

colors = ['blue', 'orange', 'red', 'purple', 'brown']
for idx, name in enumerate(model_names):
    predicted = results[name]['predicted_values'][:steps_to_plot]
    plt.plot(predicted, label=f'{name} Prediction', color=colors[idx % len(colors)])

plt.xlabel('Time Step')
plt.ylabel('Anomaly')
plt.title('Predictions vs Actual (First 100 Steps)')
plt.legend()
plt.show()
