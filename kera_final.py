import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Model
from keras.layers import Dense, LSTM, GRU, Dropout, Bidirectional, Input
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import tensorflow as tf

# 可根据需要设定随机种子
# np.random.seed(42)
# tf.random.set_seed(42)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ---------------------------
# 数据加载与准备
# ---------------------------
input_file = "dataProcessed/ABE5Y1_HA_processed.csv"
df = pd.read_csv(input_file, delimiter=',')
total_train_points = 10000
look_back_points = 100
total_test_points = 3000

total_required = total_train_points + look_back_points + total_test_points
if len(df) < total_required:
    raise ValueError(f"Dataset too small. Required: {total_required}, Current: {len(df)}")

# 指定固定的起始位置（例如从索引 5000 开始）
start_index = 95000

if start_index + total_required > len(df):
    raise ValueError(f"Specified start_index={start_index} is too large. Not enough data points.")

print(f"Selected start index: {start_index}")

# 从指定的起始点选取数据
df_sampled = df.iloc[start_index:start_index + total_required].reset_index(drop=True)

df_sampled = df.iloc[start_index:start_index + total_required].reset_index(drop=True)

anomaly = df_sampled['anomaly'].values.reshape(-1, 1)
utide_original = df_sampled['utide'].values.reshape(-1, 1)

# 定义utide_factor来调整utide权重（仅在训练和预测输入阶段有影响）
utide_factor = 0.2
utide_adjusted = utide_original * utide_factor

# 为训练数据分开对anomaly和factor*utide进行缩放
scaler_anomaly_train = MinMaxScaler(feature_range=(0,1))
scaler_utide_train = MinMaxScaler(feature_range=(0,1))

# 拟合scaler
scaler_anomaly_train.fit(anomaly)  # 使用原始anomaly拟合，不改变anomaly的真实尺度
scaler_utide_train.fit(utide_adjusted)  # 使用factor后的utide拟合

anomaly_scaled = scaler_anomaly_train.transform(anomaly)
utide_scaled = scaler_utide_train.transform(utide_adjusted)

# 单特征数据集（仅 anomaly）
dataset_scaled_single = anomaly_scaled
train_single = dataset_scaled_single[:total_train_points]
test_single = dataset_scaled_single[total_train_points : total_train_points + total_test_points + look_back_points]

# 双特征数据集（anomaly + factor*utide）
# 注意这里组合的是经过独立缩放的anomaly_scaled与utide_scaled
dataset_scaled_multi = np.hstack((anomaly_scaled, utide_scaled))
train_multi = dataset_scaled_multi[:total_train_points]
test_multi = dataset_scaled_multi[total_train_points : total_train_points + total_test_points + look_back_points]

def create_dataset_single(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i+look_back])  # predict anomaly
    return np.array(dataX), np.array(dataY)

def create_dataset_multi(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i+look_back, 0]) # 预测 anomaly
    return np.array(dataX), np.array(dataY)

trainX_single, trainY_single = create_dataset_single(train_single, look_back_points)
testX_single, testY_single = create_dataset_single(test_single, look_back_points)

trainX_multi, trainY_multi = create_dataset_multi(train_multi, look_back_points)
testX_multi, testY_multi = create_dataset_multi(test_multi, look_back_points)

print("Single-feature TrainX:", trainX_single.shape, "TrainY:", trainY_single.shape)
print("Single-feature TestX:", testX_single.shape, "TestY:", testY_single.shape)
print("Multi-feature TrainX:", trainX_multi.shape, "TrainY:", trainY_multi.shape)
print("Multi-feature TestX:", testX_multi.shape, "TestY:", testY_multi.shape)

num_features_single = 1
num_features_multi = 2

# 定义逆归一化函数，用于 anomaly 和 utide 的预测值和实际值还原
def inverse_anomaly_array(arr_scaled):
    # arr_scaled为N个预测值或实际值的列
    arr_scaled = arr_scaled.reshape(-1,1)
    return scaler_anomaly_train.inverse_transform(arr_scaled).flatten()

def inverse_utide_array_factor(arr_scaled):
    # arr_scaled为factor后缩放的utide值
    arr_scaled = arr_scaled.reshape(-1,1)
    utide_factor_domain = scaler_utide_train.inverse_transform(arr_scaled).flatten()  # 得到factor*utide_original
    # 除以factor还原到原始utide
    return utide_factor_domain / utide_factor

# ---------------------------------------
# 定义模型
# ---------------------------------------
def build_lstm_single(input_shape):
    input_layer = Input(shape=input_shape)
    x = LSTM(500, return_sequences=True, kernel_regularizer=l2(0.0001))(input_layer)
    x = Dropout(0.4)(x)
    x = LSTM(500, kernel_regularizer=l2(0.0001))(x)
    x = Dropout(0.4)(x)
    output_layer = Dense(1)(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

def build_bilstm_single(input_shape):
    input_layer = Input(shape=input_shape)
    x = Bidirectional(LSTM(700, return_sequences=True, kernel_regularizer=l2(0.0001)))(input_layer)
    x = Dropout(0.4)(x)
    x = Bidirectional(LSTM(700, kernel_regularizer=l2(0.0001)))(x)
    x = Dropout(0.4)(x)
    output_layer = Dense(1)(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

def build_gru_single(input_shape):
    input_layer = Input(shape=input_shape)
    x = Bidirectional(GRU(700, return_sequences=True, kernel_regularizer=l2(0.0001)))(input_layer)
    x = Dropout(0.4)(x)
    x = Bidirectional(GRU(700, kernel_regularizer=l2(0.0001)))(x)
    x = Dropout(0.4)(x)
    output_layer = Dense(1)(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

def build_bilstm_multi(input_shape):
    input_layer = Input(shape=input_shape)
    x = Bidirectional(LSTM(700, return_sequences=True, kernel_regularizer=l2(0.0001)))(input_layer)
    x = Dropout(0.4)(x)
    x = Bidirectional(LSTM(700, kernel_regularizer=l2(0.0001)))(x)
    x = Dropout(0.4)(x)
    output_layer = Dense(1)(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

def build_gru_multi(input_shape):
    input_layer = Input(shape=input_shape)
    x = Bidirectional(GRU(700, return_sequences=True, kernel_regularizer=l2(0.0001)))(input_layer)
    x = Dropout(0.4)(x)
    x = Bidirectional(GRU(700, kernel_regularizer=l2(0.0001)))(x)
    x = Dropout(0.4)(x)
    output_layer = Dense(1)(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# ---------------------------------------
# 滚动预测函数
# ---------------------------------------
def rolling_prediction_single_fixed(model, test_multi, look_back):
    total_tp = len(test_multi) - look_back
    current_input = test_multi[:look_back,0].reshape(-1,1)
    predictions_scaled = []
    actual_scaled = []
    utide_scaled_arr = []

    for i in range(total_tp):
        input_sequence = current_input.reshape(1, look_back, 1)
        prediction = model.predict(input_sequence, verbose=0)[0,0]  # scaled anomaly
        predictions_scaled.append(prediction)

        actual_scaled_val = test_multi[look_back + i, 0] # scaled anomaly
        utide_scaled_val = test_multi[look_back + i, 1]  # scaled factor*utide

        actual_scaled.append(actual_scaled_val)
        utide_scaled_arr.append(utide_scaled_val)

        current_input = np.roll(current_input, -1, axis=0)
        current_input[-1,0] = prediction

    # 逆归一化
    # anomaly
    predicted_anomaly = inverse_anomaly_array(np.array(predictions_scaled))
    actual_anomaly = inverse_anomaly_array(np.array(actual_scaled))
    # utide factor domain to original
    actual_utide = inverse_utide_array_factor(np.array(utide_scaled_arr))

    # 计算r2_scores
    r2_scores = []
    utide_r2_scores = []
    for i in range(len(predicted_anomaly)):
        if i >= 20:
            r2_scores.append(r2_score(actual_anomaly[:i+1], predicted_anomaly[:i+1]))
            utide_r2_scores.append(r2_score(actual_anomaly[:i+1], actual_utide[:i+1]))
        else:
            r2_scores.append(np.nan)
            utide_r2_scores.append(np.nan)

    return {
        'actual_values': actual_anomaly,
        'predicted_values': predicted_anomaly,
        'utide_values': actual_utide,
        'r2_scores': np.array(r2_scores),
        'utide_r2_scores': np.array(utide_r2_scores)
    }

def rolling_prediction_multi(model, test, look_back):
    total_tp = len(test) - look_back
    current_input = test[:look_back].copy()
    predictions_scaled = []
    actual_scaled = []
    utide_scaled_arr = []

    for i in range(total_tp):
        input_sequence = current_input.reshape(1, look_back, 2)
        prediction = model.predict(input_sequence, verbose=0)[0,0] # scaled anomaly
        predictions_scaled.append(prediction)

        actual_scaled_val = test[look_back + i, 0]
        utide_scaled_val = test[look_back + i, 1]

        actual_scaled.append(actual_scaled_val)
        utide_scaled_arr.append(utide_scaled_val)

        current_input = np.roll(current_input, -1, axis=0)
        current_input[-1,0] = prediction
        if i < total_tp - 1:
            next_utide_scaled = test[look_back + i + 1, 1]
        else:
            next_utide_scaled = test[look_back + i, 1]
        current_input[-1,1] = next_utide_scaled

    # 逆归一化 anomaly和utide
    predicted_anomaly = inverse_anomaly_array(np.array(predictions_scaled))
    actual_anomaly = inverse_anomaly_array(np.array(actual_scaled))
    actual_utide = inverse_utide_array_factor(np.array(utide_scaled_arr))

    # 计算r2_scores
    r2_scores = []
    utide_r2_scores = []
    for i in range(len(predicted_anomaly)):
        if i >= 20:
            r2_scores.append(r2_score(actual_anomaly[:i+1], predicted_anomaly[:i+1]))
            utide_r2_scores.append(r2_score(actual_anomaly[:i+1], actual_utide[:i+1]))
        else:
            r2_scores.append(np.nan)
            utide_r2_scores.append(np.nan)

    return {
        'actual_values': actual_anomaly,
        'predicted_values': predicted_anomaly,
        'utide_values': actual_utide,
        'r2_scores': np.array(r2_scores),
        'utide_r2_scores': np.array(utide_r2_scores)
    }

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

epochs = 40
batch_size = 128

histories = {}

for name, model in all_models.items():
    print(f"Training {name} (utide_factor={utide_factor})...")
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

plt.figure(figsize=(12,8))
for name, history in histories.items():
    plt.plot(history.history['loss'], label=f'{name} Train Loss')
    plt.plot(history.history['val_loss'], label=f'{name} Val Loss')
plt.title(f'Model Loss Comparison (utide_factor={utide_factor})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

results = {}
for name, model in all_models.items():
    print(f"Performing rolling prediction with {name}...")
    if '2feat' in name:
        res = rolling_prediction_multi(model, test_multi, look_back_points)
    else:
        res = rolling_prediction_single_fixed(model, test_multi, look_back_points)
    results[name] = res
    print(f"{name} rolling prediction completed.\n")

actual_values_full = df_sampled['anomaly'][total_train_points+look_back_points : total_train_points+look_back_points + total_test_points].values
utide_values_full = df_sampled['utide'][total_train_points+look_back_points : total_train_points+look_back_points + total_test_points].values
r2_utide = r2_score(actual_values_full, utide_values_full)

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

fig, axes = plt.subplots(1,2, figsize=(20,8))

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

start_index_plot = 0
n_steps_plot = 300
indices = np.arange(start_index_plot, min(start_index_plot + n_steps_plot, total_test_points))

for name in model_names:
    model_r2 = []
    model_pred = results[name]['predicted_values']
    act = results[name]['actual_values']
    for i in range(len(model_pred)):
        if i < 20:
            model_r2.append(np.nan)
        else:
            model_r2.append(r2_score(act[:i+1], model_pred[:i+1]))
    model_r2 = np.array(model_r2)[indices]
    falloff_indices = np.where(model_r2 < r2_utide)[0]
    first_falloff = falloff_indices[0] if falloff_indices.size > 0 else "Never"
    print(f"{name} R² falls below Utide at step: {first_falloff}")

plt.figure(figsize=(20,10))
plt.plot(actual_values_full[:steps_to_plot], label='Actual', color='black')
plt.plot(utide_values_full[:steps_to_plot], label='Utide', color='green', linestyle='--')

colors = ['blue', 'orange', 'red', 'purple', 'brown']
for idx, name in enumerate(model_names):
    predicted = results[name]['predicted_values'][:steps_to_plot]
    plt.plot(predicted, label=f'{name} Prediction', color=colors[idx % len(colors)])

plt.xlabel('Time Step')
plt.ylabel('Anomaly')
plt.title(f'Predictions vs Actual (First 100 Steps, utide_factor={utide_factor})')
plt.legend()
plt.show()

########################################
# 新增的代码段：分别对前50步和前100步计算MSE、MAE、R²（包含Utide）
# 绘制6个直方图：前50步和前100步的MSE、MAE、R²(每种3个图)
# Utide为第一个列，与模型数据对齐
########################################

def compute_metrics_for_steps(steps, results, actual_values_full, utide_values_full):
    # 计算utide在前steps步的MSE、MAE、R²（原始值，不经factor）
    utide_mse_val = mean_squared_error(actual_values_full[:steps], utide_values_full[:steps])
    utide_mae_val = mean_absolute_error(actual_values_full[:steps], utide_values_full[:steps])
    utide_r2_val = r2_score(actual_values_full[:steps], utide_values_full[:steps])

    mse_vals = [utide_mse_val]
    mae_vals = [utide_mae_val]
    r2_vals = [utide_r2_val]

    for name in model_names:
        actual = results[name]['actual_values'][:steps]
        pred = results[name]['predicted_values'][:steps]
        mse_val = mean_squared_error(actual, pred)
        mae_val = mean_absolute_error(actual, pred)
        r2_val = r2_score(actual, pred)
        mse_vals.append(mse_val)
        mae_vals.append(mae_val)
        r2_vals.append(r2_val)

    labels = ['Utide'] + model_names
    return labels, mse_vals, mae_vals, r2_vals

steps_list = [50, 100]
for steps_ in steps_list:
    labels, mse_vals, mae_vals, r2_vals = compute_metrics_for_steps(steps_, results, actual_values_full, utide_values_full)

    x = np.arange(len(labels))
    width = 0.5

    # MSE图（steps_）
    plt.figure(figsize=(8,6))
    plt.bar(x, mse_vals, width, color='blue')
    plt.title(f'MSE for First {steps_} Steps (utide_factor={utide_factor})')
    plt.xlabel('Utide/Model')
    plt.ylabel('MSE')
    plt.xticks(x, labels, rotation=45)
    plt.tight_layout()
    plt.show()

    # MAE图（steps_）
    plt.figure(figsize=(8,6))
    plt.bar(x, mae_vals, width, color='orange')
    plt.title(f'MAE for First {steps_} Steps (utide_factor={utide_factor})')
    plt.xlabel('Utide/Model')
    plt.ylabel('MAE')
    plt.xticks(x, labels, rotation=45)
    plt.tight_layout()
    plt.show()

    # R²图（steps_）
    plt.figure(figsize=(8,6))
    plt.bar(x, r2_vals, width, color='red')
    plt.title(f'R² for First {steps_} Steps (utide_factor={utide_factor})')
    plt.xlabel('Utide/Model')
    plt.ylabel('R²')
    plt.xticks(x, labels, rotation=45)
    plt.tight_layout()
    plt.show()
