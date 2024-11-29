import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Input
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import time


np.random.seed(5)
input_file = "dataProcessed/ABE5Y1_HA_processed.csv"
df = read_csv(input_file, delimiter=',')
feature_columns = ['anomaly', 'utide']
dataset = df[feature_columns].values
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)
#---------------------------------------------------------------lookback
look_back = 100
train_size = 11000
test_size = 1000
num_points = train_size + test_size 

ls = 0.5 
ha = 0.5  

weighted_errors_rmse = []
utide_errors_rmse = []
weighted_errors_r2 = []
utide_errors_r2 = []


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])  # 目标是下一个时间步的 anomaly
    return np.array(dataX), np.array(dataY)

# x time iteration
for iteration in range(15):
    print(f"Iteration {iteration + 1}/10")
    total_length = len(dataset_scaled)
    max_start_index = total_length - num_points
    start_index = np.random.randint(0, max_start_index + 1)
    selected_data = dataset_scaled[start_index:start_index + num_points]
    train, test = selected_data[:train_size], selected_data[train_size:train_size + test_size]
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    num_features = trainX.shape[2]

    #model-------------------------------------------------
    input_layer = Input(shape=(look_back, num_features))
    x = Bidirectional(LSTM(250, kernel_regularizer=l2(0.0001)))(input_layer)
    x = Dropout(0.4)(x)
    output_layer = Dense(1)(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optimizer)
    #dynamic lr
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=10, min_lr=1e-6, verbose=1)

    # train-------------------------------------------------
    model.fit(trainX, trainY, validation_split=0.1, epochs=200, batch_size=512, verbose=1, callbacks=[reduce_lr])

    # rolling forcast
    current_input = testX[0].copy()
    utide_test = test[:, 1] 
    rolling_predictions = []
    for i in range(len(testY)):
        current_input_reshaped = current_input.reshape(1, look_back, num_features)
        prediction = model.predict(current_input_reshaped, verbose=0)
        rolling_predictions.append(prediction[0, 0])
        current_input = np.roll(current_input, -1, axis=0)
        current_input[-1, 0] = prediction[0, 0]
        current_input[-1, 1] = utide_test[look_back + i]

    rolling_predictions = np.array(rolling_predictions)

    # inverse
    predictions_full = np.zeros((len(rolling_predictions), num_features))
    predictions_full[:, 0] = rolling_predictions
    predictions_full[:, 1] = test[look_back:, 1] 

    testY_full = np.zeros((len(testY), num_features))
    testY_full[:, 0] = testY
    testY_full[:, 1] = test[look_back:, 1]

    predictions_inverse = scaler.inverse_transform(predictions_full)[:, 0]
    testY_inverse = scaler.inverse_transform(testY_full)[:, 0]
    utide_inverse = scaler.inverse_transform(test[look_back:])[:, 1]

    combined_predictions_inverse = ls * predictions_inverse + ha * utide_inverse

    # eva
    rmse_weighted = math.sqrt(mean_squared_error(testY_inverse, combined_predictions_inverse))
    rmse_utide = math.sqrt(mean_squared_error(testY_inverse, utide_inverse))
    r2_weighted = r2_score(testY_inverse, combined_predictions_inverse)
    r2_utide = r2_score(testY_inverse, utide_inverse)

    weighted_errors_rmse.append(rmse_weighted)
    utide_errors_rmse.append(rmse_utide)
    weighted_errors_r2.append(r2_weighted)
    utide_errors_r2.append(r2_utide)

    print(f"Weighted Average RMSE: {rmse_weighted:.2f}, R²: {r2_weighted:.2f}")
    print(f"Utide RMSE: {rmse_utide:.2f}, R²: {r2_utide:.2f}")


avg_weighted_rmse = np.mean(weighted_errors_rmse)
avg_utide_rmse = np.mean(utide_errors_rmse)
avg_weighted_r2 = np.mean(weighted_errors_r2)
avg_utide_r2 = np.mean(utide_errors_r2)


print(f"Weighted Average RMSE (Mean over 10 runs): {avg_weighted_rmse:.2f}, R²: {avg_weighted_r2:.2f}")
print(f"Utide RMSE (Mean over 10 runs): {avg_utide_rmse:.2f}, R²: {avg_utide_r2:.2f}")


plt.figure(figsize=(12, 8))
x_ticks = range(1, 11)
plt.plot(x_ticks, weighted_errors_rmse, marker='o', label='Weighted Average RMSE')
plt.plot(x_ticks, utide_errors_rmse, marker='x', label='Utide RMSE')
plt.axhline(avg_weighted_rmse, color='blue', linestyle='--', label=f'Avg Weighted RMSE: {avg_weighted_rmse:.2f}')
plt.axhline(avg_utide_rmse, color='red', linestyle='--', label=f'Avg Utide RMSE: {avg_utide_rmse:.2f}')
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.title('Comparison of Weighted Average vs Utide RMSE')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(x_ticks, weighted_errors_r2, marker='o', label='Weighted Average R²')
plt.plot(x_ticks, utide_errors_r2, marker='x', label='Utide R²')
plt.axhline(avg_weighted_r2, color='blue', linestyle='--', label=f'Avg Weighted R²: {avg_weighted_r2:.2f}')
plt.axhline(avg_utide_r2, color='red', linestyle='--', label=f'Avg Utide R²: {avg_utide_r2:.2f}')
plt.xlabel('Iteration')
plt.ylabel('R²')
plt.title('Comparison of Weighted Average vs Utide R²')
plt.legend()
plt.grid(True)
plt.show()
