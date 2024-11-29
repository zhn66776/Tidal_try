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

np.random.seed(5)
input_file = "dataProcessed/ABE5Y1_HA_processed.csv"
df = read_csv(input_file, delimiter=',')
total_length = len(df)
print(f"data length：{total_length}")
num_points = 12000

if total_length >= num_points:
    max_start_index = total_length - num_points
    np.random.seed(int(time.time()))
    start_index = np.random.randint(0, max_start_index + 1)
    end_index = start_index + num_points - 1
    start_time = df.iloc[start_index]['time']
    end_time = df.iloc[end_index]['time']
    df = df.iloc[start_index:end_index + 1]

    print(f"data start from {start_index} to {end_index}")
    print(f"time start from {start_time} to {end_time}。")
else:
    print("less data")

#use two features
feature_columns = ['anomaly', 'utide']
dataset = df[feature_columns].values
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)
#--------------------------------------------------------------------------------------------------------------data split
train_size = 11000
test_size = 1000
train, test = dataset_scaled[:train_size], dataset_scaled[train_size:train_size + test_size]
print(f"train size: {train.shape[0]}, test size: {test.shape[0]}")
look_back = 30  
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

print(f"trainX shape: {trainX.shape}, trainY shape: {trainY.shape}")
print(f"testX shape: {testX.shape}, testY shape: {testY.shape}")

num_features = trainX.shape[2]

learning_rate = 0.001
weight_decay = 0.0001

input_layer = Input(shape=(look_back, num_features))
# x = Bidirectional(LSTM(250, return_sequences=True, kernel_regularizer=l2(weight_decay)))(input_layer)
# x = Bidirectional(LSTM(100, kernel_regularizer=l2(weight_decay)))(x)

x = Bidirectional(LSTM(200, return_sequences=True, kernel_regularizer=l2(weight_decay)))(input_layer)
x = Bidirectional(LSTM(200, return_sequences=True, kernel_regularizer=l2(weight_decay)))(x)
x = Bidirectional(LSTM(50, kernel_regularizer=l2(weight_decay)))(x)
x = Dropout(0.4)(x)
output_layer = Dense(1)(x)

model = Model(inputs=input_layer, outputs=output_layer)
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='mse', optimizer=optimizer)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=10, min_lr=1e-6, verbose=1)
#early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
def scheduler(epoch, lr):
    if epoch % 50 == 0 and epoch != 0:
        lr = lr * 0.5
    return lr

lr_scheduler = LearningRateScheduler(scheduler)

#train
history = model.fit(
    trainX, trainY,
    validation_split=0.1,
    epochs=200,
    batch_size=512,
    verbose=1,
    callbacks=[reduce_lr, lr_scheduler]
)

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('mse')
plt.title('loss function')
plt.legend()
plt.show()

#test
current_input = testX[0].copy()  # shape (look_back, num_features)
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

predictions_full = np.zeros((len(rolling_predictions), num_features))
predictions_full[:, 0] = rolling_predictions
predictions_full[:, 1] = test[look_back:, 1] 
testY_full = np.zeros((len(testY), num_features))
testY_full[:, 0] = testY
testY_full[:, 1] = test[look_back:, 1]

# inverse
predictions_inverse = scaler.inverse_transform(predictions_full)[:, 0]
testY_inverse = scaler.inverse_transform(testY_full)[:, 0]
utide_inverse = scaler.inverse_transform(test[look_back:])[:, 1]

#evaluate
rmse = math.sqrt(mean_squared_error(testY_inverse, predictions_inverse))
mae = mean_absolute_error(testY_inverse, predictions_inverse)
r2 = r2_score(testY_inverse, predictions_inverse)

aligned_utide = utide_inverse

#error
error_anomaly = testY_inverse - predictions_inverse
error_utide = testY_inverse - utide_inverse



plt.figure(figsize=(18, 6))
plt.plot(error_anomaly, label='LSTM_error (Anomaly)', color='red')
plt.plot(error_utide, label='utide_error (Anomaly vs Utide)', color='blue')
plt.xlabel('time step')
plt.ylabel('error')
plt.title('error compare')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(18, 10))
plt.plot(testY_inverse, label="real (Anomaly)")
plt.plot(predictions_inverse, label="LSTM_predicted (Anomaly)", color='red', linestyle="--")
plt.plot(aligned_utide, label="(Utide)", color='blue', linestyle="--", linewidth=1.5)
plt.xlabel("time step")
plt.ylabel("Anomaly")
plt.title("real vs predicted vs utide")
plt.legend()
plt.show()


#weight combine
alpha = 0.5
beta = 0.5
combined_predictions_inverse = alpha * predictions_inverse + beta * aligned_utide
rmse_combined = math.sqrt(mean_squared_error(testY_inverse, combined_predictions_inverse))
mae_combined = mean_absolute_error(testY_inverse, combined_predictions_inverse)
r2_combined = r2_score(testY_inverse, combined_predictions_inverse)

plt.figure(figsize=(18, 10))
plt.plot(testY_inverse, label="real (Anomaly)")
plt.plot(combined_predictions_inverse, label="combined (LSTM & Utide)", color='purple', linestyle="--")
plt.xlabel("time step")
plt.ylabel("Anomaly")
plt.title("combined (LSTM & Utide)")
plt.legend()
plt.show()

error_combined = testY_inverse - combined_predictions_inverse

plt.figure(figsize=(18, 6))
plt.plot(error_combined, label='combined error', color='purple')
plt.xlabel('time step')
plt.ylabel('error')
plt.title('combined error')
plt.legend()
plt.grid(True)
plt.show()


print("各模型的误差指标：\n")

print("LSTM error：")
print(f'Test RMSE: {rmse:.2f}')
print(f'Test MAE: {mae:.2f}')
print(f'Test R²: {r2:.2f}')

print("\nHA：")
harmonic_rmse = math.sqrt(mean_squared_error(testY_inverse, utide_inverse))
harmonic_mae = mean_absolute_error(testY_inverse, utide_inverse)
harmonic_r2 = r2_score(testY_inverse, utide_inverse)
print(f'Harmonic RMSE: {harmonic_rmse:.2f}')
print(f'Harmonic MAE: {harmonic_mae:.2f}')
print(f'Harmonic R²: {harmonic_r2:.2f}')

print("\nLSTM HA combine：")
print(f'Combined RMSE: {rmse_combined:.2f}')
print(f'Combined MAE: {mae_combined:.2f}')
print(f'Combined R²: {r2_combined:.2f}')

plt.figure(figsize=(8, 8))
plt.scatter(testY_inverse, predictions_inverse, alpha=0.5, label=f'LSTM R² = {r2:.2f}', color='red')
plt.plot([testY_inverse.min(), testY_inverse.max()], [testY_inverse.min(), testY_inverse.max()], 'r--', label='theo y=x')
plt.xlabel('real Anomaly')
plt.ylabel('prediceted Anomaly')
plt.title('real Anomaly vs LSTM predicted Anomaly')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(testY_inverse, utide_inverse, alpha=0.5, label=f'Harmonic R² = {harmonic_r2:.2f}', color='blue')
plt.plot([testY_inverse.min(), testY_inverse.max()], [testY_inverse.min(), testY_inverse.max()], 'b--', label='theo y=x')
plt.xlabel('real Anomaly')
plt.ylabel('(Utide)')
plt.title('real Anomaly vs utide Anomaly')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 8))
plt.scatter(testY_inverse, combined_predictions_inverse, alpha=0.5, label=f'Combined R² = {r2_combined:.2f}', color='purple')
plt.plot([testY_inverse.min(), testY_inverse.max()], [testY_inverse.min(), testY_inverse.max()], 'g--', label='theo y=x')
plt.xlabel('Anomaly')
plt.ylabel('combined Anomaly')
plt.title('real Anomaly vs Anomaly')
plt.legend()
plt.grid(True)
plt.show()
