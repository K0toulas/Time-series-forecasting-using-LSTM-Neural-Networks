import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time 
import statsmodels as stmd
from statsmodels.tsa.arima.model import ARIMA

#Wrapper for gathering the data from .csv file. We only need the closing prices for the model and dates for the plots.
df = pd.read_csv("BTC-2021_01_01_2022_03_01_min_basis.csv")
df['date'] = pd.to_datetime(df['date'])
closing_prices = df['close'].values.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0,1))
closing_prices_scaled = scaler.fit_transform(closing_prices)

#Number of epoch iterations and sequence length(parameters that controll accuracy of the model)
epoch = 10
sequence_length = 1440

X, Y = [], []

for i in range(len(closing_prices_scaled) - sequence_length):
    X.append(closing_prices_scaled[i:i+sequence_length])  # Input sequence
    Y.append(closing_prices_scaled[i+sequence_length])  # Target price

X, Y = np.array(X), np.array(Y)

#Reshape input to match LSTM expectations: (samples, time steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

X = X.astype(np.float32)
Y = Y.astype(np.float32)

forecast_days = 10
forecast_minutes = forecast_days * 24 * 60  # Total minutes in 10 days

#Get test data for validation
X_test = X[:forecast_minutes]
Y_test = Y[:forecast_minutes]

#Build LSTM model
model = keras.Sequential([
    layers.LSTM(50, activation='relu', input_shape=(sequence_length,1), return_sequences=True),
    layers.LSTM(50, activation='relu'),
    layers.Dense(1, activation='linear')
])

#Compile model
model.compile(optimizer='adam', loss='mse')

#Train model
history = model.fit(X, Y, epochs=epoch, batch_size=32, verbose=1)

#Get weights and save them in a file
ltsm_weights = model.get_weights()

with open("lstm_weights.txt", "w") as f:
    for i, w in enumerate(ltsm_weights):
        f.write(f"Layer {i + 1} Weights:\n")
        np.savetxt(f, w, delimiter=",")
        f.write("\n")  # Add spacing for readability

print("LSTM weights saved to 'lstm_weights.txt'")

#Plotting MSE during training
final_mse = history.history['loss']
print(f"MSE for each epoch in order is: {final_mse}")

#Plot for MSE of training phase of LTSM model
plt.figure(figsize=(16, 8))
plt.title('MSE during training')
plt.plot(final_mse)
plt.ylabel('MSE Value')
plt.legend()
plt.show()

#Do validation with historic data.
Y_pred_scaled = model.predict(X_test)

Y_pred = scaler.inverse_transform(Y_pred_scaled)
Y_test_actual = scaler.inverse_transform(Y_test)

#Compute MSE
mse_valid = mean_squared_error(Y_test_actual, Y_pred)
print(f"Validation MSE: {mse_valid}")

#PLot for validation of model 
plt.figure(figsize=(16, 8))
plt.title('Validation: Actual vs. Predicted Prices (Last 10 Days)')
plt.plot(df['date'].iloc[:forecast_minutes], Y_test_actual, label='Actual Price', color='blue')
plt.plot(df['date'].iloc[:forecast_minutes], Y_pred, label='Predicted Price', color='red')
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.legend()
plt.show()

forecast_predictions = []

start_index = forecast_minutes - sequence_length
if start_index < 0:
    raise ValueError("Not enough data for the forecast period.")

# Get the initial sequence from the correct starting point
current_sequence = closing_prices_scaled[start_index : forecast_minutes]
current_sequence = current_sequence.reshape(1, sequence_length, 1)

#Forecast with LTSM model for the final 10 days.
start_time = time.time()

for _ in range(forecast_minutes): # Iterate through each minute of the forecast
    predicted_price_scaled = model.predict(current_sequence, verbose = 0)
    forecast_predictions.append(predicted_price_scaled[0, 0])

    # Update the current sequence
    current_sequence = np.roll(current_sequence, 1, axis=1)  # Shift the sequence by one
    current_sequence[0, 1, 0] = predicted_price_scaled[0, 0]  # Replace the last element with the prediction

end_time = time.time()
forecast_time = end_time - start_time

print(f'Forecasting time taken: {forecast_time:.2f} seconds')
forecast_predictions = np.array(forecast_predictions).reshape(-1, 1)
forecast_predictions = scaler.inverse_transform(forecast_predictions)

#Arima model forecasting
arima_model = ARIMA(df['close'], order=(5,1,1))  # (p,d,q) values can be tuned
arima_model_fit = arima_model.fit()
arima_forecast = arima_model_fit.forecast(steps=forecast_minutes)

forecast_dates = pd.date_range(start=df['date'].iloc[start_index] + pd.Timedelta(minutes=1), periods=forecast_minutes, freq='min')

#Plot for Actual vs LTSM vs ARIMA for final 10 days.
plt.figure(figsize=(16, 8))
plt.title('Actual vs. Predicted Prices', fontsize=24)
plt.plot(df['date'], df['close'], label='Actual Price', color='blue')  # Plot actual price range
plt.plot(forecast_dates, forecast_predictions, label='Predicted Price (Last 10 Days)', color='green')
plt.plot(forecast_dates, arima_forecast, label='ARIMA Prediction', color='red')
plt.axvspan(forecast_dates[0], forecast_dates[-1], alpha=0.2, color='lightgreen') # Plot the forecast period
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD', fontsize=18)
plt.legend()
plt.show()