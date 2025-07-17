import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import numpy as np
import os
from utils import create_dataset, scale_data

# Load stock data
ticker = 'AAPL'
df = yf.download(ticker, start='2015-01-01', end='2023-12-31')
df = df[['Close']]

# Scale data
scaled_data, scaler = scale_data(df)
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - 60:]

X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)

# Load or train model
model_path = 'models/lstm_model.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Loaded existing model.")
else:
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32)
    model.save(model_path)

# Predict
predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))
mse = mean_squared_error(actual, predicted)
print(f"Test MSE: {mse}")

# Plot
plt.figure(figsize=(12,6))
plt.plot(actual, label='Actual Price')
plt.plot(predicted, label='Predicted Price')
plt.legend()
plt.title(f'{ticker} Stock Price Forecast')
plt.xlabel('Days')
plt.ylabel('Price')
plt.show()

# Next-day prediction
last_60 = scaled_data[-60:].reshape(1, 60, 1)
next_pred = model.predict(last_60)
next_price = scaler.inverse_transform(next_pred)
print(f"Next Day Predicted Price: ${next_price[0][0]:.2f}")
