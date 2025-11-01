import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, concatenate
from tensorflow.keras.models import Model
import joblib
import os

# Stock tickers
tickers = [
    'NVDA', 'TSLA', 'INTC', 'F', 'AAPL', 'MSFT', 'AMZN',
    'GOOGL', 'META', 'BRK-B', 'JPM', 'V', 'JNJ', 'WMT',
    'PG', 'XOM', 'NKE', 'CVS', 'PM', 'NEM'
]

start_date = '2020-04-01'
end_date = '2025-04-01'
time_steps = 10

model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

def create_lstm_data(data, time_steps=1):
    x, y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])  # Using the value after the time_steps as the target
    return np.array(x), np.array(y)

def prepare_all_data(tickers, start_date, end_date, time_steps):
    all_data = []
    scalers = {}
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        close_prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        close_prices_scaled = scaler.fit_transform(close_prices)

        x, y = create_lstm_data(close_prices_scaled, time_steps)
        all_data.append(x)
        scalers[ticker] = scaler
    return all_data, scalers

# Prepare data for all tickers
all_data, scalers = prepare_all_data(tickers, start_date, end_date, time_steps)

# Concatenate data across all tickers into one dataset (as multi-inputs)
# Instead of directly concatenating them, we now stack the data per ticker
X_all = np.concatenate([data for data in all_data], axis=0)  # Concatenate all sequences
y_all = np.concatenate([data[:, -1] for data in all_data], axis=0)  # Concatenate the last value of each sequence as the label

# Now reshape X_all to have 3 dimensions: (samples, time_steps, num_tickers)
# Since we have 20 tickers, we will reshape X_all into (24920, 10, 20)
X_all_reshaped = np.reshape(X_all, (X_all.shape[0], time_steps, len(tickers)))

# Check the shape of X_all_reshaped and y_all
print(f"X_all_reshaped shape: {X_all_reshaped.shape}")
print(f"y_all shape: {y_all.shape}")

# Build the LSTM model
inputs = []
lstm_outputs = []

# Create one input layer per ticker
for i in range(len(tickers)):
    input_layer = Input(shape=(time_steps, 1), name=f"input_{tickers[i]}")
    lstm_layer = LSTM(units=50, return_sequences=False)(input_layer)
    lstm_outputs.append(lstm_layer)
    inputs.append(input_layer)

merged = concatenate(lstm_outputs)
dense_layer = Dense(units=1)(merged)

model = Model(inputs=inputs, outputs=dense_layer)
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit([X_all_reshaped[:, :, i] for i in range(len(tickers))], y_all, epochs=50, batch_size=32, verbose=1)

# Save the model
model.save(f"{model_dir}/combined_model.h5")

# Save all scalers for later use
for ticker, scaler in scalers.items():
    joblib.dump(scaler, f"{model_dir}/{ticker}_scaler.gz")

print(f"Saved combined model and scalers for all tickers.")

def predict_next_days(ticker, days=10, time_steps=10, end_date='2025-04-05'):
    model = load_model(f"models/combined_model.h5")
    scaler = joblib.load(f"models/{ticker}_scaler.gz")

    data = yf.download(ticker, start='2020-04-01', end=end_date)
    close_prices = data['Close'].values.reshape(-1, 1)
    close_prices_scaled = scaler.transform(close_prices)

    last_data = close_prices_scaled[-time_steps:]
    last_data = np.reshape(last_data, (1, time_steps, 1))

    future_predictions = []
    for _ in range(days):
        pred = model.predict([last_data[:, :, i:i+1] for i in range(len(tickers))], verbose=0)
        future_predictions.append(pred[0][0])
        last_data = np.append(last_data[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions

# Example usage
print(predict_next_days('NVDA', 1, 10, '2025-04-05'))
