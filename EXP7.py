
#exp 7
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Load Data (Apple stock example)
df = yf.download("AAPL", start="2015-01-01", end="2023-01-01")
data = df[['Close']].values

# 2. Normalize Data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# 3. Create sequences
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step])
        y.append(data[i+time_step])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(data_scaled, time_step)

# 4. Train-Test Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 5. Build LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 6. Train Model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 7. Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Inverse transform
train_pred = scaler.inverse_transform(train_pred)
test_pred = scaler.inverse_transform(test_pred)
y_train_actual = scaler.inverse_transform(y_train)
y_test_actual = scaler.inverse_transform(y_test)

# 8. Plot Results
plt.figure(figsize=(12,6))
plt.plot(df.index, data, label="Actual Price")

train_plot = np.empty_like(data)
train_plot[:] = np.nan
train_plot[time_step:train_size+time_step] = train_pred

test_plot = np.empty_like(data)
test_plot[:] = np.nan
test_plot[train_size+time_step:] = test_pred

plt.plot(df.index, train_plot, label="Train Prediction")
plt.plot(df.index, test_plot, label="Test Prediction")

plt.legend()
plt.title("Stock Price Prediction using LSTM")
plt.show()


# Results:

# Epoch 1/20
# 49/49 ━━━━━━━━━━━━━━ 3s 26ms/step - loss: 0.0057 - val_loss: 0.0020
# Epoch 2/20
# 49/49 ━━━━━━━━━━━━━━ 1s 23ms/step - loss: 0.0010 - val_loss: 0.0025
# Epoch 3/20
# 49/49━━━━━━━━━━━━━━ 1s 24ms/step - loss: 0.0010 - val_loss: 0.0017
# Epoch 4/20
# 49/49 ━━━━━━━━━━━━━━1s 23ms/step - loss: 9.4730e-04 - val_loss: 0.0016
# Epoch 5/20
# 49/49 ━━━━━━━━━━━━━━1s 23ms/step - loss: 8.8275e-04 - val_loss: 0.0014
# Epoch 6/20
# 49/49 ━━━━━━━━━━━━━━ 1s 23ms/step - loss: 8.4495e-04 - val_loss: 0.0023
# Epoch 7/20
# 49/49 ━━━━━━━━━━━━━━1s 25ms/step - loss: 7.8307e-04 - val_loss: 0.0019
# Epoch 8/20
# 49/49 ━━━━━━━━━━━━━━1s 23ms/step - loss: 7.7224e-04 - val_loss: 0.0028
# Epoch 9/20
# 49/49━━━━━━━━━━━━━━ 1s 24ms/step - loss: 7.6048e-04 - val_loss: 0.0015
# Epoch 10/20
# 49/49 ━━━━━━━━━━━━━━1s 23ms/step - loss: 7.4930e-04 - val_loss: 0.0026
# Epoch 11/20
# 49/49 ━━━━━━━━━━━━━━1s 23ms/step - loss: 7.0710e-04 - val_loss: 0.0029
# Epoch 12/20
# 49/49━━━━━━━━━━━━━━ 1s 23ms/step - loss: 7.0458e-04 - val_loss: 0.0015
# Epoch 13/20
# 49/49 ━━━━━━━━━━━━━━1s 24ms/step - loss: 6.4549e-04 - val_loss: 0.0017
# Epoch 14/20
# 49/49 ━━━━━━━━━━━━━━1s 25ms/step - loss: 6.9553e-04 - val_loss: 0.0015
# Epoch 15/20
# 49/49 ━━━━━━━━━━━━━━1s 24ms/step - loss: 5.9993e-04 - val_loss: 0.0011
# Epoch 16/20
# 49/49 ━━━━━━━━━━━━━━1s 23ms/step - loss: 6.4003e-04 - val_loss: 0.0020
# Epoch 17/20
# 49/49 ━━━━━━━━━━━━━━1s 24ms/step - loss: 5.7396e-04 - val_loss: 0.0015
# Epoch 18/20
# 49/49 ━━━━━━━━━━━━━━1s 23ms/step - loss: 6.0528e-04 - val_loss: 0.0012
# Epoch 19/20
# 49/49━━━━━━━━━━━━━━ 1s 24ms/step - loss: 5.5371e-04 - val_loss: 0.0036
# Epoch 20/20
# 49/49━━━━━━━━━━━━━━ 1s 23ms/step - loss: 5.3254e-04 - val_loss: 0.0012
#  (.venv) (base) ARYANs-MacBook-Air-2:DL LAB ARYAN DUTTA 23CS084 aryandutta$