# Smart-energy-prediction-using-LSTM-and-RNN
# ===============================
# Smart Home Energy Consumption Prediction using LSTM & RNN
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("smart_home_energy_consumption.csv")

# Rename if encoding issue occurs
df.columns = df.columns.str.strip()

# Combine Date + Time into one column
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df = df.sort_values("Datetime")

# Drop unnecessary columns
df = df.drop(columns=["Date", "Time"])

# -------------------------------
# 2. Encode Categorical Features
# -------------------------------
le_appliance = LabelEncoder()
le_season = LabelEncoder()

df["Appliance Type"] = le_appliance.fit_transform(df["Appliance Type"])
df["Season"] = le_season.fit_transform(df["Season"])

# -------------------------------
# 3. Select Features
# -------------------------------
features = ["Appliance Type", "Outdoor Temperature (°C)", "Season", "Household Size", "Energy Consumption (kWh)"]
data = df[features].values

# -------------------------------
# 4. Feature Scaling
# -------------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# -------------------------------
# 5. Create Sequences
# -------------------------------
SEQ_LEN = 48

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, -1])  # Target = next energy consumption
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, SEQ_LEN)

# Split into train/test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -------------------------------
# 6. Build and Train RNN Model
# -------------------------------
rnn_model = Sequential([
    SimpleRNN(64, return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
    Dropout(0.2),
    SimpleRNN(32),
    Dense(1)
])

rnn_model.compile(optimizer='adam', loss='mse')
print("\nTraining Simple RNN...")
rnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# -------------------------------
# 7. Build and Train LSTM Model
# -------------------------------
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')
print("\nTraining LSTM...")
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# -------------------------------
# 8. Predictions & Evaluation
# -------------------------------
rnn_pred = rnn_model.predict(X_test)
lstm_pred = lstm_model.predict(X_test)

def evaluate_model(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}")

evaluate_model(y_test, rnn_pred, "RNN")
evaluate_model(y_test, lstm_pred, "LSTM")

# -------------------------------
# 9. Visualization
# -------------------------------
plt.figure(figsize=(10,5))
plt.plot(y_test[:200], label="Actual", linewidth=2)
plt.plot(lstm_pred[:200], label="LSTM Predicted", linestyle='--')
plt.plot(rnn_pred[:200], label="RNN Predicted", linestyle=':')
plt.legend()
plt.title("Energy Consumption Prediction")
plt.xlabel("Time Steps")
plt.ylabel("Scaled Energy Consumption")
plt.show()
