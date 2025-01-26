```markdown
# BTC Price Prediction with LSTM

This project demonstrates how to use an **LSTM (Long Short-Term Memory)** neural network to predict BTC prices. It includes steps for data preprocessing, calculating technical indicators (e.g., **MACD**, **RSI**), preparing data sequences for LSTM, training the model, and finally testing the model on new data.

---

## Overview
In this project, we:
- **Load BTC price data** (Open, High, Low, Close, Volume).
- **Calculate technical indicators** like MACD, RSI, and moving averages.
- **Transform data** into the necessary format for an LSTM network.
- **Train an LSTM model** to forecast future BTC prices.
- **Evaluate the model** using metrics like Mean Squared Error (MSE) or Mean Absolute Error (MAE).
- **Test** the trained model on new data.

An LSTM is particularly suited for time-series data because it can capture long-term dependencies and temporal patterns more effectively than simple feed-forward networks.


## Data Preparation

1. **Data Format**  
   Make sure your BTC data has columns such as `Datetime, Open, High, Low, Close, Volume`.

2. **Datetime Conversion**  
   Convert `Datetime` to a proper `datetime` object and sort by date/time if needed.

3. **Handling Missing Values**  
   Decide how to handle missing data points (impute, drop, or fill).

4. **Feature Engineering**  
   - Shift the target variable if you want to predict future prices (e.g., next day’s Close).
   - Calculate additional features (see [Technical Indicators](#technical-indicators)).

5. **Normalization/Scaling**  
   LSTM models often benefit from scaled features. Use `MinMaxScaler` or `StandardScaler` from `scikit-learn`.

---

## Technical Indicators

Technical indicators can help the LSTM learn relevant patterns. Common ones include:

**1. MACD**
```python
data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['EMA_12'] - data['EMA_26']
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
```

**2. RSI (14-period)**
```python
delta = data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14, min_periods=1).mean()
avg_loss = loss.rolling(window=14, min_periods=1).mean()
rs = avg_gain / avg_loss
data['RSI_14'] = 100 - (100 / (1 + rs))
```

**3. Moving Averages (SMA, EMA), Bollinger Bands, etc.**

---

## LSTM Model Training

Below is a **simplified example** using **TensorFlow Keras**. Adapt the parameters (like sequence length) to your data.

1. **Create Sequences**  
   LSTMs require 3D input: `(samples, timesteps, features)`. For example, if you use a window of 60 days to predict the next day’s price:

   ```python
   import numpy as np

   def create_sequences(features, target, window_size=60):
       X, y = [], []
       for i in range(len(features) - window_size):
           X.append(features[i : i + window_size])
           y.append(target[i + window_size])
       return np.array(X), np.array(y)
   ```

2. **Prepare Data for Training**  
   ```python
   from sklearn.preprocessing import MinMaxScaler

   # Assume data is your pandas DataFrame with all features, including 'Close'
   features = data[['MACD', 'Signal_Line', 'RSI_14', 'Volume', 'Close']].values

   # Scale features
   scaler = MinMaxScaler()
   scaled_features = scaler.fit_transform(features)

   # We’ll predict the next day’s Close (which is the last column if included)
   target = scaled_features[:, -1]

   # Create sequences
   X, y = create_sequences(scaled_features, target, window_size=60)

   # Train-Test Split
   split_idx = int(len(X) * 0.8)
   X_train, X_test = X[:split_idx], X[split_idx:]
   y_train, y_test = y[:split_idx], y[split_idx:]
   ```

3. **Build the LSTM Model**  
   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import LSTM, Dense, Dropout

   model = Sequential()
   model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
   model.add(Dropout(0.2))
   model.add(LSTM(50, return_sequences=False))
   model.add(Dropout(0.2))
   model.add(Dense(1))  # Predicting one value (next day's Close)

   model.compile(optimizer='adam', loss='mean_squared_error')
   model.summary()
   ```

4. **Train the Model**  
   ```python
   history = model.fit(
       X_train,
       y_train,
       epochs=20,
       batch_size=32,
       validation_data=(X_test, y_test),
       verbose=1
   )
   ```

---

## Model Evaluation

1. **Loss Curves**  
   Visualize training vs. validation loss to check for overfitting or underfitting:
   ```python
   import matplotlib.pyplot as plt

   plt.plot(history.history['loss'], label='Train Loss')
   plt.plot(history.history['val_loss'], label='Validation Loss')
   plt.title('Training and Validation Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.legend()
   plt.grid(True)
   plt.show()
   ```

2. **Predict and Compare**  
   ```python
   import numpy as np
   from sklearn.metrics import mean_squared_error

   y_pred = model.predict(X_test)
   mse = mean_squared_error(y_test, y_pred)
   print(f"Test MSE: {mse}")
   ```

   > **Note**: Since we scaled the data, `y_test` and `y_pred` are in scaled form. You may need to invert the scaling to interpret actual prices.

3. **Inverse Scaling** (if necessary):  
   ```python
   # If 'Close' is the last feature column
   y_test_expanded = np.zeros((len(y_test), scaled_features.shape[1]))
   y_test_expanded[:, -1] = y_test

   y_pred_expanded = np.zeros((len(y_pred), scaled_features.shape[1]))
   y_pred_expanded[:, -1] = y_pred[:, 0]

   y_test_inversed = scaler.inverse_transform(y_test_expanded)[:, -1]
   y_pred_inversed = scaler.inverse_transform(y_pred_expanded)[:, -1]
   ```
