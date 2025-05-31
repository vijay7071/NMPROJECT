import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, Attention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import math

# Streamlit configuration
st.set_page_config(layout="wide", page_title="Stock Price Predictor")
st.title("📈 Stock Price Prediction App with CNN-BiLSTM-Attention")

# Input for API Key and Ticker
api_key = st.text_input("🔑 Enter your Alpha Vantage API Key:", value="", type="password")
tickers = st.text_input("📊 Enter stock ticker symbols (comma-separated, e.g., AAPL, TSLA):").upper().split(',')

if st.button("🔍 Predict"):
    ts = TimeSeries(key=api_key, output_format='pandas')

    def compute_rsi(df, window=14):
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def create_attention_layer(inputs):
        attention = Attention()([inputs, inputs])
        return attention

    for ticker in tickers:
        ticker = ticker.strip()
        st.subheader(f"🔁 Processing: {ticker}")

        try:
            data, _ = ts.get_daily(symbol=ticker, outputsize='full')
            st.write(f"✅ Retrieved {ticker} data of shape: {data.shape}")
            data = data[['4. close', '5. volume']]
            data.rename(columns={'4. close': 'Close', '5. volume': 'Volume'}, inplace=True)
            data.index = pd.to_datetime(data.index)
            data = data.sort_index()
        except Exception as e:
            st.error(f"❌ Error retrieving data for {ticker}: {e}")
            continue

        if data.empty:
            st.warning(f"No data found for {ticker}")
            continue

        # Feature engineering
        data['SMA'] = data['Close'].rolling(window=14).mean()
        data['EMA'] = data['Close'].ewm(span=14, adjust=False).mean()
        data['RSI'] = compute_rsi(data)
        data['LogReturn'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Volume_scaled'] = MinMaxScaler().fit_transform(data[['Volume']])
        data.dropna(inplace=True)

        feature_cols = ['Close', 'SMA', 'EMA', 'RSI', 'LogReturn', 'Volume_scaled']
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[feature_cols])

        train_len = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_len]
        test_data = scaled_data[train_len:]

        # Prepare training sequences
        x_train, y_train = [], []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i])
            y_train.append(train_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Model
        inputs = Input(shape=(x_train.shape[1], x_train.shape[2]))
        x = Conv1D(64, kernel_size=3, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Bidirectional(LSTM(100, return_sequences=True))(x)
        x = Dropout(0.3)(x)
        x = create_attention_layer(x)
        x = Bidirectional(LSTM(100))(x)
        x = Dropout(0.3)(x)
        x = Dense(50, activation='relu')(x)
        output = Dense(1)(x)

        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='mean_squared_error')

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)

        model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.1,
                  callbacks=[early_stop, reduce_lr], verbose=0)

        # Test data
        x_test, y_test = [], []
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i])
            y_test.append(test_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        predictions = model.predict(x_test)

        predictions_rescaled = scaler.inverse_transform(
            np.hstack((predictions, np.zeros((predictions.shape[0], scaled_data.shape[1] - 1))))
        )[:, 0]

        actual_prices = scaler.inverse_transform(
            np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), scaled_data.shape[1] - 1))))
        )[:, 0]

        rmse = math.sqrt(mean_squared_error(actual_prices, predictions_rescaled))
        st.success(f"✅ {ticker} RMSE on test set: {rmse:.2f}")

        valid = data.iloc[train_len:].copy().iloc[60:]
        if len(predictions_rescaled) < len(valid):
            st.warning("⚠️ Predictions shorter than actual data. Trimming 'valid' to match.")
            valid = valid.iloc[:len(predictions_rescaled)]
        valid['Predictions'] = predictions_rescaled

        st.write("📊 valid shape:", valid.shape)
        st.write("📊 predictions_rescaled shape:", predictions_rescaled.shape)

        # Plot actual vs predicted
        st.write("🖼️ Plotting Historical Prediction...")
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(data['Close'], label="Training Data")
        ax1.plot(valid['Close'], label="Actual Test Price", color='blue')
        ax1.plot(valid['Predictions'], label="Predicted Test Price", color='orange')
        ax1.set_title(f"{ticker} - Historical Prediction")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid()
        st.pyplot(fig1)
        st.write("✅ Historical plot rendered.")

        # Future forecasting
        future_days = 30
        future_predictions = []
        last_60_days = scaled_data[-60:].reshape(1, 60, scaled_data.shape[1])

        for _ in range(future_days):
            pred = model.predict(last_60_days)[0][0]
            future_predictions.append(pred)
            next_input = np.append(last_60_days[:, 1:, :], [[[pred] + [0] * (scaled_data.shape[1] - 1)]], axis=1)
            last_60_days = next_input

        st.write("🧮 Future prediction shape before inverse transform:", np.array(future_predictions).shape)

        future_predictions_rescaled = scaler.inverse_transform(
            np.hstack((np.array(future_predictions).reshape(-1, 1), np.zeros((future_days, scaled_data.shape[1] - 1))))
        )[:, 0]

        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days, freq='B')

        # Future forecast plot
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(data['Close'], label="Historical Close")
        ax2.plot(valid['Predictions'], label="Past Predicted", color='orange', linestyle='--')
        ax2.plot(future_dates, future_predictions_rescaled, label="Future Prediction", color='green')
        ax2.set_title(f"{ticker} - Future {future_days}-Day Forecast")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price")
        ax2.legend()
        ax2.grid()
        st.pyplot(fig2)
