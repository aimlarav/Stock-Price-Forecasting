import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import load_model

# Streamlit App Title
st.title("Stock Market Prediction")

# Load the saved LSTM model
model_path = 'lstm_model.h5'
try:
    model = load_model(model_path)
    st.success("LSTM model loaded successfully!")
except FileNotFoundError:
    st.error(f"Model file '{model_path}' not found. Please ensure the model is saved in the correct path.")
    st.stop()

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# File Upload Section
st.header("Upload Historical Stock Data")
uploaded_file = st.file_uploader("Upload a CSV file with 'Date' and 'Close' columns", type=["csv"])

if uploaded_file:
    # Load the uploaded file
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(data.tail())
    
    # Convert 'Date' column to datetime and set it as index
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Extract 'Close' prices
    stock_prices = data['Close']

    # Display historical data visualization
    st.subheader("Historical Stock Prices")
    st.line_chart(stock_prices)
       
    # Process the closing price for LSTM
    closing_price = data[['Close']].values
    scaled_closing_price = scaler.transform(closing_price)

    # Prepare the dataset for LSTM
    time_step = 60
    X = []
    for i in range(len(scaled_closing_price) - time_step):
     X.append(scaled_closing_price[i:i + time_step, 0])

    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Predict the next 30 days
    predictions = model.predict(X[-30:])

    # Rescale the predicted prices
    predicted_prices = scaler.inverse_transform(predictions)

    # Calculate predicted dates correctly
    predicted_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30).strftime('%Y-%m-%d')
    # Display prediction results
    st.subheader(f"Predicted Stock Prices for Future Days")
    predicted_df = pd.DataFrame(predicted_prices, columns=["Predicted Close"], index=predicted_dates)
    st.write(predicted_df)

    # Plot the results
    st.subheader(f" Plotted Stock Prices of Predicted Future Days")
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_df.index, predicted_df['Predicted Close'], label="Predicted Price", color="orange")
    plt.title('Predicted Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)
    plt.show()
else:
    st.info("Please upload a CSV file to proceed.")