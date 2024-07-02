import streamlit as st
import pandas as pd

# Load file csv - ensure the file is loaded in your Python environment
data = pd.read_csv('MLYBY.csv')

# Visualize top 10 rows of the sample
print(data.head(10))

st.write(data)

#####################################ANALYSIS DATA############################################
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'df'
df = pd.read_csv('MLYBY.csv')  # Replace with your actual data loading logic

# Create a new DataFrame 'df2' with the 'Close' column
df2 = df.reset_index()['Close']

# Plot the 'Close' column
plt.plot(df2)

plt.show()
st.pyplot(plt) 
st.write("Raw data:", df)

##############################################################################################
##What was the change in price of the stock overtime?
##stock information with pandas, and how to analyze basic attributes of a stock.
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Function to download stock data with retry mechanism
def download_stock_data_with_retry(symbol, start, end, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            data = yf.download(symbol, start=start, end=end)
            return data
        except Exception as e:
            retries += 1
            if retries == max_retries:
                st.error(f"Error occurred after {max_retries} retries: {e}")
                return None
            st.warning(f"Error occurred: {e}. Retrying...")

# Set up Streamlit sidebar for user input
st.sidebar.title('Stock Data Analysis')
symbol = st.sidebar.text_input("Enter stock symbol", "MLYBY")
start_date = st.sidebar.date_input("Start date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End date", datetime.now())

# Main content area to display data
st.title('Stock Data Analysis')

if symbol:
    st.write(f"Downloading data for {symbol} from {start_date} to {end_date}...")
    data = download_stock_data_with_retry(symbol, start_date, end_date)
    
    if data is not None:
        st.write("Data successfully downloaded:")
        st.write(data.head())
        
        # Optionally, you can add more visualizations or analysis here
        # Example: Plotting closing prices
        st.subheader("Closing Prices")
        st.line_chart(data['Close'])

#######################################MOVING AVERAGE##########################################
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

ma_day = [10, 20, 50]

# Assuming you have a DataFrame named MLYBY
MLYBY = pd.read_csv('MLYBY.csv')

for ma in ma_day:
    column_name = f"MA for {ma} days"
    MLYBY[column_name] = MLYBY['Adj Close'].rolling(ma).mean()

fig, axes = plt.subplots(figsize=(15, 5))

MLYBY[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes)
axes.set_title('MLYBY')
plt.show()
st.pyplot(plt)

################################DAILY RETURN#######################################################
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

ma_day = [10, 20, 50]

# Assuming you have a DataFrame named MLYBY
MLYBY = pd.read_csv('MLYBY.csv')

# Calculate daily returns
MLYBY['Daily Return'] = MLYBY['Adj Close'].pct_change()

# Calculate moving averages for daily returns
for ma in ma_day:
    column_name = f"MA for {ma} days"
    MLYBY[column_name] = MLYBY['Daily Return'].rolling(ma).mean()

# Plot the data
fig, axes = plt.subplots(figsize=(15, 5))
# x- date
# y - daily stock return

MLYBY[['Daily Return', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes)
axes.set_title('MLYBY - Daily Return and Moving Averages')
plt.show()
st.pyplot(plt)

######################################PREDICTION###################################################
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# Load data from CSV file (replace 'MLYBY.csv' with your actual file)
data = pd.read_csv('MLYBY.csv')

# Check if 'Date' column exists in the DataFrame
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y', errors='coerce')

if 'Date' in data.columns:
    # Sort the data by date
    data.sort_values(by='Date', inplace=True)
    
    # Create a new column 'Day_Index' for easier indexing
    data['Day_Index'] = range(1, len(data) + 1)

    # Using 'Day_Index' as the feature
    X = data[['Day_Index']].values

    # Using 'Close' as the target variable
    y = data['Close'].values

    # Normalize the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Splitting the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    # Build and train the model (example using Linear Regression)
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Generate predictions
    predictions = model.predict(X_scaled)

    # Adding 'Predictions' to the DataFrame
    data['Predictions'] = predictions

    # Set up the Streamlit app layout
    st.title('MLYBY Stock Price Prediction')
    st.subheader('Model Prediction vs Actual Data')

    # Plotting the training data
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Date'][:len(x_train)], data['Close'][:len(x_train)], '-', label='Train', color='blue')

    # Plotting the validation data and predictions
    ax.plot(data['Date'][len(x_train):], data['Close'][len(x_train):], '-', label='Validation', color='green')
    ax.plot(data['Date'], data['Predictions'], '-', label='Predictions', color='orange')

    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Close Price USD ($)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True)

    # Display plot in Streamlit
    st.pyplot(fig)

else:
    st.error("Error: 'Date' column not found in the DataFrame.")
    
################################################CANDLESTICK#########################################
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import mplfinance as mpf

# Load data from CSV file (replace 'MLYBY.csv' with your actual file)
data = pd.read_csv('MLYBY.csv')

# Check if 'Date' column exists in the DataFrame
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y', errors='coerce')

if 'Date' in data.columns:
    # Sort the data by date
    data.sort_values(by='Date', inplace=True)
    
    # Create a new column 'Year_Index' for easier indexing by year
    data['Year_Index'] = data['Date'].dt.year

    # Using 'Year_Index' as the feature
    X = data[['Year_Index']].values

    # Using 'Close' as the target variable
    y = data['Close'].values

    # Normalize the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Splitting the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    # Build and train the model (example using Linear Regression)
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Generate predictions
    predictions = model.predict(X_scaled)

    # Adding 'Predictions' to the DataFrame
    data['Predictions'] = predictions

    # Prepare data for candlestick plotting
    plot_data = data[['Date', 'Open', 'High', 'Low', 'Close']].copy()
    plot_data['Predictions'] = predictions
    plot_data.set_index('Date', inplace=True)

    # Set up the Streamlit app layout
    st.title('MLYBY Stock Price Prediction')
    st.subheader('Model Prediction vs Actual Data')

    # Create the candlestick chart with predictions as an additional plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plotting candlestick chart
    mpf.plot(plot_data, type='candle', style='charles', ax=ax, title='Actual vs Predicted Stock Prices', ylabel='Stock Price')

    # Adding predicted values as an additional plot
    ax.plot(plot_data.index, plot_data['Predictions'], color='orange', label='Predictions')

    # Customize the plot
    ax.legend()
    ax.grid(True)

    # Display plot in Streamlit
    st.pyplot(fig)

else:
    st.error("Error: 'Date' column not found in the DataFrame.")

