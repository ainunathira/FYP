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

######################################prediction###################################################
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

###########################################Model Evaluation######################################################
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Load the CSV file into a DataFrame
data = pd.read_csv('MLYBY.csv')

# Assuming 'Close' is the column you want to predict (adjust as needed)
dataset = data.filter(['Close']).values

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Define the sequence length for input to the LSTM model
sequence_length = 60  # Number of timesteps to look back

# Create the training data set
training_data_len = int(len(dataset) * 0.8)  # 80% of data for training
train_data = scaled_data[0:training_data_len, :]

# Prepare the training data in sequences
x_train = []
y_train = []
for i in range(sequence_length, len(train_data)):
    x_train.append(train_data[i-sequence_length:i, 0])  # Using past 'sequence_length' values
    y_train.append(train_data[i, 0])  # Target value is the next value after the sequence

# Convert to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data for LSTM input [samples, timesteps, features]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Define and train the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=128, epochs=1)

# Prepare the test data
test_data = scaled_data[training_data_len - sequence_length:, :]
x_test = []
y_test = dataset[training_data_len:, :]  # Actual values for comparison

for i in range(sequence_length, len(test_data)):
    x_test.append(test_data[i-sequence_length:i, 0])

# Convert to numpy array
x_test = np.array(x_test)

# Reshape the data for LSTM input [samples, timesteps, features]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the model's predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Calculate the root mean squared error (RMSE)
rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
print("Root Mean Squared Error (RMSE):", rmse)

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:].copy()  # Create a copy to avoid the warning
valid.loc[:, 'Predictions'] = predictions  # Use .loc to set the predictions

# Visualize the data
plt.figure(figsize=(16, 8))
plt.title('MLYBY Predicted Stock Price ')
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()   
st.pyplot(fig)
