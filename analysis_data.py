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

######################################HEATMAPS###################################################
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

def main():
    st.title('MLYBY Stock Price Analysis')

    # Sidebar
    st.sidebar.header('User Input Parameters')
    start_date = st.sidebar.date_input("Start date", value=pd.to_datetime('2020-01-01'))
    end_date = st.sidebar.date_input("End date", value=pd.to_datetime('today'))

    # Load data
    @st.cache
    def load_data():
        data = yf.download('MLYBY', start=start_date, end=end_date)
        return data

    data = load_data()

    # Display data
    st.subheader('Stock Data')
    st.write(data.head())

    # Plot closing price
    st.subheader('Closing Price')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Close Price', color='b')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.set_title('MLYBY Closing Price')
    ax.grid(True)
    st.pyplot(fig)

if __name__ == '__main__':
    main()
