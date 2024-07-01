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
# src/app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    return pd.read_csv(file_path)

def calculate_moving_averages(df, ma_days=[10, 20, 50]):
    for ma in ma_days:
        column_name = f"MA for {ma} days"
        df[column_name] = df['Adj Close'].rolling(ma).mean()
    return df

def plot_data(df):
    fig, ax = plt.subplots(figsize=(15, 5))
    df[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=ax)
    ax.set_title('MLYBY Stock Prices with Moving Averages')
    st.pyplot(fig)

def main():
    st.title('MLYBY Stock Price Analysis')

    # Load data
    file_path = 'data/MLYBY.csv'
    df = load_data(file_path)

    # Calculate moving averages
    df = calculate_moving_averages(df)

    # Plotting
    plot_data(df)

    # Show raw data
    if st.checkbox('Show raw data'):
        st.write(df)

if __name__ == '__main__':
    main()
