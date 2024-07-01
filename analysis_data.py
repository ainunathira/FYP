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
