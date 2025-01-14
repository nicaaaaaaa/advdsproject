import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit app title and description
st.title("Data Analysis for Pisang Berangan in Perak")
st.markdown("""
## Objective
- Analyze the price trends of pisang berangan
""")

@st.cache_data
def load_data():
    # Load the datasets
    pricecatcher_jan = pd.read_csv('https://raw.githubusercontent.com/nicaaaaaaa/advdsproject/refs/heads/main/012024.csv')
    pricecatcher_feb = pd.read_csv('https://raw.githubusercontent.com/nicaaaaaaa/advdsproject/refs/heads/main/pc022024.csv')
    pricecatcher_mar = pd.read_csv('https://raw.githubusercontent.com/nicaaaaaaa/advdsproject/refs/heads/main/032024.csv')
    pricecatcher_apr = pd.read_csv('https://raw.githubusercontent.com/nicaaaaaaa/advdsproject/refs/heads/main/042024.csv')
    pricecatcher_may = pd.read_csv('https://raw.githubusercontent.com/nicaaaaaaa/advdsproject/refs/heads/main/052024.csv')
    pricecatcher_june = pd.read_csv('https://raw.githubusercontent.com/nicaaaaaaa/advdsproject/refs/heads/main/062024.csv')
    lookup_premise = pd.read_csv('https://raw.githubusercontent.com/nicaaaaaaa/advdsproject/refs/heads/main/lookup_premise.csv')
    income_data = pd.read_csv('https://raw.githubusercontent.com/nicaaaaaaa/advdsproject/refs/heads/main/hh_income_state.csv')
    district_data = pd.read_csv('https://raw.githubusercontent.com/nicaaaaaaa/advdsproject/refs/heads/main/hies_district.csv')
    
    # Combine monthly datasets
    pricecatcher_combined = pd.concat(
        [pricecatcher_jan, pricecatcher_feb, pricecatcher_mar, pricecatcher_apr, pricecatcher_may, pricecatcher_june],
        ignore_index=True
    )

    # Select columns and merge
    pricecatcher_combined['date'] = pd.to_datetime(pricecatcher_combined['date'], format='%d/%m/%Y')
    pricecatcher_selected = pricecatcher_combined[['premise_code', 'item_code', 'price', 'date']].rename(
        columns={'price': 'item_price'}
    )
    lookup_premise_selected = lookup_premise[['premise_code', 'premise', 'premise_type', 'state', 'district']]
    merged_data = pd.merge(pricecatcher_selected, lookup_premise_selected, on='premise_code', how='inner')

    return merged_data

# Load and filter the data
merged_data = load_data()
merged_data_perak = merged_data[merged_data['state'] == 'Perak']

# Streamlit app
st.markdown("""
## Pisang Berangan Price Data for Perak
""")
st.write("This is the filtered dataset containing price data for Perak state.")

# Show the dataset for Perak
st.write("### Merged Dataset for Perak")
st.dataframe(merged_data_perak)

# Allow user to download the filtered dataset
st.download_button(
    label="Download Perak Dataset as CSV",
    data=merged_data_perak.to_csv(index=False),
    file_name="merged_data_perak.csv",
    mime="text/csv"
)

# Sidebar filters
st.sidebar.header("Filters")
selected_district = st.sidebar.multiselect(
    "Select Districts:",
    options=merged_data_perak['district'].unique(),
    default=merged_data_perak['district'].unique()
)
filtered_data = merged_data_perak[merged_data_perak['district'].isin(selected_district)]

# Price Trend Graph
st.subheader("Price Trend of Pisang Berangan Over Time")
price_trend = filtered_data.groupby('date')['item_price'].mean().reset_index()
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=price_trend, x='date', y='item_price', marker='o', color='green', ax=ax)
ax.set_title('Pisang Berangan Price Trend (Perak)', fontsize=16)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Average Price (RM)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Price Distribution
st.subheader("Distribution of Item Prices")
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(filtered_data['item_price'], bins=20, kde=True, color='blue', ax=ax)
ax.set_title('Distribution of Pisang Berangan Prices in Perak', fontsize=16)
ax.set_xlabel('Pisang Berangan (RM)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
plt.tight_layout()
st.pyplot(fig)

# District-wise Price Analysis
st.subheader("Average Price by District")
district_price = filtered_data.groupby('district')['item_price'].mean().reset_index().sort_values(by='item_price', ascending=False)
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=district_price, x='district', y='item_price', palette='viridis', ax=ax)
ax.set_title('Average Price by District in Perak', fontsize=16)
ax.set_xlabel('District', fontsize=12)
ax.set_ylabel('Average Price (RM)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig)
