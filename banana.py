import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# Streamlit app title
st.title("Data Analysis Dashboard")

# Load datasets
@st.cache
def load_data():
    pricecatcher = pd.read_csv('https://raw.githubusercontent.com/nicaaaaaaa/advdsproject/refs/heads/main/012024.csv')
    lookup_premise = pd.read_csv('https://raw.githubusercontent.com/nicaaaaaaa/advdsproject/refs/heads/main/lookup_premise.csv')
    income_data = pd.read_csv('https://raw.githubusercontent.com/nicaaaaaaa/advdsproject/refs/heads/main/hh_income_state.csv')
    district_data = pd.read_csv('https://raw.githubusercontent.com/nicaaaaaaa/advdsproject/refs/heads/main/hies_district.csv')
    return pricecatcher, lookup_premise, income_data, district_data

# Load data
pricecatcher, lookup_premise, income_data, district_data = load_data()

# Display dataframes
if st.checkbox("Show Pricecatcher Dataset"):
    st.write(pricecatcher.head())

if st.checkbox("Show Lookup Premise Dataset"):
    st.write(lookup_premise.head())

if st.checkbox("Show Income Data"):
    st.write(income_data.head())

if st.checkbox("Show District Data"):
    st.write(district_data.head())

# Merge datasets
pricecatcher_selected = pricecatcher[['premise_code', 'item_code', 'price']].rename(columns={'price': 'item_price'})
lookup_premise_selected = lookup_premise[['premise_code', 'premise', 'premise_type', 'state', 'district']]
merged_data = pd.merge(pricecatcher_selected, lookup_premise_selected, on='premise_code', how='inner')
merged_data_perak = merged_data[merged_data['state'] == 'Perak']

# Analysis and Visualizations
st.subheader("Average Price by District in Perak")
district_price_perak = merged_data_perak.groupby('district')['item_price'].mean().reset_index().sort_values(by='item_price', ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=district_price_perak, x='district', y='item_price', palette='viridis', ax=ax)
plt.xticks(rotation=45, ha='right')
plt.title('Average Price by District in Perak')
st.pyplot(fig)

st.subheader("Price Distribution by District in Perak")
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=merged_data_perak, x='district', y='item_price', palette='coolwarm', ax=ax)
plt.xticks(rotation=45, ha='right')
plt.title('Price Distribution by District in Perak')
st.pyplot(fig)

# Descriptive statistics
st.subheader("Descriptive Statistics for Item Prices in Perak")
st.write(merged_data_perak['item_price'].describe())

# Income analysis
st.subheader("Income Analysis")
st.write("Descriptive Statistics for Income")
st.write(income_data['income_mean'].describe())

# Income distribution
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(income_data['income_mean'], bins=30, kde=True, color='blue', ax=ax)
plt.title('Income Distribution')
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=income_data, y='income_mean', color='orange', ax=ax)
plt.title('Income Distribution (Boxplot)')
st.pyplot(fig)

# Skewness and kurtosis
skewness = income_data['income_mean'].skew()
kurt = income_data['income_mean'].kurt()
st.write(f"Skewness: {skewness:.2f}")
st.write(f"Kurtosis: {kurt:.2f}")

# Summary statistics by district
district_summary = merged_data_perak.groupby('district')['item_price'].describe()
st.subheader("Summary Statistics by District")
st.write(district_summary)
