import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit app title
st.title("Data Analysis Dashboard")

# Load datasets
@st.cache_data
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

# Sidebar options
st.sidebar.header("Filters")
sort_order = st.sidebar.radio("Sort Districts By:", ("Ascending", "Descending"), index=1)

# Analysis and Visualizations
st.subheader("Average Price by District in Perak")
district_price_perak = merged_data_perak.groupby('district')['item_price'].mean().reset_index().sort_values(by='item_price', ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=district_price_perak, x='district', y='item_price', palette='viridis', ax=ax)
plt.xticks(rotation=45, ha='right')
plt.title('Average Price by District in Perak')
st.pyplot(fig)

# Select and preprocess the datasets
    pricecatcher_selected = pricecatcher[['premise_code', 'item_code', 'price']].copy()
    pricecatcher_selected.rename(columns={'price': 'item_price'}, inplace=True)

    lookup_premise_selected = lookup_premise[['premise_code', 'premise', 'premise_type', 'state', 'district']].copy()

    # Merge datasets
    merged_data = pd.merge(pricecatcher_selected, lookup_premise_selected, on=['premise_code'], how='inner')
    
    # Filter for Perak state
    return merged_data[merged_data['state'] == 'Perak']

# Load the data
merged_data_perak = load_data()

# Group and calculate average prices
district_premise_price = merged_data_perak.groupby(['district', 'premise_type'])['item_price'].mean().reset_index()

# Sidebar filters
st.sidebar.header("Filters")
district_filter = st.sidebar.multiselect(
    "Select District(s):", 
    options=district_premise_price['district'].unique(), 
    default=district_premise_price['district'].unique()
)

# Apply filters
filtered_data = district_premise_price[district_premise_price['district'].isin(district_filter)]

# Plot the data
st.subheader("Bar Plot of Average Prices by Premise Type in Perak Districts")
fig, ax = plt.subplots(figsize=(14, 8))
sns.barplot(data=filtered_data, x='district', y='item_price', hue='premise_type', palette='viridis', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_title('Average Price by Premise Type in Perak Districts', fontsize=16)
ax.set_xlabel('District', fontsize=12)
ax.set_ylabel('Average Price (RM)', fontsize=12)
plt.tight_layout()

# Display the plot
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

# Load datasets
@st.cache_data
def load_data():
    pricecatcher = pd.read_csv('https://raw.githubusercontent.com/nicaaaaaaa/advdsproject/refs/heads/main/012024.csv')
    lookup_premise = pd.read_csv('https://raw.githubusercontent.com/nicaaaaaaa/advdsproject/refs/heads/main/lookup_premise.csv')
    return pricecatcher, lookup_premise

pricecatcher, lookup_premise = load_data()

# Streamlit app title
st.title("Price Analysis Dashboard")

# Merge and preprocess data
pricecatcher['date'] = pd.to_datetime(pricecatcher['date'], errors='coerce')
pricecatcher_selected = pricecatcher[['premise_code', 'item_code', 'price', 'date']].copy()
pricecatcher_selected.rename(columns={'price': 'item_price'}, inplace=True)
lookup_premise_selected = lookup_premise[['premise_code', 'premise', 'premise_type', 'state', 'district']].copy()
merged_data = pd.merge(pricecatcher_selected, lookup_premise_selected, on='premise_code', how='inner')

# Filter data for Perak
merged_data_perak = merged_data[merged_data['state'] == 'Perak']

# Sidebar options
st.sidebar.header("Filters")
show_trend = st.sidebar.checkbox("Show Price Trend Over Time", True)
show_urban_rural = st.sidebar.checkbox("Compare Urban vs Rural Prices", True)
show_district_summary = st.sidebar.checkbox("Show District Summary", True)

# Display filtered options
if show_trend:
    st.subheader("Average Price Trend Over Time")
    price_trend = merged_data_perak.groupby('date')['item_price'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=price_trend, x='date', y='item_price', marker='o', color='green', ax=ax)
    ax.set_title('Average Item Price Over Time', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Average Price (RM)', fontsize=12)
    st.pyplot(fig)

if show_urban_rural:
    st.subheader("Average Prices by Premise Type")
    urban_rural_prices = merged_data_perak.groupby('premise_type')['item_price'].mean().reset_index()
    st.write(urban_rural_prices)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=urban_rural_prices, x='premise_type', y='item_price', palette='mako', ax=ax)
    ax.set_title('Average Prices by Premise Type in Perak', fontsize=16)
    ax.set_xlabel('Premise Type', fontsize=12)
    ax.set_ylabel('Average Price (RM)', fontsize=12)
    st.pyplot(fig)

if show_district_summary:
    st.subheader("Summary Statistics by District")
    district_summary = merged_data_perak.groupby('district')['item_price'].describe()
    st.write(district_summary)

    # Boxplot for district price distribution
    st.subheader("Price Distribution by District")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=merged_data_perak, x='district', y='item_price', palette='coolwarm', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title('Price Distribution by District in Perak', fontsize=16)
    ax.set_xlabel('District', fontsize=12)
    ax.set_ylabel('Item Price (RM)', fontsize=12)
    st.pyplot(fig)

st.sidebar.info("Select the filters above to customize the dashboard view.")

# Streamlit app title
st.title("District Price Prediction Using Linear Regression")

# Load district price data (replace with your source or data loading method)
@st.cache_data
def load_district_data():
    # Sample DataFrame (replace with actual data loading code)
    district_price_perak = pd.DataFrame({
        'district': ['A', 'B', 'C', 'D'],
        'item_price': [15.5, 20.1, 13.2, 18.4]
    })
    return district_price_perak

# Load data
district_price_perak = load_district_data()

# One-hot encoding of districts
district_encoded = pd.get_dummies(district_price_perak['district'], prefix='district')
district_price_perak = pd.concat([district_price_perak.reset_index(drop=True), district_encoded], axis=1)

# Define features and target
X = district_price_perak.drop(['district', 'item_price'], axis=1)
y = district_price_perak['item_price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display evaluation metrics
st.subheader("Model Evaluation")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R-squared: {r2:.2f}")

# Prediction for all districts
st.subheader("Predicted Prices for Districts")
all_districts = district_price_perak['district'].unique()
predicted_prices = {}

for district in all_districts:
    # Create input data for prediction
    district_data = pd.DataFrame(0, index=[0], columns=X.columns)
    district_data[f'district_{district}'] = 1
    predicted_price = model.predict(district_data)[0]
    predicted_prices[district] = predicted_price

# Display predicted prices
predicted_df = pd.DataFrame(list(predicted_prices.items()), columns=['District', 'Predicted Price'])
st.write(predicted_df)

# Visualization: Actual vs. Predicted Prices
st.subheader("Actual vs. Predicted Prices")
results = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})

# Bar plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=pd.melt(results.reset_index(), id_vars=['index'], var_name='Price Type', value_name='Price'),
            x='index', y='Price', hue='Price Type', ax=ax)
plt.xticks(rotation=45, ha='right')
plt.title('Actual vs. Predicted Prices')
plt.ylabel('Price')
st.pyplot(fig)

# Scatter plot
st.subheader("Scatter Plot: Actual vs. Predicted Prices")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, ax=ax)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', linewidth=2)  # Diagonal line
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs. Predicted Prices')
st.pyplot(fig)
# Summary statistics by district
district_summary = merged_data_perak.groupby('district')['item_price'].describe()
st.subheader("Summary Statistics by District")
st.write(district_summary)
