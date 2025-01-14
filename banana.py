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
st.title("Pisang Berangan Price Data for Perak")
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

# Premise Type Analysis
st.subheader("Average Price by Premise Type")
premise_price = filtered_data.groupby(['district', 'premise_type'])['item_price'].mean().reset_index()
fig, ax = plt.subplots(figsize=(14, 8))
sns.barplot(data=premise_price, x='district', y='item_price', hue='premise_type', palette='viridis', ax=ax)
ax.set_title('Average Price by Premise Type in Perak Districts', fontsize=16)
ax.set_xlabel('District', fontsize=12)
ax.set_ylabel('Average Price (RM)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig)

# Skewness and Kurtosis
st.subheader("Distribution Shape")
skewness = filtered_data['item_price'].skew()
kurt = filtered_data['item_price'].kurt()
st.write(f"Skewness: {skewness:.2f}")
st.write(f"Kurtosis: {kurt:.2f}")

# Price Trend Over Time
st.subheader("Price Trend Over Six Months")
price_trend = filtered_data.groupby('date')['item_price'].mean().reset_index()
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=price_trend, x='date', y='item_price', marker='o', color='green', ax=ax)
ax.set_title('Average Pisang Berangan Price Trend (6 Months)', fontsize=16)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Average Price (RM)', fontsize=12)
ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Descriptive Statistics
st.subheader("Descriptive Statistics")
st.write(filtered_data['item_price'].describe())


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

# Streamlit app title
st.title("Diagnostic Analysis")

st.title("Banana Price Trends and Influencing Factors")

st.markdown("""
## Impact of Ease of Cultivation on Banana Prices  
- Bananas classified as easy to cultivate attract more farmers due to lower barriers and costs.  
- Increased cultivation leads to higher market supply, potentially driving prices down.  
- Challenges like diseases or adverse weather reduce supply, causing prices to rise.  

## Supply and Demand Dynamics  
- Excess supply leads to price decreases (basic economics).  
- Reduced supply with steady or rising demand leads to price increases.  

## Bananas in Malaysian Diets and Institutional Demand  
- Bananas are a staple for children due to affordability and nutrition (potassium, vitamin C).  
- Kindergartens and early education centers drive consistent demand through meal plans and bulk purchases, stabilizing the market.  
- Seasonal variations and supply chain issues can still influence prices.  
""")

factor = st.selectbox(
    "Select a factor to explore:",
    ["Impact of Ease of Cultivation", "Supply and Demand Dynamics", "Bananas in Malaysian Diets and Institutional Demand"]
)

if factor == "Impact of Ease of Cultivation":
    st.write("""
    - Bananas classified as easy to cultivate attract more farmers due to lower barriers and costs.  
    - Increased cultivation leads to higher market supply, potentially driving prices down.  
    - Challenges like diseases or adverse weather reduce supply, causing prices to rise.  
    """)
elif factor == "Supply and Demand Dynamics":
    st.write("""
    - Excess supply leads to price decreases (basic economics).  
    - Reduced supply with steady or rising demand leads to price increases.  
    """)
elif factor == "Bananas in Malaysian Diets and Institutional Demand":
    st.write("""
    - Bananas are a staple for children due to affordability and nutrition (potassium, vitamin C).  
    - Kindergartens and early education centers drive consistent demand through meal plans and bulk purchases, stabilizing the market.  
    - Seasonal variations and supply chain issues can still influence prices.  
    """)
