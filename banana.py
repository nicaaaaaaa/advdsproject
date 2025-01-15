import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


# Streamlit app title and description
st.title("Data Analysis for Pisang Berangan in Perak")
st.markdown("""
## Objective
- Analyze the price trends of pisang berangan
- Build predictive model to forecast future price trends
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

# Streamlit app title
st.title("Descriptive Analysis")

# Descriptive statistics for item prices in the merged dataset
st.subheader("Descriptive Statistics for Item Prices (Overall)")
overall_stats = merged_data['item_price'].describe()
st.write(overall_stats)

# Descriptive statistics for Perak-specific data
st.subheader("Descriptive Statistics for Pisang Berangan Prices in Perak")
perak_stats = merged_data_perak['item_price'].describe()
st.write(perak_stats)

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

# Price Distribution
st.subheader("Distribution of Item Prices")
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(filtered_data['item_price'], bins=20, kde=True, color='blue', ax=ax)
ax.set_title('Distribution of Pisang Berangan Prices in Perak', fontsize=16)
ax.set_xlabel('Pisang Berangan (RM)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
plt.tight_layout()
st.pyplot(fig)

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

with st.expander("References"):
    st.markdown("""
    - **Article**:
      - [Banana easy to cultivate](https://www.hmetro.com.my)
      - [Kid's diet](https://www.moh.gov.my/moh/resources/auto%20download%20images/589d765c1b95f.pdf,
         https://www.krinstitute.org/assets/contentMS/img/template/editor/Discussion%20Paper_Addressing%20Malnutrition%20in%20Malaysia.pdf,
         https://www.nutriweb.org.my/picture/upload/file/Nutrition%20Guide%20for%20early%20childhood.pdf)
    """)



# Streamlit app title
st.title("Predictive Analysis")
st.markdown("""
## District Price Prediction Using Decision Tree Regression
""")

# Load district price data (replace with your source or data loading method)
@st.cache_data
def load_data():
    # Example dataset (replace this with actual data source)
    district_price_perak = pd.DataFrame({
        'district': ['Muallim', 'Perak Tengah', 'Kerian', 'Kinta', 'Hulu Perak', 
                     'Manjung', 'Kuala Kangsar', 'Larut, Matang & Selama', 
                     'Hilir Perak', 'Batang Padang','Bagan Datuk'],
        'item_price': [6.89, 6.71, 6.67, 6.40, 6.05, 6.03, 6.33, 6.07, 6.17, 6.10,4.0]
    })
    return district_price_perak
   
# Load data
district_price_perak = load_data()

# Encode districts using one-hot encoding
district_encoded = pd.get_dummies(district_price_perak['district'], prefix='district')
district_encoded = district_encoded.reset_index(drop=True)
district_price_perak = pd.concat([district_price_perak.reset_index(drop=True), district_encoded], axis=1)

# Define features and target
X = district_price_perak.drop(['district', 'item_price'], axis=1)
y = district_price_perak['item_price']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of your train datasets
print(X_train.shape)
print(y_train.shape)

# Ensure there are no missing values
print(X_train.isnull().sum())
print(y_train.isnull().sum())

# Handle missing values if needed (example)
X_train = X_train.fillna(X_train.mean())  # Impute missing values in features
y_train = y_train.fillna(y_train.mean())  # Impute missing values in target

# Check if all columns in X_train are numeric
print(X_train.dtypes)

# If necessary, apply one-hot encoding to categorical columns
X_train = pd.get_dummies(X_train)

# Check data consistency
assert len(X_train) == len(y_train), "Mismatch between X_train and y_train samples"

# Train the model
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# Assuming you have district predictions and actual district price data
district_predictions_perak = {
    "Muallim": 6.08,
    "Perak Tengah": 6.30,
    "Kerian": 6.50,
    "Kinta": 6.89,
    "Hulu Perak": 6.73,
    "Manjung": 6.30,
    "Kuala Kangsar": 6.21,
    "Larut, Matang & Selama": 6.34,
    "Hilir Perak": 7.46,
    "Batang Padang": 6.30,
    "Bagan Datuk":4.0
}

# Define the district price DataFrame
district_price_perak = pd.DataFrame({
    'district': ['Muallim', 'Perak Tengah', 'Kerian', 'Kinta', 'Hulu Perak', 
                 'Manjung', 'Kuala Kangsar', 'Larut, Matang & Selama', 
                 'Hilir Perak', 'Batang Padang','Bagan Datuk'],
    'item_price': [6.08, 6.30, 6.5, 6.89, 6.72, 6.30, 6.21, 6.34, 7.46, 6.30,40]
})

# Sort both lists by district name to ensure consistency in order
district_predictions_sorted = {k: district_predictions_perak[k] for k in sorted(district_predictions_perak.keys())}
district_price_sorted = district_price_perak.sort_values('district', ascending=True)

# Ensure the district names match in order
if list(district_predictions_sorted.keys()) != list(district_price_sorted['district']):
    st.error("District names do not match between predictions and actual prices!")
else:
    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        'District': list(district_predictions_sorted.keys()),
        'Predicted Price': list(district_predictions_sorted.values()),
        'Actual Price': district_price_sorted['item_price'].values
    })

    # Melt the DataFrame for easier plotting with seaborn
    plot_data_melted = pd.melt(plot_data, id_vars=['District'], var_name='Price Type', value_name='Price')

    # Create the bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='District', y='Price', hue='Price Type', data=plot_data_melted)
    plt.xticks(rotation=45, ha='right')
    plt.title('Actual vs. Predicted Average Prices by District in Perak')
    plt.ylabel('Average Price (RM)')
    plt.tight_layout()

    # Show the plot in Streamlit
    st.pyplot(plt)

    # Display predictions in the desired format
    st.subheader("Predicted Prices for All Districts")
    for district, price in district_predictions_sorted.items():
        st.write(f"**Predicted price for {district} district:** RM {price:.2f}")

    # Visualization: Bar chart
    st.subheader("Prediction Visualization")
    predicted_df = pd.DataFrame(list(district_predictions_sorted.items()), columns=['District', 'Predicted Price'])
    st.bar_chart(predicted_df.set_index('District'))

    # One-hot encoding of districts
    district_encoded = pd.get_dummies(district_price_sorted['district'], prefix='district')
    district_price_sorted = pd.concat([district_price_sorted.reset_index(drop=True), district_encoded], axis=1)


# Model training
model =  DecisionTreeRegressor()
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


st.title ("Prescriptive Analysis")  
st.markdown("""
## Supply Chains Disruption  
- Farmers are unable to harvest bananas due to external factors, particularly weather conditions.
- Banana farms are located in less strategic areas.
- Farmers must invest in protecting their crops from adverse weather condition.
- To maintaining the farm, farmers are forced to raise banana prices.

## Increased Cost for Maintainance
- Farmers are raising banana prices to sustain their farms and ensure their livelihood.
- Farmers need to invest in tying up banana trees to protect them from storms and adverse weather conditions.
- Farmers face difficulties in maintaining an adequate workforce to manage the plantation, adding to the overall production costs.

## Disease Impact
- Banana trees are affected by a serious disease caused by the Fusarium fungus, which is highly destructive to crops.
- The disease attacks the roots and vascular system of banana plants, causing wilting, yellowing of leaves, and eventual plant death.
- Farmers need to ensure that banana trees are well-fertilized, properly watered, and carefully managed to minimize the risk of infection.
- Infected trees must be isolated from healthy ones to prevent the disease from spreading across the plantation.

## Pisang Berangan price Trend(Based on Manamurah.com)

- Higher demand for bananas as gifts, offerings, and for festive consumption during Chinese New Year.
- Delays in supply caused by worker holidays and seasonal issues, leading to reduced harvests and logistical challenges.
- Wholesalers and retailers engage in market investing, anticipating higher demand, which contributes to price increases.

""")
with st.expander("References"):
    st.markdown("""
    - **Article**:
      - [Agrimag. (n.d.). Harga pisang berangan meningkat. Retrieved February 20, 2024 from, https://agrimag.my/bm/article-details/harga-pisang-berangan-meningkat]
      - [https://manamurah.com/barang/pisang_berangan-18/negeri/perak-8]
      """)



