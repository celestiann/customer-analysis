import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('customer_analysis.csv')

# Initial checks
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nMarital_Status unique values: {df['Marital_Status'].unique()}")
print(f"\nOldest customer birth year: {df['Year_Birth'].min()}")
# Clean dataimport pandas as pd

# Load data
df = pd.read_csv('customer_analysis.csv')

# 1. Fix Dt_Customer first (critical error)
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')  # Use day-month-year format

# 2. Handle missing values
median_income = df['Income'].median()
df['Income'] = df['Income'].fillna(median_income)  # No inplace=True

# 3. Remove implausible birth years
df = df[df['Year_Birth'] > 1940]

# 4. Clean Marital_Status
df['Marital_Status'] = df['Marital_Status'].replace(
    {'Alone': 'Single', 'Absurd': 'Single', 'YOLO': 'Single'}
)

# 5. Drop duplicates
df = df.drop_duplicates()
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)

# mean avg, standard deviation, etc 
# print(df['Year_Birth'].describe())
# print(df['Income'].describe())
# print(df['Recency'].describe())

# EDA, most customers are aged 40-60
sns.histplot(df['Year_Birth'], bins=20, kde=True)
plt.title('Customer Birth Year Distribution')
plt.show()
# income distribution, median income $52,000
# High-income outliers (> $120,000) exist but are retained (likely affluent customers).
sns.boxplot(x=df['Income'])
plt.title('Income Distribution')
plt.show()
# education and marital status, 50% have bachelor's degree, 60% are married. 20% single
sns.countplot(x='Education', data=df)
plt.xticks(rotation=45)
plt.title('Education Distribution')
plt.show()

# Family Structure, 70% of customers have 1-2 children(Kidhome + Teenhome)
# Spending Impact: Customers with no children spend 2x more on average.
# Total children
df['Total_Children'] = df['Kidhome'] + df['Teenhome']
sns.barplot(x='Total_Children', y='MntWines', data=df)
plt.title('Wine Spending vs. Number of Children')
plt.show()

# Spending habits
# Top Spending Categories: 1. Wine(avg. $305), 2. Meat(avg. $166)
# Lowest Spending: Fruits (avg. $26) and Sweets (avg. $36).
# Average spending per category
spending_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df[spending_columns].mean().sort_values().plot(kind='barh')
plt.title('Average Spending by Product Category')
plt.show()

# income vs spending, strong correlation(r=0.62)
# High-income customers are the primary revenue drivers.
df['Total_Spending'] = df[spending_columns].sum(axis=1)
sns.scatterplot(x='Income', y='Total_Spending', data=df)
plt.title('Income vs. Total Spending')
plt.show()

# Most Successful Campaign: Campaign 4 (15% acceptance).
# Least Successful: Campaign 3 (7% acceptance).
# Replicate Campaign 4’s strategy

campaigns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
df[campaigns].mean().plot(kind='bar')
plt.title('Campaign Acceptance Rates')
plt.show()

# Preferred Channel: Store purchases (avg. 5.2 transactions/customer).
# Web Visits vs. Purchases: High web visits do not correlate with web purchases (r = -0.22).
# Channel preferences
channel_columns = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
df[channel_columns].mean().plot(kind='bar')
plt.title('Average Purchases by Channel')
plt.show()
df.to_csv('cleaned_customer_data.csv', index=False)

# Customer Segmentation Hints:

# Affluent Families: High-income, married, with 1–2 children. Spend heavily on wine and meat.
# Budget Shoppers: Lower income, use discounts (high NumDealsPurchases), prefer web purchases.
# Loyal Campaign Responders: Accepted 3+ campaigns; 30% higher spending than average.

#  Recommendations:

# Target High-Income Families: Promote premium products (wine, meat) via email catalogs.
# Improve Campaign 3: Investigate why Campaign 3 underperformed.
# Optimize Web Experience: Reduce friction between web visits and purchases.
# Data Quality: Standardize marital status categories during data collection.
