import pandas as pd  # Work with tables
import numpy as np  # Do math operations
import matplotlib.pyplot as plt  # Create charts
import seaborn as sns  # Make charts look nicer
from sklearn.cluster import KMeans  # K-Means algorithm
from sklearn.preprocessing import StandardScaler  # Scale numbers for better clustering

df = pd.read_csv('cleaned_file.csv')

#mean avg, standard deviation, etc 
# print(df['Year_Birth'].describe())
# print(df['Income'].describe())
# print(df['Recency'].describe())
# print(df['MntWines'].describe())
# print(df['MntFruits'].describe())
# print(df['MntMeatProducts'].describe())
# print(df['MntSweetProducts'].describe())
# print(df['MntGoldProds'].describe())
# print(df['NumDealsPurchases'].describe())
# print(df['NumWebPurchases'].describe())
# print(df['NumCatalogPurchases'].describe())
# print(df['Year_Birth'].plot(kind='hist', bins=30))


# features = ['Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
# df_cluster = df[features].dropna()

# scaler = StandardScaler()
# df_scaled = scaler.fit_transform(df_cluster)
# inertia = []
# K_range = range(1, 11)

# for k in K_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     kmeans.fit(df_scaled)
#     inertia.append(kmeans.inertia_)

# # Plot the Elbow Method
# plt.figure(figsize=(8, 5))
# plt.plot(K_range, inertia, marker='o', linestyle='-')
# plt.xlabel('Number of Clusters (K)')
# plt.ylabel('Inertia (Error)')
# plt.title('Elbow Method for Best K')
# plt.show()

# optimal_k = 3  # Change this to the best K from the graph
# kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)

# # Add cluster labels to the DataFrame
# df_cluster['Cluster'] = kmeans.fit_predict(df_scaled)

# # Also add clusters back to the original dataset
# df['Cluster'] = df_cluster['Cluster']

# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=df['Income'], y=df['MntWines'], hue=df['Cluster'], palette='viridis', s=100)
# plt.xlabel('Income')
# plt.ylabel('Amount Spent on Wine')
# plt.title('K-Means Clustering Results')
# plt.show()

import datetime

# Create an 'Age' column (assuming current year is 2025)
df['Age'] = 2025 - df['Year_Birth']

# Create a 'Total_Spending' column
product_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df['Total_Spending'] = df[product_cols].sum(axis=1)

# import matplotlib.pyplot as plt
# import seaborn as sns

# # Age distribution
# plt.figure(figsize=(8, 5))
# sns.histplot(df['Age'], bins=20, kde=True)
# plt.title('Age Distribution of Customers')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.show()

# # Income distribution
# plt.figure(figsize=(8, 5))
# sns.boxplot(x=df['Income'])
# plt.title('Income Distribution')
# plt.xlabel('Income')
# plt.show()

# # Education count
# plt.figure(figsize=(8, 5))
# sns.countplot(data=df, x='Education')
# plt.title('Education Levels of Customers')
# plt.xlabel('Education Level')
# plt.ylabel('Count')
# plt.xticks(rotation=45)
# plt.show()

# # Marital status count
# plt.figure(figsize=(8, 5))
# sns.countplot(data=df, x='Marital_Status')
# plt.title('Marital Status of Customers')
# plt.xlabel('Marital Status')
# plt.ylabel('Count')
# plt.xticks(rotation=45)
# plt.show()
# Income vs. Total Spending
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Income', y='Total_Spending', data=df)
plt.title('Income vs. Total Spending')
plt.xlabel('Income')
plt.ylabel('Total Spending')
plt.show()

# Recency vs. Total Spending
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Recency', y='Total_Spending', data=df)
plt.title('Recency vs. Total Spending')
plt.xlabel('Recency (Days since last purchase)')
plt.ylabel('Total Spending')
plt.show()

# Campaign acceptance
campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
campaign_data = df[campaign_cols].mean()  # proportion of acceptance

plt.figure(figsize=(8, 5))
sns.barplot(x=campaign_data.index, y=campaign_data.values)
plt.title('Campaign Acceptance Rates')
plt.xlabel('Campaign')
plt.ylabel('Acceptance Rate')
plt.show()

# Purchases by channel
purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
purchase_sums = df[purchase_cols].sum()

plt.figure(figsize=(8, 5))
sns.barplot(x=purchase_sums.index, y=purchase_sums.values)
plt.title('Total Purchases by Channel')
plt.xlabel('Channel')
plt.ylabel('Number of Purchases')
plt.show()

from sklearn.preprocessing import StandardScaler

# Create a new DataFrame for clustering
clustering_features = df[['Age', 'Income', 'Total_Spending', 'Recency', 
                            'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']].copy()

# Standardize the features
scaler = StandardScaler()
clustering_scaled = scaler.fit_transform(clustering_features)

from sklearn.cluster import KMeans
import numpy as np

# Elbow method
sse = []
k_range = range(1, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(clustering_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.title('Elbow Method For Optimal k')
plt.show()

# Assuming optimal clusters = 3 based on the elbow method
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(clustering_scaled)

# Visualize clusters based on two features, for example: Total_Spending vs. Income
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Income', y='Total_Spending', hue='Cluster', data=df, palette='Set1')
plt.title('Clusters based on Income and Total Spending')
plt.xlabel('Income')
plt.ylabel('Total Spending')
plt.show()

cluster_profile = df.groupby('Cluster')[['Age', 'Income', 'Total_Spending', 'Recency']].mean()
print(cluster_profile)
