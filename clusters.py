import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('customer_analysis.csv', sep=',')  
df.columns = df.columns.str.strip()
df = df.dropna()
df.to_csv('transformed_file.csv', sep=';', index=False) 
print(df.head())

# Selecting features for clustering
features = ['Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df_cluster = df[features].dropna()  # Drop missing values if any
# Scale the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cluster)
# Find optimal K using the Elbow Method
inertia = []
K_range = range(1, 11) # Trying K values from 1 to 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o', linestyle='-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# Apply K-Means with the chosen K
optimal_k = 4  # Choose the best K from the elbow method
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_cluster['Cluster'] = kmeans.fit_predict(df_scaled)

# Add cluster labels back to the original dataframe
df['Cluster'] = df_cluster['Cluster']

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Income'], y=df['MntWines'], hue=df['Cluster'], palette='viridis', s=100)
plt.xlabel('Income')
plt.ylabel('Amount Spent on Wine')
plt.title('K-Means Clustering Results')
plt.show()
