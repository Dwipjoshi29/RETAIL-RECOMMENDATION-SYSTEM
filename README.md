#Step 1 | Importing The Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.cluster import KMeans
import random

df = pd.read_excel('/kaggle/input/online-retail-xlsx/Online Retail.xlsx')
df.head()

df.info()

df = df[df['CustomerID'].notnull()]
df.info()

#Step 3 | Optimizing Data for Enhanced Clustering
#--"Creating 'InvoiceDay' Column for Date-Based Analysis":
df['InvoiceDay'] = df['InvoiceDate'].apply(lambda x: dt.datetime(x.year, x.month, x.day))
df.head()

#--"Finding Last Purchase Date for Customer Management":
dt.timedelta(1)

pin_date = max(df['InvoiceDay']) + dt.timedelta(1)
pin_date

df['TotalSum'] = df['Quantity'] * df['UnitPrice']
df.head()

#--Creating RFM Variables for Customer Analysis and Marketing Strategies:
rfm = df.groupby('CustomerID').agg({
    'InvoiceDay': lambda x: (pin_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalSum': 'sum'
})
rfm

rfm.rename(columns= {
    'InvoiceDay': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalSum': 'Monetary'
}, inplace=True)
rfm

#Step 4 | Data Preprocessing
r_labels = range(4, 0, -1) #[4, 3, 2, 1]
r_groups = pd.qcut(rfm['Recency'], q=4, labels=r_labels)
f_labels = range(1, 5) # [1, 2, 3, 4]
f_groups = pd.qcut(rfm['Frequency'], q=4, labels=f_labels)
m_labels = range(1, 5)
m_groups = pd.qcut(rfm['Monetary'], q=4, labels=m_labels)
rfm['R'] = r_groups.values
rfm['F'] = f_groups.values
rfm['M'] = m_groups.values
rfm

#Step 5 | Customer Clustering for Targeted Marketing
X = rfm[['R', 'F', 'M']]
kmeans = KMeans(n_clusters=10, init='k-means++', max_iter=300)
kmeans.fit(X)

kmeans.labels_

rfm['kmeans_cluster'] = kmeans.labels_
rfm

#Step 6 | Customer Clustering Visualization
# Number of clusters
num_clusters = 10

# Create subplots with two clusters in each row
fig, axes = plt.subplots(num_clusters // 2, 2, figsize=(12, 20))

# Flatten the axes array to iterate through subplots
axes = axes.ravel()

# Loop through each cluster and plot it
for cluster_id in range(num_clusters):
    # Filter data for the current cluster
    cluster_data = rfm[rfm['kmeans_cluster'] == cluster_id]
    
    # Plot the data with a distinct color
    sns.scatterplot(data=cluster_data, x='Recency', y='Frequency', hue='Monetary', palette='viridis', ax=axes[cluster_id])
    
    # Set the title for the subplot
    axes[cluster_id].set_title(f'Cluster {cluster_id}')
    
    # Customize axes labels, if needed
    # axes[cluster_id].set_xlabel('X-axis Label')
    # axes[cluster_id].set_ylabel('Y-axis Label')

# Add a common legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right')

# Adjust subplot spacing
plt.tight_layout()

# Show the plot
plt.show()

# Create a histogram for Recency in each cluster
plt.figure(figsize=(12, 6))
for cluster_id in range(num_clusters):
    plt.subplot(2, 5, cluster_id + 1)
    sns.histplot(rfm[rfm['kmeans_cluster'] == cluster_id]['Recency'], bins=20, kde=True)
    plt.title(f'Cluster {cluster_id}')
    plt.xlabel('Recency')
    plt.ylabel('Frequency')
    
plt.tight_layout()
plt.show()

#Step 7 | Creating a Recommendation System
Generating Top Product Recommendations for Each Cluster
# Number of clusters (groups)
num_clusters = 10

# Create an empty dictionary to store recommendations for each cluster
cluster_recommendations = {}

# Loop through each cluster
for cluster_id in range(num_clusters):
    # Find customers in the current cluster
    customers_in_cluster = rfm[rfm['kmeans_cluster'] == cluster_id].index
    
    # Find top products for customers in the current cluster
   top_products_for_cluster = df[df['CustomerID'].isin(customers_in_cluster)].groupby(['StockCode'])['InvoiceNo'].count().sort_values(ascending=False).head(10)
    
    # Store the top products for the current cluster in the dictionary
  cluster_recommendations[f'Cluster {cluster_id}'] = top_products_for_cluster.index.tolist()

# Display the recommendations for each cluster
for cluster, recommended_products in cluster_recommendations.items():
    print(f"{cluster} -> Recommended Products: {recommended_products}")

#--Cluster Analysis: Product Recommendations
def generate_cluster_recommendations(num_clusters, num_customers_to_display, rfm, df):
    # Create an empty dictionary to store recommendations for each cluster
    cluster_recommendations = {}

    # Loop through each cluster
   for cluster_id in range(num_clusters):
        # Find customers in the current cluster
        customers_in_cluster = rfm[rfm['kmeans_cluster'] == cluster_id].index

        # Find top products for customers in the current cluster  
  top_products_for_cluster = df[df['CustomerID'].isin(customers_in_cluster)].groupby(['StockCode'])['InvoiceNo'].count().sort_values(ascending=False).head(10)

        # Find customers who haven't purchased any of the top products in the current cluster
   non_buyers = [customer for customer in customers_in_cluster if not (df[(df['CustomerID'] == customer) & (df['StockCode'].isin(top_products_for_cluster.index.tolist()))]).empty]


        # Limit the number of non-buyers to the specified number
  num_customers_to_display = min(num_customers_to_display, len(non_buyers))

        # Select non-buyer customers for the current cluster
   selected_customers = non_buyers[:num_customers_to_display]

        # Store the top products and selected non-buyer customers for the current cluster in the dictionary
  cluster_recommendations[f'Cluster {cluster_id}'] = {
            'Recommended Products': top_products_for_cluster.index.tolist(),
            'Selected Non-Buyer Customers': selected_customers
        }

   return cluster_recommendations

# Example usage:
num_clusters = 10
num_customers_to_display = 5

# Assuming you already have 'rfm' and 'df' dataframes
cluster_recommendations = generate_cluster_recommendations(num_clusters, num_customers_to_display, rfm, df)

# Display the recommendations and selected non-buyer customers for each cluster
for cluster, recommendations_and_customers in cluster_recommendations.items():
    print(f"{cluster} ->")
    print("Recommended Products:")
    for customer_id in recommendations_and_customers['Selected Non-Buyer Customers']:
        print(f"Customer: {customer_id} =====>>>> Recommended Products: {recommendations_and_customers['Recommended Products']}")
    print()
