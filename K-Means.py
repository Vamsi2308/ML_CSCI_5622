# importing in built packages

import requests  ## for getting data from a server GET
import re   ## for regular expressions
import pandas as pd    ## for dataframes and related
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from sklearn import tree
from sklearn import preprocessing
import seaborn as sns
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ## Clustering
df_new = pd.read_csv( "cleaned_data.csv")

categorical_columns = df_new.select_dtypes(include=['object'])
categorical_columns.head()

numerical_columns = df_new.select_dtypes(include=['int64', 'float64'])


numerical_columns.head()

numerical_columns.isnull().sum()

##### Performing Cluserting #######

####################################

# Using these attributes: TRAFFIC_COUNT vs STATION_NUMBER

####################################

##### For K=2

# Select features for clustering
X = numerical_columnsX = numerical_columns[['STATION_NUMBER', 'TRAFFIC_COUNT', 'TRAFFIC_YEAR_COUNTED', 'CHRIS_NUMB']]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)  # You can adjust the number of clusters as needed
numerical_columns['cluster'] = kmeans.fit_predict(X_scaled)


# Check the dimensions and data
print("Dimensions of X_scaled:", X_scaled.shape)
print("Dimensions of df:", numerical_columns.shape)
print("Cluster assignments:", numerical_columns['cluster'].value_counts())

# Visualize the results
plt.scatter(numerical_columns['TRAFFIC_COUNT'], numerical_columns['CHRIS_NUMB'], c=numerical_columns['cluster'], cmap='rainbow')
plt.xlabel('TRAFFIC_COUNT')
plt.ylabel('CHRIS_NUMB')
plt.title('K-Means Clustering(K=2)')
plt.show()

##### For K=3

# Select features for clustering
X = numerical_columnsX = numerical_columns[['STATION_NUMBER', 'TRAFFIC_COUNT', 'TRAFFIC_YEAR_COUNTED', 'CHRIS_NUMB']]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # You can adjust the number of clusters as needed
numerical_columns['cluster'] = kmeans.fit_predict(X_scaled)


# Check the dimensions and data
print("Dimensions of X_scaled:", X_scaled.shape)
print("Dimensions of df:", numerical_columns.shape)
print("Cluster assignments:", numerical_columns['cluster'].value_counts())

# Visualize the results
plt.scatter(numerical_columns['TRAFFIC_COUNT'], numerical_columns['CHRIS_NUMB'], c=numerical_columns['cluster'], cmap='rainbow')
plt.xlabel('TRAFFIC_COUNT')
plt.ylabel('CHRIS_NUMB')
plt.title('K-Means Clustering(K=3)')
plt.show()

##### For K=4


# Select features for clustering
X = numerical_columnsX = numerical_columns[['STATION_NUMBER', 'TRAFFIC_COUNT', 'TRAFFIC_YEAR_COUNTED', 'CHRIS_NUMB']]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)  # You can adjust the number of clusters as needed
numerical_columns['cluster'] = kmeans.fit_predict(X_scaled)


# Check the dimensions and data
print("Dimensions of X_scaled:", X_scaled.shape)
print("Dimensions of df:", numerical_columns.shape)
print("Cluster assignments:", numerical_columns['cluster'].value_counts())

# Visualize the results
plt.scatter(numerical_columns['TRAFFIC_COUNT'], numerical_columns['CHRIS_NUMB'], c=numerical_columns['cluster'], cmap='rainbow')
plt.xlabel('TRAFFIC_COUNT')
plt.ylabel('CHRIS_NUMB')
plt.title('K-Means Clustering(K=4)')
plt.show()

####################################

# Using these attributes: TRAFFIC_COUNT vs STATION_NUMBER

####################################

##### For K=2

# Select features for clustering
X = numerical_columns[['STATION_NUMBER', 'TRAFFIC_COUNT', 'TRAFFIC_YEAR_COUNTED', 'CHRIS_NUMB']]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
numerical_columns['cluster'] = kmeans.fit_predict(X_scaled)

# Check the dimensions and data
print("Dimensions of X_scaled:", X_scaled.shape)
print("Dimensions of df:", numerical_columns.shape)
print("Cluster assignments:", numerical_columns['cluster'].value_counts())

# Visualize the results
plt.scatter(numerical_columns['TRAFFIC_COUNT'], numerical_columns['STATION_NUMBER'], c=numerical_columns['cluster'], cmap='rainbow')
plt.xlabel('TRAFFIC_COUNT')
plt.ylabel('STATION_NUMBER')
plt.title('K-Means Clustering(K=2)')
plt.show()

# Print cluster assignments
print(numerical_columns[['STATION_NUMBER', 'TRAFFIC_COUNT', 'TRAFFIC_YEAR_COUNTED', 'CHRIS_NUMB', 'cluster']])


########  For K=3


# Select features for clustering
X = numerical_columns[['STATION_NUMBER', 'TRAFFIC_COUNT', 'TRAFFIC_YEAR_COUNTED', 'CHRIS_NUMB']]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
numerical_columns['cluster'] = kmeans.fit_predict(X_scaled)

# Check the dimensions and data
print("Dimensions of X_scaled:", X_scaled.shape)
print("Dimensions of df:", numerical_columns.shape)
print("Cluster assignments:", numerical_columns['cluster'].value_counts())

# Visualize the results
plt.scatter(numerical_columns['TRAFFIC_COUNT'], numerical_columns['STATION_NUMBER'], c=numerical_columns['cluster'], cmap='rainbow')
plt.xlabel('TRAFFIC_COUNT')
plt.ylabel('STATION_NUMBER')
plt.title('K-Means Clustering(K=3)')
plt.show()

# Print cluster assignments
print(numerical_columns[['STATION_NUMBER', 'TRAFFIC_COUNT', 'TRAFFIC_YEAR_COUNTED', 'CHRIS_NUMB', 'cluster']])

##### for k=4


# Select features for clustering
X = numerical_columns[['STATION_NUMBER', 'TRAFFIC_COUNT', 'TRAFFIC_YEAR_COUNTED', 'CHRIS_NUMB']]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
numerical_columns['cluster'] = kmeans.fit_predict(X_scaled)

# Check the dimensions and data
print("Dimensions of X_scaled:", X_scaled.shape)
print("Dimensions of df:", numerical_columns.shape)
print("Cluster assignments:", numerical_columns['cluster'].value_counts())

# Visualize the results
plt.scatter(numerical_columns['TRAFFIC_COUNT'], numerical_columns['STATION_NUMBER'], c=numerical_columns['cluster'], cmap='rainbow')
plt.xlabel('TRAFFIC_COUNT')
plt.ylabel('STATION_NUMBER')
plt.title('K-Means Clustering(K=4)')
plt.show()

# Print cluster assignments
print(numerical_columns[['STATION_NUMBER', 'TRAFFIC_COUNT', 'TRAFFIC_YEAR_COUNTED', 'CHRIS_NUMB', 'cluster']])