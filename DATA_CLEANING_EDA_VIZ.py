#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# ### Read Data
df = pd.read_csv('data.csv')

# ### Data Exploration
df.head(10)
df.info()
df.shape
df.describe()

# dropping unwanted columns
df_new = df.drop(columns = ['OBJECTID','STATION_TEXT', 'BIKE_COUNT', 'BIKE_YEAR_COUNTED', 'TrafficStationID '])

# ### Missing Value Treatment 

df_new.isnull().sum()

# Calculate the mean of the column
mean_value_CHRIS_NUMB = df_new['CHRIS_NUMB'].mean()
df_new['CHRIS_NUMB'].fillna(mean_value_CHRIS_NUMB, inplace=True)

# Convert 'FloatColumn' to integers
df_new['CHRIS_NUMB'] = df_new['CHRIS_NUMB'].astype(int)
df_new['STATUS'] = df_new['STATUS'].astype(str)
df_new['STREET_NAME'] = df_new['STREET_NAME'].astype(str)
df_new['PAVETYPE'] = df_new['PAVETYPE'].astype(str)
df_new['FUNCTIONAL_CLASS'] = df_new['FUNCTIONAL_CLASS'].astype(str)


csv_file_path = "cleaned_data.csv"
df_new.to_csv(csv_file_path, index=False)

## Univariate analysis


# Histogram for a numeric column
plt.hist(df_new['STATION_NUMBER'], bins=20)
plt.xlabel('STATION_NUMBER')
plt.ylabel('Frequency')
plt.show()


# Histogram for a numeric column
plt.hist(df_new['TRAFFIC_COUNT'], bins=20)
plt.xlabel('TRAFFIC_COUNT')
plt.ylabel('Frequency')
plt.show()


# Histogram for a numeric column
plt.hist(df_new['CHRIS_NUMB'])
plt.xlabel('CHRIS_NUMB')
plt.ylabel('Frequency')
plt.show()


# Histogram for a numeric column
plt.hist(df_new['TRAFFIC_YEAR_COUNTED'], bins=20)
plt.xlabel('TRAFFIC_YEAR_COUNTED')
plt.ylabel('Frequency')
plt.show()

# Assuming df is your DataFrame
for column in df_new.select_dtypes(include=['number']):
    plt.hist(df_new[column], bins=20)
    plt.title(f'Histogram of {column}')
    plt.show()


for column in df_new.select_dtypes(include=['number']):
    sns.boxplot(data=df_new, x=column)
    plt.title(f'Box Plot of {column}')
    plt.show()


# In[24]:


for column in df_new.select_dtypes(include=['object']):
    sns.countplot(data=df_new, x=column)
    plt.title(f'Bar Plot of {column}')
    plt.xticks(rotation=90)
    plt.show()


# Assuming df is your DataFrame
numeric_columns = df_new.select_dtypes(include=['number'])
categorical_columns = df_new.select_dtypes(include=['object'])

# Create a 3x3 grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot numeric columns
for i, column in enumerate(numeric_columns.columns):
    plt.sca(axes[i])
    plt.hist(df_new[column], bins=20)
    plt.title(f'Histogram of {column}')

# Plot categorical columns
for i, column in enumerate(categorical_columns.columns):
    plt.sca(axes[i + len(numeric_columns.columns)])
    sns.countplot(data=df_new, x=column)
    plt.title(f'Bar Plot of {column}')
    plt.xticks(rotation=90)

# Hide empty subplots if there are fewer than 9 columns
for i in range(len(numeric_columns.columns) + len(categorical_columns.columns), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


# ### Bivariate analysis

sns.scatterplot(x='PAVETYPE', y='TRAFFIC_COUNT', data=df_new)
plt.show()

sns.barplot(x='FUNCTIONAL_CLASS', y='TRAFFIC_COUNT', data=df_new)
plt.show()

# Separate columns by data type
numeric_columns = df_new.select_dtypes(include=['number'])
categorical_columns = df_new.select_dtypes(include=['object'])

# Perform bivariate analysis for all combinations
for cat_column in categorical_columns.columns:
    for num_column in numeric_columns.columns:
        # Create a bar plot
        plt.figure(figsize=(8, 6))
        sns.barplot(x=cat_column, y=num_column, data=df_new)
        plt.title(f'Bivariate Analysis: {cat_column} vs. {num_column}')
        plt.show()


from scipy.stats import chi2_contingency

# Create a contingency table
contingency_table = pd.crosstab(df_new['PAVETYPE'], df_new['FUNCTIONAL_CLASS'])

# Perform Chi-Square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Print the results
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")


