# # Decision Trees

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree
import seaborn as sns

from sklearn import preprocessing
#Import scikit-learn dataset library
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection  import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Read Data
data = pd.read_csv('cleaned_data.csv')
data.head()

# data pre-processing:
data.isnull().sum()
data.dtypes
data = data.dropna()
data.isnull().sum()
data.describe()

# Converting the columns into bins and labelling them.

# Define bin edges and labels for STATION_NUMBER
station_number_bin_edges = [0, 128, 266, 432, 908]
station_number_bin_labels = ["Very Near", "Near", "Moderate", "Far"]
data["STATION_NUMBER_BINNED"] = pd.cut(data["STATION_NUMBER"], bins=station_number_bin_edges, labels=station_number_bin_labels)

# Define bin edges and labels for TRAFFIC_COUNT
traffic_count_bin_edges = [0, 250, 832, 3340, 14863]
traffic_count_bin_labels = ["Very Low", "Low", "Moderate", "High"]
data["TRAFFIC_COUNT_BINNED"] = pd.cut(data["TRAFFIC_COUNT"], bins=traffic_count_bin_edges, labels=traffic_count_bin_labels)

# Define bin edges and labels for CHRIS_NUMB
chris_numb_bin_edges = [0, 2605, 4602, 9404, 890805]
chris_numb_bin_labels = ["Very Low", "Low", "Moderate", "High"]
data["CHRIS_NUMB_BINNED"] = pd.cut(data["CHRIS_NUMB"], bins=chris_numb_bin_edges, labels=chris_numb_bin_labels)


# Specify the columns to drop
columns_to_drop = ["STATION_NUMBER", "TRAFFIC_COUNT", "CHRIS_NUMB","CHRIS_NUMB_BINNED","STATION_NUMBER_BINNED","TRAFFIC_COUNT_BINNED"]

# Drop the specified columns
data_new = data.drop(columns=columns_to_drop)

# Define features and target variable
X = data_new.drop(columns=["PAVETYPE","STREET_NAME"]) # features
# X = data_new.drop(columns=["FUNCTIONAL_CLASS","STREET_NAME"])
y = data_new['PAVETYPE'] # target variable

# Preprocess the data
# Encode categorical variables using LabelEncoder
label_encoder = LabelEncoder()
# 2nd best
# Labelling categorical features
X['STATUS'] = label_encoder.fit_transform(X['STATUS'])
# X['TRAFFIC_COUNT_BINNED'] = label_encoder.fit_transform(X['TRAFFIC_COUNT_BINNED'])
X['FUNCTIONAL_CLASS'] = label_encoder.fit_transform(X['FUNCTIONAL_CLASS'])
X['TRAFFIC_YEAR_COUNTED'] = label_encoder.fit_transform(X['TRAFFIC_YEAR_COUNTED'])

X.head()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Feature Scaling
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")


# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

plt.figure(figsize=(40, 20))
plot_tree(clf, filled=True, feature_names=X.columns)
plt.show()

# Specify the columns to drop
columns_to_drop_1 = ["STATION_NUMBER", "TRAFFIC_COUNT", "CHRIS_NUMB","CHRIS_NUMB_BINNED","STATION_NUMBER_BINNED"]
# Drop the specified columns
data_new_1 = data.drop(columns=columns_to_drop_1)

# Define features and target variable
# X = data_new.drop(columns=["FUNCTIONAL_CLASS","STREET_NAME"])
X_1 = data_new_1.drop(columns=["PAVETYPE","STREET_NAME"]) # features
y_1 = data_new_1['PAVETYPE'] # target variable

# Preprocess the data
# Encode categorical variables using LabelEncoder
label_encoder = LabelEncoder()
# Labelling categorical features
X_1['STATUS'] = label_encoder.fit_transform(X_1['STATUS'])
X_1['TRAFFIC_COUNT_BINNED'] = label_encoder.fit_transform(X_1['TRAFFIC_COUNT_BINNED'])
X_1['FUNCTIONAL_CLASS'] = label_encoder.fit_transform(X_1['FUNCTIONAL_CLASS'])
X_1['TRAFFIC_YEAR_COUNTED'] = label_encoder.fit_transform(X_1['TRAFFIC_YEAR_COUNTED'])

X_1.head(10)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_1, y_1, test_size=0.2, random_state=42)

# # Feature Scaling
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Create a heatmap for the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

plt.figure(figsize=(70, 20))
plot_tree(clf, filled=True, feature_names=X_1.columns)
plt.show()

