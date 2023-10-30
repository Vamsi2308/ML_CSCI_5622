
# # Naive Bayes


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns

from sklearn import preprocessing
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
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
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix


# importing the dataset
data = pd.read_csv('cleaned_data.csv')
data.head(10)

# Checking for the data
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


data.head(10)

data.columns

# Specify the columns to drop
columns_to_drop = ["STATION_NUMBER", "TRAFFIC_COUNT", "CHRIS_NUMB","CHRIS_NUMB_BINNED","STATION_NUMBER_BINNED"]
# columns_to_drop = ["STATION_NUMBER", "TRAFFIC_COUNT", "CHRIS_NUMB","CHRIS_NUMB_BINNED","STATION_NUMBER_BINNED","TRAFFIC_COUNT_BINNED"]

# Drop the specified columns
data_new = data.drop(columns=columns_to_drop)

# Define features and target variable
X = data_new.drop(columns=["PAVETYPE","STREET_NAME"])
# X = data_new.drop(columns=["FUNCTIONAL_CLASS","STREET_NAME"])
y = data_new['PAVETYPE']


# Preprocess the data
# Encode categorical variables using LabelEncoder
label_encoder = LabelEncoder()
# Labelling categorical features
X['STATUS'] = label_encoder.fit_transform(X['STATUS'])
X['TRAFFIC_COUNT_BINNED'] = label_encoder.fit_transform(X['TRAFFIC_COUNT_BINNED'])
X['FUNCTIONAL_CLASS'] = label_encoder.fit_transform(X['FUNCTIONAL_CLASS'])
X['TRAFFIC_YEAR_COUNTED'] = label_encoder.fit_transform(X['TRAFFIC_YEAR_COUNTED'])

# # 2nd best
# # Labelling categorical features
# X['STATUS'] = label_encoder.fit_transform(X['STATUS'])
# # X['TRAFFIC_COUNT_BINNED'] = label_encoder.fit_transform(X['TRAFFIC_COUNT_BINNED'])
# X['FUNCTIONAL_CLASS'] = label_encoder.fit_transform(X['FUNCTIONAL_CLASS'])
# X['TRAFFIC_YEAR_COUNTED'] = label_encoder.fit_transform(X['TRAFFIC_YEAR_COUNTED'])

X.head(10)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Feature Scaling
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)


# ### Guassian NB

# Create and train the Naive Bayes classifier
clf = GaussianNB()
clf.fit(X_train, y_train)
# Make predictions on the testing set
y_pred = clf.predict(X_test)
# Evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy GNB:", accuracy)
print("Classification Report GNB:", classification_report(y_test, y_pred))
# Calculate the confusion matrix
confusion = confusion_matrix(y_test, y_pred)
# Print the confusion matrix
print("Confusion Matrix GNB:")
print(confusion)
# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()


# ### Multinomial NB


# Create and train the Multinomial Naive Bayes classifier
clf_1 = MultinomialNB()
clf_1.fit(X_train, y_train)
# Make predictions on the testing set
y_pred_1 = clf_1.predict(X_test)
# Evaluate the model
accuracy_1 = accuracy_score(y_test, y_pred_1)
report_1 = classification_report(y_test, y_pred_1)
# Print the results
print("Accuracy for MNB:", accuracy_1)
print("Classification Report for MNB:\n", report_1)
# Calculate the confusion matrix
confusion_1 = confusion_matrix(y_test, y_pred_1)
# Print the confusion matrix
print("Confusion Matrix MNB:")
print(confusion)
# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# ### Archive

# # Sample data
# data_1 = ["TRAFFIC_YEAR_COUNTED"]
# # Create a LabelEncoder
# label_encoder = LabelEncoder()
# # Fit the encoder to the data and transform the data
# encoded_data = label_encoder.fit_transform(data_1)
# # Get the label classes
# label_classes = label_encoder.classes_
# # Create a dictionary mapping labels to their encoded values
# label_to_value = {label: value for label, value in zip(label_classes, range(len(label_classes)))}
# # Print the label-to-value mapping
# print(label_to_value)


# # Split the data into a training set and a testing set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # Feature extraction (for text data, in this case 'STREET_NAME')
# vectorizer = CountVectorizer()
# X_train_street = vectorizer.fit_transform(X_train['STREET_NAME'])
# X_test_street = vectorizer.transform(X_test['STREET_NAME'])
# # # Combine numerical features with the extracted 'STREET_NAME' features
# # X_train_combined = X_train[['STATION_NUMBER', 'TRAFFIC_COUNT', 'TRAFFIC_YEAR_COUNTED', 'CHRIS_NUMB', 'PAVETYPE', 'FUNCTIONAL_CLASS']].join(pd.DataFrame(X_train_street.toarray(), columns=vectorizer.get_feature_names_out()))
# # X_test_combined = X_test[['STATION_NUMBER', 'TRAFFIC_COUNT', 'TRAFFIC_YEAR_COUNTED', 'CHRIS_NUMB', 'PAVETYPE', 'FUNCTIONAL_CLASS']].join(pd.DataFrame(X_test_street.toarray(), columns=vectorizer.get_feature_names_out()))
# # Combine numerical features with the extracted 'STREET_NAME' features
# X_train_combined = X_train[['STATUS','FUNCTIONAL_CLASS']].join(pd.DataFrame(X_train_street.toarray(), columns=vectorizer.get_feature_names_out()))
# X_test_combined = X_test[['STATUS', 'FUNCTIONAL_CLASS']].join(pd.DataFrame(X_test_street.toarray(), columns=vectorizer.get_feature_names_out()))

# # Create and train the Multinomial Naive Bayes model
# clf = MultinomialNB()
# clf.fit(X_train_combined, y_train)

# # Make predictions
# y_pred = clf.predict(X_test_combined)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)

# print(f'Accuracy: {accuracy}')
# print(report)


# In[25]:


# # Data preprocessing - Encode categorical variables, define features and target
# # For example, let's assume you want to predict the "STATUS" based on other features.
# X = data.drop(columns=["STATUS"])
# # y = data["STATUS"]

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train the Naive Bayes classifier
# clf = GaussianNB()
# clf.fit(X_train, y_train)

# # Make predictions on the testing set
# y_pred = clf.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# print(classification_report(y_test, y_pred))

