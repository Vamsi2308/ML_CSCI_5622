# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

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
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.svm import SVC

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
# y = data_new['FUNCTIONAL_CLASS']


# Preprocess the data
# Encode categorical variables using LabelEncoder
label_encoder = LabelEncoder()
# # Labelling categorical features
X['STATUS'] = label_encoder.fit_transform(X['STATUS'])
X['TRAFFIC_COUNT_BINNED'] = label_encoder.fit_transform(X['TRAFFIC_COUNT_BINNED'])
X['FUNCTIONAL_CLASS'] = label_encoder.fit_transform(X['FUNCTIONAL_CLASS'])
# X['PAVETYPE'] = label_encoder.fit_transform(X['PAVETYPE'])
X['TRAFFIC_YEAR_COUNTED'] = label_encoder.fit_transform(X['TRAFFIC_YEAR_COUNTED'])

# # # 2nd best
# # # Labelling categorical features
# X['STATUS'] = label_encoder.fit_transform(X['STATUS'])
# # X['TRAFFIC_COUNT_BINNED'] = label_encoder.fit_transform(X['TRAFFIC_COUNT_BINNED'])
# X['FUNCTIONAL_CLASS'] = label_encoder.fit_transform(X['FUNCTIONAL_CLASS'])
# # X['PAVETYPE'] = label_encoder.fit_transform(X['PAVETYPE'])
# X['TRAFFIC_YEAR_COUNTED'] = label_encoder.fit_transform(X['TRAFFIC_YEAR_COUNTED'])

X.head(5)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.head()

X_test.head()

y_train.head()

y_test.head()


# ## Kernal function

# Try different cost values for each kernel
cost_values = [0.1, 1, 10]

# Create a subplot for confusion matrix visualizations
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for i, kernel in enumerate(['linear', 'poly', 'rbf']):
    for j, cost in enumerate(cost_values):
        # Create SVM classifier
        svm_classifier = SVC(kernel=kernel, C=cost)

        # Train the SVM classifier
        svm_classifier.fit(X_train, y_train)

        # Make predictions on the testing set
        y_pred = svm_classifier.predict(X_test)

        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Kernel: {kernel}, Cost: {cost}, Accuracy: {accuracy}")

        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2'],
                    yticklabels=['Class 0', 'Class 1', 'Class 2'], ax=axes[i, j])
        axes[i, j].set_title(f"{kernel.capitalize()} Kernel, Cost={cost}")
        axes[i, j].set_xlabel('Predicted')
        axes[i, j].set_ylabel('True')

plt.tight_layout()
plt.show()


# ### SVM - Linear Kernal

# Try different cost values for each kernel
cost_values = [0.1, 1, 10]

for cost in cost_values:
    print(f"\n--- Cost Value: {cost} ---")


    # Create an SVM classifier
    svm_classifier = SVC(kernel='linear', C=cost)  # You can choose different kernels like 'poly' or 'rbf'

    # Train the SVM classifier
    svm_classifier.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = svm_classifier.predict(X_test)

    # Evaluate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Generate a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


# ### SVM  - Polynomial Kernal

# Try different cost values for each kernel
cost_values = [0.1, 1, 10]

for cost in cost_values:
    print(f"\n--- Cost Value: {cost} ---")

    # Polynomial Kernel SVM
    poly_svm_classifier = SVC(kernel='poly', degree=2, C=cost)
    poly_svm_classifier.fit(X_train, y_train)
    y_pred_poly = poly_svm_classifier.predict(X_test)
    accuracy_poly = accuracy_score(y_test, y_pred_poly)
    print(f"Accuracy (Polynomial Kernel): {accuracy_poly}")
    
    # Plot confusion matrices
    plt.figure(figsize=(8, 6))

    # Generate confusion matrices
    cm_poly = confusion_matrix(y_test, y_pred_poly)

    # Polynomial Kernel Confusion Matrix
    sns.heatmap(cm_poly, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2'])
    plt.title('Polynomial Kernel Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()


# ## SVM - RBF (Radial Basis Function) Kernel

# Try different cost values for each kernel
cost_values = [0.1, 1, 10]

for cost in cost_values:
    print(f"\n--- Cost Value: {cost} ---")

    # RBF Kernel SVM
    rbf_svm_classifier = SVC(kernel='rbf', C=cost)
    rbf_svm_classifier.fit(X_train, y_train)
    y_pred_rbf = rbf_svm_classifier.predict(X_test)
    accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
    print(f"Accuracy (RBF Kernel): {accuracy_rbf}")
    
    # Plot confusion matrices
    plt.figure(figsize=(8, 6))

    # Generate confusion matrices
    cm_rbf = confusion_matrix(y_test, y_pred_rbf)
    print(f"Confusion Accuracy (RBF Kernel):\n {cm_rbf}")
    
    # RBF Kernel Confusion Matrix
    sns.heatmap(cm_rbf, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2'])
    plt.title('RBF Kernel Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()


# ## Analysis of Results

import matplotlib.pyplot as plt

# Results from your experiment
kernels = ['linear', 'poly', 'rbf']
cost_values = [0.1, 1, 10]
accuracies = {
    'linear': [0.9076923076923077, 0.9076923076923077, 0.9076923076923077],
    'poly': [0.8769230769230769, 0.8769230769230769, 0.9076923076923077],
    'rbf': [0.7384615384615385, 0.8769230769230769, 0.9076923076923077]
}

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for i, kernel in enumerate(kernels):
    axes[i].plot(cost_values, accuracies[kernel], marker='o', label=f'{kernel.capitalize()} Kernel')
    axes[i].set_title(f'{kernel.capitalize()} Kernel')
    axes[i].set_xlabel('Cost Value')
    axes[i].set_ylabel('Accuracy')
    axes[i].legend()

plt.suptitle('Accuracy vs. Cost for Different Kernels')
plt.show()

