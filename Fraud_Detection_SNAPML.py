from __future__ import print_function
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

raw_data = pd.read_csv("creditcard.csv")


# inflate original dataset to simulate a financial intitution
n_replicas = 10
big_raw_data = pd.DataFrame(
    np.repeat(raw_data, n_replicas, axis=0), columns=raw_data.columns
)

# get the set of distinct classes
labels = big_raw_data.Class.unique()


# get the count of each class
sizes = big_raw_data.Class.value_counts().values

# plot the class value counts
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct="%1.3f%%")
ax.set_title("Target Variable Value Counts")
# plt.show()


# Distribuition of the differents amounts CC transactions, plotting 90%
# plt.hist(big_raw_data.Amount.values, 6, histtype="bar", facecolor="g")
# plt.show()

# print("Minimum amount value is ", np.min(big_raw_data.Amount.values))
# print("Maximum amount value is ", np.max(big_raw_data.Amount.values))
# print(
#     "90% of the transactions have an amount less or equal than ",
#     np.percentile(raw_data.Amount.values, 90),
# )

# data processing

# standardize features by removing the mean and scaling to unit variance
big_raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(big_raw_data.iloc[:, 1:30])
data_matrix = big_raw_data.values

x = data_matrix[:, 1:30]
y = data_matrix[:, 30]

# data normalization with l1
x = normalize(x, norm="l1")

# print("x.shape= ", x.shape, "y.shape= ", y.shape)

# train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42, stratify=y
)
print("x train shape= ", x_train.shape, "y train shape= ", y_train.shape)
print("x test shape= ", x_test.shape, "y test shape= ", y_test.shape)


# decision tree classifier using sklearn
