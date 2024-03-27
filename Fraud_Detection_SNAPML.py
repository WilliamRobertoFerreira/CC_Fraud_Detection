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
