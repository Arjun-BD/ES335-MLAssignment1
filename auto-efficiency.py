import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn
data = data[data["horsepower"] != "?"]
data["horsepower"] = data["horsepower"].astype('float')
data.drop(columns=["car name"], inplace=True)
X_train , X_test, y_train, y_test = train_test_split(data.iloc[:, 1:], data["mpg"], test_size = 0.3, random_state = 42)
y_test = y_test.reset_index(drop = True)
y_train = y_train.reset_index(drop = True)

classifier = DecisionTree(max_depth = 10, criterion="mse", Type="Regression", discrete_features=False)
classifier.fit(X_train, y_train)
res = classifier.predict(X_test)

print("Root mean square error for our implementation of the decision tree" ,rmse(res, y_test))
print("Mean absolute error for our implementation of the decision tree" ,mae(res, y_test))

cls = DecisionTreeRegressor(max_depth=10)
cls.fit(X_train, y_train)
res = cls.predict(X_test)

y_test = y_test.reset_index(drop = True)
print("Root mean square error for sklearn decision tree",rmse(y_test, res))
print("Mean absolute error for sklearn decision tree" ,mae(res, y_test))