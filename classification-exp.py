import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)
print(X[:, 0])
# Write the code for Q2 a) and b) below. Show your results.

X = pd.DataFrame(X, columns=["x1", "x2"])
y = pd.Series(y)

X_train , X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)
classifier = DecisionTree(max_depth=10, criterion="entropy", Type="Classification", discrete_features=False)
classifier.fit(X_train, y_train)
res = classifier.predict(X_test)

y_test = y_test.reset_index(drop = True)

print("Below are the results using entropy as the criterion and max_depth as 10")
print("The accuracy is " , accuracy(y_test, res))
print("The precision in predicting Class 0 is ", precision(y_test, res, 0))
print("The precision in predicting Class 1 is ", precision(y_test, res, 1))
print("The recall in predicting Class 0 is ", recall(y_test, res, 0))
print("The recall in predicting Class 1 is ",recall(y_test, res, 1))
print("Below are the results using gini index as the criterion and max_depth as 10")


classifier = DecisionTree(max_depth=10, criterion="gini_index", Type="Classification", discrete_features=False)
classifier.fit(X_train, y_train)
res = classifier.predict(X_test)
print("The accuracy is " , accuracy(y_test, res))
print("The precision in predicting Class 0 is ", precision(y_test, res, 0))
print("The precision in predicting Class 1 is ", precision(y_test, res, 1))
print("The recall in predicting Class 0 is ", recall(y_test, res, 0))
print("The recall in predicting Class 1 is ",recall(y_test, res, 1))




