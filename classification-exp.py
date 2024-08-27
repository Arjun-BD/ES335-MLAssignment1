import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

np.random.seed(42)

def crossvalidation(X, y):
    k = 5  
    fold_size = len(X) // k

    accuracies = []

    for i in range(k):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        X_test = X[test_start:test_end].reset_index(drop=True)
        y_test = y[test_start:test_end].reset_index(drop=True)
        
        X_train = pd.concat([X[:test_start], X[test_end:]]).reset_index(drop=True)
        y_train = pd.concat([y[:test_start], y[test_end:]]).reset_index(drop=True)
        
        accuracies_inner_fold = []

        for j in range(k):
            inner_fold_size = len(X_train) // k
            validation_start = j * inner_fold_size
            validation_end = (j + 1) * inner_fold_size
            
            X_val = X_train[validation_start:validation_end].reset_index(drop=True)
            y_val = y_train[validation_start:validation_end].reset_index(drop=True)
            
            X_inner_train = pd.concat([X_train[:validation_start], X_train[validation_end:]]).reset_index(drop=True)
            y_inner_train = pd.concat([y_train[:validation_start], y_train[validation_end:]]).reset_index(drop=True)
            
            accuracy_depths = []

            for depth in range(2, 9):
                classifier = DecisionTree(max_depth=depth, criterion="gini_index", Type="Classification", discrete_features=False)
                classifier.fit(X_inner_train, y_inner_train)
                predictions_val = classifier.predict(X_val)

                accuracy_depths.append(accuracy(y_val, predictions_val))
            
            accuracies_inner_fold.append(accuracy_depths)

        mean_accuracies = np.mean(np.array(accuracies_inner_fold), axis=0)
        optimal_depth = np.argmax(mean_accuracies) + 2 

        print("Fold {}: Optimal Depth: {:.4f}".format(i+1, optimal_depth))
        
        classifier_final = DecisionTree(max_depth=optimal_depth, criterion="gini_index", Type="Classification", discrete_features=False)
        classifier_final.fit(X_train, y_train)
        predictions_test = classifier_final.predict(X_test)
        
        fold_accuracy = accuracy(y_test, predictions_test)
        accuracies.append(fold_accuracy)

    for i in range(k):
        print("Fold {}: Accuracy: {:.4f}".format(i+1, accuracies[i]))

X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5
)

# Plot the dataset
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()


X = pd.DataFrame(X, columns=["x1", "x2"])
y = pd.Series(y)


depth = 10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)


classifier = DecisionTree(max_depth=depth, criterion="entropy", Type="Classification", discrete_features=False)
classifier.fit(X_train, y_train)
res = classifier.predict(X_test)

y_test = y_test.reset_index(drop=True)

print(f"Below are the results using entropy as the criterion and max_depth as {depth}")
print("The accuracy is ", accuracy(y_test, res))
print("The precision in predicting Class 0 is ", precision(y_test, res, 0))
print("The precision in predicting Class 1 is ", precision(y_test, res, 1))
print("The recall in predicting Class 0 is ", recall(y_test, res, 0))
print("The recall in predicting Class 1 is ", recall(y_test, res, 1))


classifier_ = DecisionTree(max_depth=depth, criterion="gini_index", Type="Classification", discrete_features=False)
classifier_.fit(X_train, y_train)
res = classifier_.predict(X_test)

print(f"Below are the results using gini index as the criterion and max_depth as {depth}")
print("The accuracy is ", accuracy(y_test, res))
print("The precision in predicting Class 0 is ", precision(y_test, res, 0))
print("The precision in predicting Class 1 is ", precision(y_test, res, 1))
print("The recall in predicting Class 0 is ", recall(y_test, res, 0))
print("The recall in predicting Class 1 is ", recall(y_test, res, 1))

crossvalidation(X, y)
