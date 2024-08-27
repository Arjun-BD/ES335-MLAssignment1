import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

np.random.seed(42)

def crossvalidation(X,y):
    temp = pd.concat([X,y], axis = 1)
    temp = temp.sample(frac = 1)

    X = temp.iloc[:,:-1]
    y = temp.iloc[:,-1]
    k = 5

    predictions = {}
    accuracies = []
    fold_size = len(X) // k

    for i in range(k):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        test_set = X[test_start:test_end]
        test_labels = y[test_start:test_end]
        
        training_set = np.concatenate((X[:test_start], X[test_end:]), axis=0)
        training_labels = np.concatenate((y[:test_start], y[test_end:]), axis=0)

        training_set = pd.DataFrame(training_set).reset_index(drop = True)
        training_labels = pd.Series(training_labels).reset_index(drop  = True)

        test_set = pd.DataFrame(test_set).reset_index(drop  = True)
        test_labels = pd.Series(test_labels).reset_index(drop  = True)
        
        accuracies_cross_validation = []

        for j in range(k):
            inner_fold_size = int(len(training_set)/k)
            validation_start = j * inner_fold_size
            validation_end = (j + 1) * inner_fold_size
            validation_set = training_set[validation_start:validation_end]
            validation_labels = training_labels[validation_start:validation_end]
            validation_set = pd.DataFrame(validation_set).reset_index(drop  = True)
            validation_labels = pd.Series(validation_labels).reset_index(drop  = True)
            
            inner_training_set = np.concatenate((training_set[:validation_start], training_set[validation_end:]), axis=0)
            inner_training_labels = np.concatenate((training_labels[:validation_start], training_labels[validation_end:]), axis=0)
            inner_training_set = pd.DataFrame(training_set).reset_index(drop  = True)
            inner_training_labels = pd.Series(training_labels).reset_index(drop  = True)
            accuracy_row = []

            for depth in range(2,9):
                classifier = DecisionTree(max_depth=depth, criterion="gini_index", Type="Classification", discrete_features=False)
                classifier.fit(inner_training_set, inner_training_labels)
                res = classifier.predict(validation_set)

                accuracy_row.append(accuracy(validation_labels, res))
            accuracies_cross_validation.append(accuracy_row)

        accuracies_mean_cross_validation = np.mean(np.array(accuracies_cross_validation),axis = 0)

        optimal_depth = np.argmax(accuracies_mean_cross_validation) + 2

        print("Fold {}: Optimal Depth: {:.4f}".format(i+1, optimal_depth))
        
        
        dt_classifier = DecisionTree(max_depth=optimal_depth, criterion="gini_index", Type="Classification", discrete_features=False)
        dt_classifier.fit(training_set, training_labels)
        
        fold_predictions = dt_classifier.predict(test_set)
        
        fold_accuracy = np.mean(fold_predictions == test_labels)
        
        predictions[i] = fold_predictions
        accuracies.append(fold_accuracy)

    for i in range(k):
        print("Fold {}: Accuracy: {:.4f}".format(i+1, accuracies[i]))


# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()
# Write the code for Q2 a) and b) below. Show your results.

X = pd.DataFrame(X, columns=["x1", "x2"])
y = pd.Series(y)
depth = 10
X_train , X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, shuffle = False)
classifier = DecisionTree(max_depth=depth, criterion="entropy", Type="Classification", discrete_features=False)
classifier.fit(X_train, y_train)
res = classifier.predict(X_test)

y_test = y_test.reset_index(drop = True)

print(f"Below are the results using entropy as the criterion and max_depth as {depth}")
print("The accuracy is " , accuracy(y_test, res))
print("The precision in predicting Class 0 is ", precision(y_test, res, 0))
print("The precision in predicting Class 1 is ", precision(y_test, res, 1))
print("The recall in predicting Class 0 is ", recall(y_test, res, 0))
print("The recall in predicting Class 1 is ",recall(y_test, res, 1))
print(f"Below are the results using gini index as the criterion and max_depth as {depth}")


classifier_ = DecisionTree(max_depth=depth, criterion="gini_index", Type="Classification", discrete_features=False)
classifier_.fit(X_train, y_train)
res = classifier_.predict(X_test)
print("The accuracy is " , accuracy(y_test, res))
print("The precision in predicting Class 0 is ", precision(y_test, res, 0))
print("The precision in predicting Class 1 is ", precision(y_test, res, 1))
print("The recall in predicting Class 0 is ", recall(y_test, res, 0))
print("The recall in predicting Class 1 is ",recall(y_test, res, 1))


crossvalidation(X,y)
