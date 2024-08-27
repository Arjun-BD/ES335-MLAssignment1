# Decision Tree Classification with Nested Cross-Validation

## Overview

We tested the implementation of a Decision Tree classifier to perform binary classification on a synthetic dataset. The model is evaluated using nested cross-validation to determine the optimal tree depth, and the performance metrics (accuracy, precision, and recall) are calculated. Additionally, the classifier's performance using both the entropy and Gini index criteria is compared.

## Code Explanation

### 1. Importing Necessary Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
```

### 2. Defining Cross Validation Function

```python
def crossvalidation(X, y):
    temp = pd.concat([X, y], axis=1)
    temp = temp.sample(frac=1)

    X = temp.iloc[:, :-1]
    y = temp.iloc[:, -1]
    k = 5


    accuracies = []
    fold_size = len(X) // k

    for i in range(k):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        test_set = X[test_start:test_end]
        test_labels = y[test_start:test_end]

        training_set = np.concatenate((X[:test_start], X[test_end:]), axis=0)
        training_labels = np.concatenate((y[:test_start], y[test_end:]), axis=0)

        training_set = pd.DataFrame(training_set).reset_index(drop=True)
        training_labels = pd.Series(training_labels).reset_index(drop=True)

        test_set = pd.DataFrame(test_set).reset_index(drop=True)
        test_labels = pd.Series(test_labels).reset_index(drop=True)

        accuracies_cross_validation = []

        for j in range(k):
            inner_fold_size = int(len(training_set) / k)
            validation_start = j * inner_fold_size
            validation_end = (j + 1) * inner_fold_size
            validation_set = training_set[validation_start:validation_end]
            validation_labels = training_labels[validation_start:validation_end]
            validation_set = pd.DataFrame(validation_set).reset_index(drop=True)
            validation_labels = pd.Series(validation_labels).reset_index(drop=True)

            inner_training_set = np.concatenate((training_set[:validation_start], training_set[validation_end:]), axis=0)
            inner_training_labels = np.concatenate((training_labels[:validation_start], training_labels[validation_end:]), axis=0)
            inner_training_set = pd.DataFrame(inner_training_set).reset_index(drop=True)
            inner_training_labels = pd.Series(inner_training_labels).reset_index(drop=True)

            accuracy_row = []

            for depth in range(2, 9):
                classifier = DecisionTree(max_depth=depth, criterion="gini_index", Type="Classification", discrete_features=False)
                classifier.fit(inner_training_set, inner_training_labels)
                res = classifier.predict(validation_set)
                accuracy_row.append(accuracy(validation_labels, res))

            accuracies_cross_validation.append(accuracy_row)

        accuracies_mean_cross_validation = np.mean(np.array(accuracies_cross_validation), axis=0)
        optimal_depth = np.argmax(accuracies_mean_cross_validation) + 2

        print(f"Fold {i+1}: Optimal Depth: {optimal_depth:.4f}")

        dt_classifier = DecisionTree(max_depth=optimal_depth, criterion="gini_index", Type="Classification", discrete_features=False)
        dt_classifier.fit(training_set, training_labels)

        fold_predictions = dt_classifier.predict(test_set)
        fold_accuracy = np.mean(fold_predictions == test_labels)


        accuracies.append(fold_accuracy)

    for i in range(k):
        print(f"Fold {i+1}: Accuracy: {accuracies[i]:.4f}")
```

#### Explanantion of the function

<ul>
<li>Firstly, the dataset is shuffled to ensure that there is no bias in the training dataset or the testing points</li>

<li>The number of Folds is defined to be 5 and the fold size is figured out by dividing the length of the dataframe by the number of folds </li>

<li> We then iterate through each fold,(each with different testing and training datasets)</li>

<li>We then perform nested cross validation with 5 validation folds</li>

<li>Within each validation fold, we train decision trees from depth 2 to 8, and store the accuracies as an array in the `accuracies_cross_validation` variable.</li>

<li>After iterating through all the validataion folds, we find the optimal depth for a decision tree for that fold by taking the mean across the accuracies of the predicted values given on each of the validataion folds</li>

<li>After figuring out the optimal depth, a new decision tree of the optimal dpeth is trained on both the train and validatation set. This decision tree is then tested for accuract\y on the test set, and this accuracy is reported</li>
</ul>

### Creating Dataset

The code the create the dataset:

```python
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)
```

### Testing implemeneted Decision Tree

The created dataset is split into a 70 - 30 test train split. The decision tree is trained on the train data with depth 10 (arbitrary depth). For the criterion hyper parameters, both gini and entropy are taken to train different decision trees, ( Which in the end gave the same accuracy on the test dataset)

The metrics such as accuracy precision and recall are outputted and are as follows :

<center>
    <img src = "../Graphs/classification_exp_results.png">
</center>
