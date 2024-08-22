"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    return pd.get_dummies(X, dtype = "int64")
    pass

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    if y.dtype == "categorical":
        return True
    else:
        return False

    pass


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    probabilities = Y.value_counts(normalize=True)
    return -np.sum(probabilities * np.log2(probabilities))
    pass


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    probabilities = Y.value_counts(normalize=True)
    return 1 - np.sum(probabilities**2)
    pass


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    if criterion == 'entropy':
        total_entropy = entropy(Y)
        weighted_entropy = sum((len(subset) / len(Y)) * entropy(subset) for _, subset in Y.groupby(attr))
        return total_entropy - weighted_entropy

    elif criterion == 'gini':
        total_gini = gini_index(Y)
        weighted_gini = sum((len(subset) / len(Y)) * gini_index(subset) for _, subset in Y.groupby(attr))
        return total_gini - weighted_gini

    elif criterion == 'mse':
        total_variance = np.var(Y)
        weighted_variance = sum((len(subset) / len(Y)) * np.var(subset) for _, subset in Y.groupby(attr))
        return total_variance - weighted_variance

    pass


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """
    # if(discrete):
        # maxgain = -float('inf')
        # maxattr = None

        # for attr in features:
        #     gain = information_gain(y, X[attr], criterion)
        #     if gain > maxgain:
        #         maxgain = gain
        #         maxattr = attr

        # return maxattr
    
    
    maxattr = None
    maxgain = -float('inf')
    split_point = None

    for attr in features:
            unique_values = sorted(X[attr].unique())
            for i in range(len(unique_values) - 1):
                gain = information_gain(y, X[attr] > (unique_values[i] + unique_values[i+1])/2 , criterion)
                if gain > maxgain:
                    split_point = (unique_values[i] + unique_values[i+1])/2
                    maxgain = gain
                    maxattr = attr

    return maxattr,split_point

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

    pass

def split_value(X: pd.DataFrame, y : pd.Series, attribute):
    X_column = X[attribute]
    X_column


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value, discrete : bool):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    n = len(X)
    X_above = []
    y_above = []
    X_below = []
    y_below = []
    if check_ifreal(X[attribute]) == False:
        for i in range(n):
            if X.iloc[i][attribute] > value:
                X_above.append(X.iloc[i])
                y_above.append(y.iloc[i])
            else:
                X_below.append(X.iloc[i])
                y_below.append(y.iloc[i])
    else:
        for i in range(n):
            if X.iloc[i][attribute] >= value:
                X_above.append(X.iloc[i])
                y_above.append(y.iloc[i])
            else:
                X_below.append(X.iloc[i])
                y_below.append(y.iloc[i])

    X_above = pd.DataFrame(X_above, columns=X.columns)
    y_above = pd.Series(y_above, index=X_above.index)
    X_below = pd.DataFrame(X_below, columns=X.columns)
    y_below = pd.Series(y_below, index=X_below.index)

    if(discrete):
        X_above.drop(attribute, axis = 1, inplace= True)
        X_below.drop(attribute, axis = 1, inplace= True)
        
    return X_above, y_above, X_below, y_below

    pass
