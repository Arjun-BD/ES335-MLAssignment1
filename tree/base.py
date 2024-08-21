"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

np.random.seed(42)

@dataclass
class Node:
    left : Optional['Node'] = None
    right : Optional['Node'] = None
    split_on : Optional[str] = None
    value : Optional[str] = None

@dataclass
class DecisionTreeClassifier:
    criterion: Literal["entropy", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.curr_depth = max_depth

    def encode_discrete(self, X: pd.DataFrame):
         return one_hot_encoding(X)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, currNode : Node = None) -> None:
        """
        Function to train and construct the decision tree
        """
        if(self.curr_depth == 0):
            currNode.value = y.mode()
        if(self.curr_depth == self.max_depth):
            self.rootNode = Node()
            currNode = self.rootNode
        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 
        split_attr = opt_split_attribute(X,y,self.criterion,X.columns)
        X_true, y_true, X_false , y_false = split_data(X, y , split_attr, 0.5)
        currNode.split_on = split_attr
        currNode.left = Node()
        currNode.rigt = Node()
        self.curr_depth -= 1 
        self.fit(X_true, y_true, currNode.left)
        self.fit(X_false, y_false, currNode.right)        
        self.curr_depth += 1

        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        y = []
        for index, row in X.iterrows():
            node = self.rootNode
            while node.left is not None and node.right is not None:
                if row[node.split_on] < 0.5:
                    node = node.left
                else:
                    node = node.right

            y.append(node.value)

        return pd.Series(y)
        # Traverse the tree you constructed to return the predicted values for the given test inputs.
        pass

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        pass