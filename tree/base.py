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
from tree.utils import *

np.random.seed(42)

@dataclass
class Node: #class to represent the nodes of the decision tree
    left : Optional['Node'] = None
    right : Optional['Node'] = None
    split_on : Optional[str] = None
    value : Optional[any] = None
    split_value : Optional[int] = None

@dataclass
class DecisionTree:
    Type : Literal["Regression", "Classification"]
    criterion: Literal["entropy", "gini_index", "mse"]
    max_depth: int  # The maximum depth the tree can grow to
    discrete_features : bool = False

    def __init__(self, criterion, Type, discrete_features, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.curr_depth = max_depth
        self.discrete_features = discrete_features
        self.Type = Type

    def encode_discrete(self, X: pd.DataFrame): #performs one hot encoding on the discrete features
         return one_hot_encoding(X)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, currNode : Node = None) -> None:
        """
        Function to train and construct the decision tree
        """
        
        if(self.curr_depth == 0 or X.empty): #check if depth has been reached or if the data is empty
            if not len(y) == 0 : 
                if(self.Type == "Classification") : currNode.value = (y.mode())[0]
                else : currNode.value = y.mean()
            return
        
        if(self.curr_depth == self.max_depth): #check if we are at the root node and then initialize the root node
            self.rootNode = Node()
            currNode = self.rootNode

        if(len(set(y.to_list())) == 1): #check if all the labels are the same, so we can stop splitting
            if(self.Type == "Classification") : currNode.value = y.mode()[0]
            else : currNode.value = y.mean()
            return 

    
        split_attr, split_point = opt_split_attribute(X,y,self.criterion,X.columns) #find the best attribute to split on
        if(split_attr is None):
            if(self.Type == "Classification") : currNode.value = y.mode()[0]
            else : currNode.value = y.mean()
            return 
        X_true, y_true, X_false , y_false = split_data(X, y , split_attr, split_point, discrete = self.discrete_features)
        currNode.split_on = split_attr
        currNode.split_value = split_point
        currNode.left = Node()
        currNode.right = Node()
        self.curr_depth -= 1                                         
        self.fit(X_true, y_true, currNode.left)#recursively call the fit function on the left and right nodes
        self.fit(X_false, y_false, currNode.right)        
        self.curr_depth += 1

        pass

    def predict(self, X: pd.DataFrame) -> list:
        """
        Funtion to run the decision tree on test inputs
        """
        y = []
        for index, row in X.iterrows(): #iterate over the rows of the test data and traverse the tree to find the predicted value
            node = self.rootNode
            while node.left is not None and node.right is not None:
                if row.iloc[node.split_on] > node.split_value:
                    node = node.left
                else:
                    node = node.right

            y.append(node.value)

        return pd.Series(y)
        pass

    def plot(self, node : Node = None, indent : str = "") -> None:
        """
        Function to plot the tree
        Note that , 
        Y : Yes
        N : No
        """
        if node is None:
            node = self.rootNode           

        if node.value is not None:
            print(node.value)
            return
        
        if(not self.discrete_features): print("?(" + str(node.split_on) + ">" + str(node.split_value) +  ")")
        else : print("?( is " + node.split_on + ")")
        indent += "   "
    
        print(indent + " Y:", end='  ')
        self.plot(node.left, indent + "  ")

    
        print(indent + " N:", end='  ')
        self.plot(node.right, indent + "  ")
       



