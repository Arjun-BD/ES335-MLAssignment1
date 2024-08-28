from typing import Union
import pandas as pd
import math


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    return ((y_hat == y).sum())/len(y)
    pass


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """

    assert y_hat.size == y.size
    
    z = (y == cls)
    z_hat = (y_hat == cls)
    return (((z_hat == True) & (z == True)).sum())/(z_hat.sum())
    pass


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    z = (y == cls)
    z_hat = (y_hat == cls)
    return (((z_hat == True) & (z == True)).sum())/(z.sum())
    pass


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    z = math.sqrt(((y_hat - y)**2).sum()/len(y))
    return z
    pass


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    z = (abs(y_hat - y)).sum()/len(y)
    return z
    pass

