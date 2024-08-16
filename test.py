import pandas as pd
from metrics import *

a = pd.Series([5,7,13,45,66,77,4,55,9])
b = pd.Series([6,7,15,40,66,79,9,49,5])

print(rmse(a,b))