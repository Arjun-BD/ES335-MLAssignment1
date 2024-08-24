import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values


# Function to create fake data (take inspiration from usage.py)
# ...
def generate_fake_data(type : str , N : int , M : int):
    if(type == 'RIRO'):
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randn(N))
        
    elif(type == "DIRO"):
        X = pd.DataFrame({i: pd.Series(np.random.randint(2, size=M), dtype="category") for i in range(N)})
        y = pd.Series(np.random.randn(N))
    
    elif(type == "RIDO"):
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randint(2, size=N), dtype="category")

    elif(type == "DIDO"):
        X = pd.DataFrame({i: pd.Series(np.random.randint(2, size=M), dtype="category") for i in range(N)})
        y  = pd.Series(np.random.randint(2, size=N), dtype="category")

    else:
        raise ValueError
    
    return X, y
    

# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
def calc_time(type : str, N : int , M : int, depth : int):
    X, y = generate_fake_data(type, N, M)
    if(type == "DIRO" or type == "RIRO"):
        t = "Regression"
        criterion = "mse"
    else:
        t = "Classification"
        criterion = "entropy"
    
    if(type == "DIDO" or type == "DIRO"):
        discrete = True
    else:
        discrete = False

    times_learn = []
    classifier = DecisionTree(Type = t , criterion= criterion, discrete_features= discrete, max_depth = depth)
    if(discrete):
            X = classifier.encode_discrete(X)
    for i in range(100):
        start = time.perf_counter()
        classifier.fit(X, y)
        times_learn.append(time.perf_counter() - start)

    
    times_predict = []
    for i in range(100):
        start = time.perf_counter()
        classifier.predict(X)
        times_predict.append(time.perf_counter() - start)


    return np.mean(np.array(times_learn)), np.mean(np.array(times_predict))


n = [10,30,50,70,100]
m = [5,10,15,20,25]

# Function to plot the results
def plot_results_vary_n(n : list, j : str):
        results_fit = []
        results_predict = []
        for i in n:
            time_fit, time_predict = calc_time(j, i, 10, depth=10)
            results_fit.append(time_fit)
            results_predict.append(time_predict)
            print("done")

        plt.figure(figsize=(10,10))
        plt.plot(n, results_fit)
        plt.title("N is varied, keeping M constant at value 10")
        plt.xlabel("N")
        plt.ylabel(f"Time taken to learn the decision tree, type = {j}")
  
        plt.figure(figsize=(10,10))
        plt.plot(n, results_predict)
        plt.title("N is varied, keeping M constant at value 10")
        plt.xlabel("N")
        plt.ylabel(f"Time taken to for the decision tree to predict, type = {j}")
    
        plt.show()


plot_results_vary_n(n, "DIDO")
# ...
# Other functions
# ...
# Run the functions, Learn the DTs and Show the results/plots
