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
    np.random.seed(42)
    if(type == 'RIRO'):
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randn(N))
        
    elif(type == "DIRO"):
        X = pd.DataFrame({i: pd.Series(np.random.randint(2, size=N)) for i in range(M)})
        y = pd.Series(np.random.randn(N))
    
    elif(type == "RIDO"):
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randint(M, size=N), dtype="category")

    elif(type == "DIDO"):
        X = pd.DataFrame({i: pd.Series(np.random.randint(2, size=N)) for i in range(M)})
        y  = pd.Series(np.random.randint(M, size=N), dtype="category")

    else:
        raise ValueError
    
    return X, y
    

# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
def calc_time(type : str, N : int , M : int, depth : int):
    np.random.seed(42)
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
    classifier = DecisionTree(Type = t , criterion = criterion, discrete_features= discrete, max_depth = depth)

    for i in range(100):
        start = time.perf_counter()
        classifier.fit(X, y)
        times_learn.append(time.perf_counter() - start)

    
    times_predict = []
    for i in range(100):
        start = time.perf_counter()
        classifier.predict(X)
        times_predict.append(time.perf_counter() - start)


    return np.mean(np.array(times_learn)), np.mean(np.array(times_predict)) , np.std(np.array(times_learn)), np.std(np.array(times_predict))

n = [10,30,50,70,100]
m = [5,10,15,20,25]

# Function to plot the results
def plot_results_vary_n(n : list, j : str):
        np.random.seed(42)
        results_fit = []
        results_predict = []
        results_fit_std = []
        results_predict_std = []
        for i in n:
            time_fit, time_predict, time_fit_std, time_predict_std = calc_time(type = j, N = i,  M = 10, depth=10)
            results_fit.append(time_fit)
            results_predict.append(time_predict)
            results_fit_std.append(time_fit_std)
            results_predict_std.append(time_predict_std)
            print("done")

        plt.figure(figsize=(10,10))
        plt.plot(n, results_fit)
        plt.errorbar(n , results_fit ,yerr = results_fit_std, fmt ='o')
        plt.title("N is varied, keeping M constant at value 10")
        plt.xlabel("N")
        plt.ylabel(f"Time taken to learn the decision tree, type = {j}")
  
        plt.figure(figsize=(10,10))
        plt.plot(n, results_predict)
        plt.errorbar(n , results_predict ,yerr = results_predict_std, fmt ='o')
        plt.title("N is varied, keeping M constant at value 10")
        plt.xlabel("N")
        plt.ylabel(f"Time taken to for the decision tree to predict, type = {j}")
    
        plt.show()

def plot_results_vary_m(n : list, j : str):
        np.random.seed(42)
        results_fit = []
        results_predict = []
        results_fit_std = []
        results_predict_std = []
        for i in n:
            time_fit, time_predict, time_fit_std, time_predict_std = calc_time(type = j, N = 20 , M = i , depth = 10)
            results_fit.append(time_fit)
            results_predict.append(time_predict)
            results_fit_std.append(time_fit_std)
            results_predict_std.append(time_predict_std)
            print("done")

        plt.figure(figsize=(10,10))
        plt.plot(n, results_fit)
        plt.errorbar(n , results_fit ,yerr = results_fit_std, fmt ='o')
        plt.title("M is varied, keeping N constant at value 20")
        plt.xlabel("M")
        plt.ylabel(f"Time taken to learn the decision tree, type = {j}")
  
        plt.figure(figsize=(10,10))
        plt.plot(n, results_predict)
        plt.errorbar(n , results_predict ,yerr = results_predict_std, fmt ='o')
        plt.title("M is varied, keeping N constant at value 20")
        plt.xlabel("M")
        plt.ylabel(f"Time taken to for the decision tree to predict, type = {j}")
    
        plt.show()


##plotting the results for different values of N and M for different types of decision trees
plot_results_vary_m(m, "RIRO")
