## Testing Decision Tree on the automotive efficiency problem

The code used to obtain all our results can be found in `auto-efficiency.py`

The first preprocessing step that we took up was to remove all the rows with junk data, this was particularly common in the `horsepower` column where cars had '?' instead of the horsepower value. Also all these values were strings which werent particularly useful so they were all converted to floating point values. Also the `car name` column was dropped to utilise the Real Input Real Output version of the decison tree on the given data.

```python

data = data[data["horsepower"] != "?"]
data["horsepower"] = data["horsepower"].astype('float')
data.drop(columns=["car name"], inplace=True)

```

The data was then split into training and testing data in a 70 : 30 split and trained on our implementation of the decision tree and sklearn.The code for the same can be found below.

```python
X_train , X_test, y_train, y_test = train_test_split(data.iloc[:, 1:], data["mpg"], test_size = 0.3, random_state = 42)
y_test = y_test.reset_index(drop = True)
y_train = y_train.reset_index(drop = True)

classifier = DecisionTree(max_depth = 10, criterion="mse", Type="Regression", discrete_features=False)
classifier.fit(X_train, y_train)
res = classifier.predict(X_test)

print("Root mean square error for our implementation of the decision tree" ,rmse(res, y_test))
print("Mean absolute error for our implementation of the decision tree" ,mae(res, y_test))

cls = DecisionTreeRegressor(max_depth=10)
cls.fit(X_train, y_train)
res = cls.predict(X_test)

y_test = y_test.reset_index(drop = True)
print("Root mean square error for sklearn decision tree",rmse(y_test, res))
print("Mean absolute error for sklearn decision tree" ,mae(res, y_test))


```

The results we obtained are as follows.

<img src = "../Graphs/auto_efficiency_results.png"/>

We can see that the difference in Mean Absolute error and Root Mean square error from the sklearn implementation and our implementation is pretty small, indicating that our implementation performs well in such regression tasks.
