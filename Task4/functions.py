import pandas as pd
import numpy as np
import tsfel
from itertools import product
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score
import matplotlib.pyplot as plt

def decision_tree_training(dataframeX,dataframeY,depth = None,random_state= 2):
    clf = tree.DecisionTreeClassifier(max_depth = depth,random_state=random_state)
    clf = clf.fit(dataframeX, dataframeY)
    return clf

def metrics(y_pred, y_test, cond = False):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted',zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    labels = [1,2,3,4,5,6]
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    if cond == True:
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print("\nConfusion Matrix:\n")
        print(cm_df)

    return accuracy,cm

def decision_Tree_predict(clf,test_data_x):
    y_pred = clf.predict(test_data_x)
    y_pred = np.array(y_pred)
    return y_pred


def decision_Tree_plot(clf,y_pred,y_test,features):
    
    plt.figure(figsize=(20, 10))
    tree.plot_tree(clf, filled=True, feature_names=features)
    plt.title('Decision Tree Visualization')
    plt.show()
    acc,cm = metrics(y_pred,y_test,cond=True)
    return acc,cm

def selected_featurizer(dataframeX):
        correlation_matrix = dataframeX.corr()
        threshold = 0.98
        removals = []

        pairs = product(range(len(correlation_matrix.columns)), repeat=2)

        for i, j in pairs:
            if i > j:
                correlation_value = correlation_matrix.iloc[i, j]
                if abs(correlation_value) > threshold and correlation_value != 1:
                    colname_i = correlation_matrix.columns[i]
                    colname_j = correlation_matrix.columns[j]
                    if colname_i not in removals and colname_j not in removals:
                        removals.append(colname_i)

        selected_features = [col for col in dataframeX.columns if col not in removals]

        return selected_features

def bias_variance_plotter(accuracy_dataframe):
    plt.figure(figsize=(10, 6))
    
    plt.plot(accuracy_dataframe['Depth'], accuracy_dataframe['Train Accuracy'], 
             marker='o', label='Train Accuracy', color='blue')
    
    plt.plot(accuracy_dataframe['Depth'], accuracy_dataframe['Test Accuracy'], 
             marker='o', label='Test Accuracy', color='orange')
    
    plt.title('Accuracy vs. Depth')
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()

def decision_tree_raw(X_test,X_train,y_test,y_train,depth):
    flattened_X_train = []
    for sample in X_train:
        flattened_sample = []
        for item in sample:
            for j in item:
                flattened_sample.append(j)
        flattened_X_train.append(flattened_sample)
    
    flattened_X_test = []
    for sample in X_test:
        flattened_sample = []
        for item in sample:
            for j in item:
                flattened_sample.append(j)
        flattened_X_test.append(flattened_sample)
    
    dataframe_train  = pd.DataFrame(flattened_X_train)
    dataframe_test  = pd.DataFrame(flattened_X_test)
    clf = decision_tree_training(dataframe_train,y_train,depth)
    y_pred = decision_Tree_predict(clf,dataframe_test)
    y_pred = np.array(y_pred)
    
    acc_test,cm = decision_Tree_plot(clf,y_pred,y_test,[f'acc_{axis}_{i}' for i in range(1, 501) for axis in 'xyz'])
   
    y_testing = decision_Tree_predict(clf,dataframe_train)
    acc_train,cm_train = metrics(y_testing,y_train)

    return acc_test,acc_train,cm

def decision_tree_raw2(X_test,X_train,y_test,y_train,depth):
    accx = []
    accy =[]
    accz = []
    output = []
 
    for i in range(len(X_train)):
        for j in X_train[i]:
            accx.append(j[0])
            accy.append(j[1])
            accz.append(j[2])
            output.append(y_train[i])
    
    flattened_X_train = []

    for sample in X_train:
        for item in sample:
            flattened_X_train.append(item)

    flattened_X_test = []

    for sample in X_test:
        for item in sample:
            flattened_X_test.append(item)

    Xtrain_dataframe = pd.DataFrame(flattened_X_train, columns=['accx', 'accy', 'accz'])
    Xtest_dataframe = pd.DataFrame(flattened_X_test, columns=['accx', 'accy', 'accz'])
    ytrainout = pd.Series(np.repeat(y_train, 500))
    clf = decision_tree_training(Xtrain_dataframe,ytrainout,depth)

    y_pred = decision_Tree_predict(clf,Xtest_dataframe)
    y_pred = np.array(y_pred)  
    num_rows = int(len(y_pred)/500)
    y_pred_reshaped = y_pred.reshape(num_rows, 500)
    modes = []
    for row in y_pred_reshaped:
        mode = pd.Series(row).mode()
        modes.append(mode.iloc[0])
    aggr_y_pred = np.array(modes)
    acc_test,cm = decision_Tree_plot(clf,aggr_y_pred,y_test,['Acceleration X','Acceleration Y','Acceleration Z'])


    y_pred2 = decision_Tree_predict(clf,Xtrain_dataframe)
    num_rows2 = int(len(y_pred2)/500)
    y_pred_reshaped2 = y_pred2.reshape(num_rows2, 500)
    modes2 = []
    for row in y_pred_reshaped2:
        mode2 = pd.Series(row).mode()
        modes2.append(mode2.iloc[0])
    aggr_y_pred2 = np.array(modes2)
    acc_train,cm_train = metrics(aggr_y_pred2,y_train)

    return acc_test,acc_train,cm


def decision_tree_TSFEL(feature_df,feature_df_test,Y_train,Y_test,depth = None):
    selected_features = selected_featurizer(feature_df)
    feature_df_filtered = feature_df[selected_features]
    feature_df_test_filtered = feature_df_test[selected_features]
    clf =decision_tree_training(feature_df_filtered,Y_train,depth)
    y_pred = decision_Tree_predict(clf,feature_df_test_filtered)
    acc_test,cm = decision_Tree_plot(clf,y_pred,Y_test,feature_df_test_filtered.columns) 

    y_pred2 = decision_Tree_predict(clf,feature_df_filtered)
    acc_train,cm_train = metrics(y_pred2,Y_train)

    return acc_test,acc_train,cm
    
def decision_tree_features(depth = None,cond = True):
    feature_labels = pd.read_csv(r'../HAR/UCI HAR Dataset/features.txt',sep = '\\s+',header=None)
    dataframeX = pd.read_csv(r'../HAR/UCI HAR Dataset/train/X_train.txt',sep = '\\s+',header=None)
    dataframeY = pd.read_csv(r'../HAR/UCI HAR Dataset/train/y_train.txt',sep = '\\s+',header=None)
    test_values_y = pd.read_csv(r'../HAR/UCI HAR Dataset/test/y_test.txt',sep = "\\s+",header=None)
    test_data_X = pd.read_csv(r'../HAR/UCI HAR Dataset/test/X_test.txt',sep = "\\s+",header=None)
    
    selected_features = selected_featurizer(dataframeX)
    # selected_features = dataframeX.columns
    
    filtered_dfX = dataframeX[selected_features]
    features = []
    for i in selected_features:
        features.append(feature_labels[1][i])

    
    clf = decision_tree_training(filtered_dfX,dataframeY,depth)
    
    filtered_testx =test_data_X[selected_features]
    y_test = test_values_y[0].to_numpy()
    y_pred = decision_Tree_predict(clf,filtered_testx)
    if cond:
        acc_test,cm = decision_Tree_plot(clf,y_pred,y_test,features) 
    else:
        acc_test,cm = metrics(y_pred,y_test)

    y_pred2 = decision_Tree_predict(clf,filtered_dfX)
    acc_train,cm_train = metrics(y_pred2,dataframeY)

    return acc_test,acc_train,cm