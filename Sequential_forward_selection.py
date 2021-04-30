## importing all the libraries
import numpy as np
import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier

## Reading the dataset and converting it into numpy array
df = pd.read_csv("UCI_Credit_Card.csv")
data = df.to_numpy()

## Last column of the dataset is out labels
X = data[:,:-1]
Y = data[:,-1]
n_features = X.shape[1]


## Function to normalize the data
def normalize_data(X):
    features = X.shape[1]  ### number of features
    new_data = np.zeros((X.shape))
    
    ## Dividing by the max value of that column to scale it to close to -1 to 1
    for i in range(features):
        new_data[:,i] = X[:,i]/abs(max(X[:,i],key=abs)) 
    return new_data


def stratified_k_fold(data, n_fold):
    
    ## Considering last column contains the labels
    Y = data[:,-1]
    ### Total number of classes
    classes = np.unique(Y)
    n_class = len(classes)
    class_samp = []
    n_class_samp_perfold = []
    
    for i in classes:
        ## getting samples which have label i
        samples_in_class = data[np.where(Y==i)]
        ## Shuffling the samples
        random.shuffle(samples_in_class)
        class_samp.append(samples_in_class)
        ## Calculating samples required in per fold from this class
        ## to maintain the percent of samples from each class in every fold
        samples_perfold = int(len(samples_in_class)/n_fold)
        n_class_samp_perfold.append(samples_perfold)
    dataset_split = []

    ## We need to maintain the percent of samples from each class 
    ## in every fold    
    for i in range(n_fold):
        fold = []
        for j in range(n_class):
            n_samp = n_class_samp_perfold[j]
            if(i!=n_fold-1):
                fold.extend(class_samp[j][i*n_samp:(i+1)*n_samp])
            else:
                fold.extend(class_samp[j][i*n_samp:])
        random.shuffle(fold)
        dataset_split.append(fold)
    return dataset_split


## Fucntion to give a set of train test with the given kth fold
def data_preparation(data_splits, k, n_folds):
    test = data_splits[k]
    arr = np.arange(n_folds)
    arr = arr.tolist()
    arr.remove(k)
    train = data_splits[arr[0]]
    for i in range(1,len(arr)):
        train = np.concatenate((train, data_splits[arr[i]]))
    test = np.array(test)
    X_train = train[:,:-1]
    X_test = test[:,:-1]
    Y_train = train[:,-1]
    Y_test = test[:,-1]
    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)
    return([X_train, X_test, Y_train, Y_test])
    
    
    
def fit(data_splits, n_folds):
    ### Getting the splits, data_fold contains n_folds with
    ### X_train, X_test, Y_train and Y_test arrays using stratified sampling
    data_folds = []
    for i in range(n_folds):
        data_folds.append(data_preparation(data_splits, i, n_folds))
    
    ## array containing the features from sequential forward selection
    feature = []
    score = []
    ### First feature selection
    for i in range(n_features):
        ## accuracy of every features
        accuracy = []
        for j in range(n_folds):
            ## data_fold[j] is used for stratifies-k-fold-crossvalidation
            [X_train, X_test, Y_train, Y_test] = data_folds[j]
            ## Decision tree classifier 
            clf = DecisionTreeClassifier(max_depth = 2).fit(X_train[:,i].reshape(-1,1), Y_train)
            ### Accuracy of decision tree on test set for the particular fold
            accuracy.append(clf.score(X_test[:,i].reshape(-1,1), Y_test))
        ## taking mean of all the folds accuracies with the feature
        score.append(np.mean(accuracy))
    ## Considering the feature with maximum accuracy
    feature.append(np.argmax(score))
    
    ## array containing the accuracies from those features
    acc_scores = []
    
    ## While loop till 75% features selected 
    while(len(feature) < 0.75*n_features):
        score = []
        ## Looping over all features to add with the current list of feature
        for i in range(n_features):
            ## Can not consider features which are already in the feature list
            ## so just assiging accuracy 0
            if(i in feature):
                score.append(0)
                continue
            accuracy = []
            ## copying the current optimal feature list
            feat_list = feature.copy()
            ## adding the feature in the feature list to train
            feat_list.append(i)
            ## stratified-K-fold-cross-validation
            for j in range(n_folds):
                [X_train, X_test, Y_train, Y_test] = data_folds[j]
                ## decision tree
                clf = DecisionTreeClassifier().fit(X_train[:,feat_list], Y_train)
                accuracy.append(clf.score(X_test[:,feat_list], Y_test))
            score.append(np.mean(accuracy))
        print(np.max(score))
        print(feature)
        ## accuracy score from the current optimal feature list
        acc_scores.append(np.max(score))
        ## appending the feature to the current optimal list 
        feature.append(np.argmax(score))
        
    ### we only give the index of features which had highest accuracy
    indx = np.argmax(acc_scores)  
    
    return(feature[:indx+1])

data_splits = stratified_k_fold(data, 5)
#fit(data_splits, 5)