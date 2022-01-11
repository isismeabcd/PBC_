# In[] Preprocessing
import pandas as pd
def dataset(file=""):
    if file != "":
        dataset = pd.read_csv(file)
    else:
        dataset = None

    return dataset

# Load Data
dataset = dataset(file="Creditcard_answers.csv")

# In[] Decomposite Dataset into Independent Variables & Dependent Variables
def decomposition(dataset, x_columns, y_columns=[]):
    X = dataset.iloc[:, x_columns]
    Y = dataset.iloc[:, y_columns]

    if len(y_columns) > 0:
        return X, Y
    else:
        return X
    
X, Y = decomposition(dataset, x_columns=[i for i in range(0, 17)], y_columns=[17])

# In[] outlier
import numpy as np
def outlier_percent(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    minimum = Q1 - (1.5 * IQR)
    maximum = Q3 + (1.5 * IQR)
    num_outliers =  np.sum((data < minimum) |(data > maximum))
    num_total = data.count()
    return (num_outliers/num_total)*100

for column in X.columns:
    data = X[column]
    percent = str(round(outlier_percent(data), 2))
    str_=''
    str_+=(f'Outliers in "{column}": {percent}%\n')
    print(str_)

# In[] Feature Scaling (for PCA Feature Selection)

from sklearn.preprocessing import StandardScaler

def feature_scaling(fit_ary, transform_arys=None):
    scaler = StandardScaler()
    scaler.fit(fit_ary.astype("float64"))

    if type(transform_arys) is tuple:
        return (pd.DataFrame(scaler.transform(ary.astype("float64")), index=ary.index, columns=ary.columns) for ary in transform_arys)
    else:
        return pd.DataFrame(scaler.transform(transform_arys.astype("float64")), index=transform_arys.index, columns=transform_arys.columns)

# In[] Feature Selection: SelectKBest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np

class KBestSelector:
    __selector = None
    __significance = None
    __best_k = None
    __strategy = None

    def __init__(self, significance=0.05, best_k="auto"):
        self.significance = significance

        if type(best_k) is int:
            self.__strategy = "fixed"
            self.best_k = best_k
        else:
            self.__strategy = "auto"
            self.best_k = 1

        self.__selector = SelectKBest(score_func=chi2, k=self.best_k)

    @property
    def selector(self):
        return self.__selector

    @property
    def significance(self):
        return self.__significance

    @significance.setter
    def significance(self, significance):
        if significance > 0:
            self.__significance = significance
        else:
            self.__significance = 0.05

    @property
    def best_k(self):
        return self.__best_k

    @best_k.setter
    def best_k(self, best_k):
        if best_k >= 1:
            self.__best_k = best_k
        else:
            self.__best_k = 1

    # auto=False has been deprecated in version 2021-05-19
    def fit(self, x_ary, y_ary, verbose=False, sort=False):
        #Get the scores of every feature
        kbest = SelectKBest(score_func=chi2, k="all")
        kbest = kbest.fit(x_ary, y_ary)
        
        # if auto, choose the best K features
        if self.__strategy == "auto":
            sig_ary = np.full(kbest.pvalues_.shape, self.significance)
            feature_selected = np.less_equal(kbest.pvalues_, sig_ary)
            self.best_k = np.count_nonzero(feature_selected == True)
        
        # if verbose, show additional information
        if verbose:
            print("\nThe Significant Level: {}".format(self.significance))
            p_values_dict = dict(zip(x_ary.columns, kbest.pvalues_))
            print("\n--- The p-values of Feature Importance ---")
            
            # if sorted, rearrange p-values in ascending order
            if sort:
                name_pvalue = sorted(p_values_dict.items(), key=lambda kv: kv[1])
            else:
                name_pvalue = [(k, v) for k, v in p_values_dict.items()]
            
            # Show each feature and its p-value
            for k, v in name_pvalue:
                sig_str = "TRUE  <" if v <= self.significance else "FALSE >"
                sig_str += "{:.2f}".format(self.significance)
                print("{} {:.8e} ({})".format(sig_str, v, k))
            
            # Show how many features have been selected
            print("\nNumber of Features Selected: {}".format(self.best_k))
        
        # Start to select features
        self.__selector = SelectKBest(score_func=chi2, k=self.best_k)
        self.__selector = self.__selector.fit(x_ary, y_ary)
        
        return self

    def transform(self, x_ary):
        # indices=True will return an NDArray of integer for selected columns
        cols_kept = self.selector.get_support(indices=True)
        return x_ary.iloc[:, cols_kept]


selector = KBestSelector(best_k="auto")
X = selector.fit(x_ary=X, y_ary=Y, verbose=True, sort=True).transform(x_ary=X)

# In[] Split Training / TEsting Set
from sklearn.model_selection import train_test_split
import time

def split_train_test(x_ary, y_ary, train_size=0.75, random_state=int(time.time())):
    return train_test_split(x_ary, y_ary, test_size=(1-train_size), random_state=random_state)

X_train, X_test, Y_train, Y_test = split_train_test(x_ary=X, y_ary=Y)

# In[] DecisionTree

from sklearn.tree import DecisionTreeClassifier
import time
import pandas as pd

class DecisionTree:
    __classifier = None
    __criterion = None
    __y_columns = None
    
    def __init__(self, criterion="entropy", random_state=int(time.time())):
        self.__criterion = criterion
        self.__classifier = DecisionTreeClassifier(criterion=self.__criterion, random_state=random_state)
    
    @property
    def classifier(self):
        return self.__classifier
    
    @classifier.setter
    def classifier(self, classifier):
        self.__classifier = classifier
    
    def fit(self, x_train, y_train):
        self.classifier.fit(x_train, y_train)
        self.__y_columns = y_train.columns
        return self
    
    def predict(self, x_test):
        return pd.DataFrame(self.classifier.predict(x_test), index=x_test.index, columns=self.__y_columns)


classifier = DecisionTree()
Y_pred = classifier.fit(X_train, Y_train).predict(X_test)

# In[] Performance
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score


class KFoldClassificationPerformance:
    __k_fold = None
    __x_ary = None
    __y_ary = None
    __classifier = None
    __verbose = None
        
    def __init__(self, x_ary, y_ary, classifier, k_fold=10, verbose=False):
        self.__x_ary = x_ary
        self.__y_ary = y_ary
        self.k_fold = k_fold
        self.__classifier = classifier
        self.verbose = verbose
    
    @property
    def k_fold(self):
        return self.__k_fold
    
    @k_fold.setter
    def k_fold(self, k_fold):
        if k_fold >=2:
            self.__k_fold = k_fold
        else:
            self.__k_fold = 2

    @property
    def verbose(self):
        return self.__verbose
    
    @verbose.setter
    def verbose(self, verbose):
        if verbose:
            self.__verbose = 10
        else:
            self.__verbose = 0
        
    @property
    def classifier(self):
        return self.__classifier
    
    def accuracy(self):
        results = cross_val_score(estimator=self.classifier, X=self.__x_ary, y=self.__y_ary.values.ravel(), scoring="accuracy", cv=self.k_fold, verbose=self.verbose)
        return results.mean()
    
    def recall(self):
        def recall_scorer(estimator, X, y):
            return recall_score(y, estimator.predict(X), average="macro")
        
        results = cross_val_score(estimator=self.classifier, X=self.__x_ary, y=self.__y_ary.values.ravel(), scoring=recall_scorer, cv=self.k_fold, verbose=self.verbose)
        return results.mean()

    def precision(self):
        def precision_scorer(estimator, X, y):
            return precision_score(y, estimator.predict(X), average="macro")
        
        results = cross_val_score(estimator=self.classifier, X=self.__x_ary, y=self.__y_ary.values.ravel(), scoring=precision_scorer, cv=self.k_fold, verbose=self.verbose)
        return results.mean()

    def f_score(self):
        def f1_scorer(estimator, X, y):
            return fbeta_score(y, estimator.predict(X), beta=1, average="macro")
        
        results = cross_val_score(estimator=self.classifier, X=self.__x_ary, y=self.__y_ary.values.ravel(), scoring=f1_scorer, cv=self.k_fold, verbose=self.verbose)
        return results.mean()

K = 10
kfp = KFoldClassificationPerformance(x_ary=X, y_ary=Y, classifier=classifier.classifier, k_fold=K)

print("----- Decision Tree Classification -----")
print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))

# In[] Visualization
GRAPHVIZ_INSTALL = "C:/Users/liu/Desktop/Graphviz/bin"

from sklearn import tree
import pydotplus
import os
import pylab
import PBC80.model_drawer as md
from IPython.display import Image, display
from pydotplus import graph_from_dot_data

cls_name = ['Group1','Group2','Group3','Group4']
graph = md.tree_drawer(classifier=classifier.classifier, feature_names=X_test.columns, target_names=cls_name, graphviz_bin=GRAPHVIZ_INSTALL)
graph.write_png('treewenwen.png')
