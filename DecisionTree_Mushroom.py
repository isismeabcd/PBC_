# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 07:30:11 2019

@author: 俊男
"""

# In[] Preprocessing
import PBC80.preprocessor as pp

def draw_picture(File):
# Load Data
    dataset = pp.dataset(file=File)

# Decomposition
    X, Y = pp.decomposition(dataset, x_columns=[i for i in range(1, 23)], y_columns=[0])

# Dummy Variables
    X = pp.onehot_encoder(X, columns=[i for i in range(22)], remove_trap=True)
    Y, Y_mapping = pp.label_encoder(Y, mapping=True)

# Feature Selection
    from PBC80.preprocessor import KBestSelector
    selector = KBestSelector(best_k="auto")
    X = selector.fit(x_ary=X, y_ary=Y, verbose=True, sort=True).transform(x_ary=X)

# Split Training / TEsting Set
    X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary=X, y_ary=Y)

# In[] Decision Tree
    from PBC80.classification import DecisionTree

    classifier = DecisionTree()
    Y_pred = classifier.fit(X_train, Y_train).predict(X_test)

# In[] Performance
    from PBC80.performance import KFoldClassificationPerformance

    K = 10
    kfp = KFoldClassificationPerformance(x_ary=X, y_ary=Y, classifier=classifier.classifier, k_fold=K)

    '''
    print("----- Decision Tree Classification -----")
    print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy()))
    print("{} Folds Mean Recall: {}".format(K, kfp.recall()))
    print("{} Folds Mean Precision: {}".format(K, kfp.precision()))
    print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))
'''
# In[] Visualization


    from sklearn import tree
    import pydotplus
    from IPython.display import Image, display
    import os
    import graphviz
    cls_name = [Y_mapping[key] for key in sorted(Y_mapping.keys())]
    dot_data = tree.export_graphviz(classifier.classifier, filled=True, feature_names=X_test.columns, class_names=cls_name, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data, format="png")
    graph.render("decision_tree_graphivz")
