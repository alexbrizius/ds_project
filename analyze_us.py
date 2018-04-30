#!/usr/bin/python

import os
import pydot
import sys

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC 

from imblearn.under_sampling import RandomUnderSampler

def usage(program):
    print 'Usage: {} league'.format(program)


if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        usage(sys.argv[0])
        sys.exit(1)

    # print column labels first 
    print "Accuracy Precision Recall F1-Score"

    table_ordinal = pd.read_csv('./data/' + sys.argv[1] + '_overlap_ordinal.csv')
    table_numeric = pd.read_csv('./data/' + sys.argv[1] + '_overlap_numeric.csv')
    
    features = list(table_ordinal.columns[2:-1])
    target = list(table_ordinal.columns[-1:])[0]

    # Random Undersampling
    rus = RandomUnderSampler(random_state=1)

    X_ordinal, y_ordinal = rus.fit_sample(table_ordinal[features], table_ordinal[target])
    X_numeric, y_numeric = rus.fit_sample(table_numeric[features], table_numeric[target])

    X_train, X_test, y_train, y_test = train_test_split(X_ordinal, y_ordinal, random_state=42)

    dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3)    
    dt_entropy = dt_entropy.fit(X_train, y_train)

    # output a PNG of the decision tree
    dotfile = sys.argv[1] + '_results/' + sys.argv[1] + '_entropy_us.dot'
    png     = sys.argv[1] + '_results/' + sys.argv[1] + '_entropy_us.png'

    export_graphviz(dt_entropy, out_file=dotfile, feature_names=features)
    (tree, ) = pydot.graph_from_dot_file(dotfile)
    tree.write_png(png)
    
    
    # test
    y_predict = dt_entropy.predict(X_test)

    print "Entropy " + str(accuracy_score(y_test, y_predict))[:5] + ' ' + str(precision_score(y_test, y_predict))[:5] + ' ' + str(recall_score(y_test, y_predict))[:5] + ' ' + str(f1_score(y_test, y_predict))[:5]

    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y_numeric, random_state=42)
    dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)    
    dt_gini = dt_gini.fit(X_train, y_train)

    # output a PNG of the decision tree
    dotfile = sys.argv[1] + '_results/' + sys.argv[1] + '_gini_us.dot'
    png     = sys.argv[1] + '_results/' + sys.argv[1] + '_gini_us.png'

    export_graphviz(dt_gini, out_file=dotfile, feature_names=features)
    (tree, ) = pydot.graph_from_dot_file(dotfile)
    tree.write_png(png)

    y_predict = dt_gini.predict(X_test)

    print "Gini " + str(accuracy_score(y_test, y_predict))[:5] + ' ' + str(precision_score(y_test, y_predict))[:5] + ' ' + str(recall_score(y_test, y_predict))[:5] + ' ' + str(f1_score(y_test, y_predict))[:5]
     
    gauss = GaussianNB()
    y_predict = gauss.fit(X_train, y_train).predict(X_test)
    
    print "Bayes " + str(accuracy_score(y_test, y_predict))[:5] + ' ' + str(precision_score(y_test, y_predict))[:5] + ' ' + str(recall_score(y_test, y_predict))[:5] + ' ' + str(f1_score(y_test, y_predict))[:5]

    neuralNet = MLPClassifier(hidden_layer_sizes=(30, 30, 30), learning_rate='constant', learning_rate_init=0.001, max_iter=200)
    y_predict = neuralNet.fit(X_train, y_train).predict(X_test)
    
    print "NeuralNet " + str(accuracy_score(y_test, y_predict))[:5] + ' ' + str(precision_score(y_test, y_predict))[:5] + ' ' + str(recall_score(y_test, y_predict))[:5] + ' ' + str(f1_score(y_test, y_predict))[:5]

    random_forest = RandomForestClassifier(criterion='entropy', max_depth=3)
    X_train, X_test, y_train, y_test = train_test_split(X_ordinal, y_ordinal, random_state=42)
    random_forest.fit(X_train, y_train)
    y_predict = random_forest.predict(X_test)

    print "RandomForest " + str(accuracy_score(y_test, y_predict))[:5] + ' ' + str(precision_score(y_test, y_predict))[:5] + ' ' + str(recall_score(y_test, y_predict))[:5] + ' ' + str(f1_score(y_test, y_predict))[:5]
    
    ada = AdaBoostClassifier(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_ordinal, y_ordinal, random_state=42)
    ada.fit(X_train, y_train)
    y_predict = ada.predict(X_test)
    print "AdaBoost " + str(accuracy_score(y_test, y_predict))[:5] + ' ' + str(precision_score(y_test, y_predict))[:5] + ' ' + str(recall_score(y_test, y_predict))[:5] + ' ' + str(f1_score(y_test, y_predict))[:5]
    
    
    svc = SVC(C=.5)
    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y_numeric, random_state=42)
    svc.fit(X_train, y_train)
    y_predict = svc.predict(X_test)
    print "SVM " + str(accuracy_score(y_test, y_predict))[:5] + ' ' + str(precision_score(y_test, y_predict))[:5] + ' ' + str(recall_score(y_test, y_predict))[:5] + ' ' + str(f1_score(y_test, y_predict))[:5]
    

