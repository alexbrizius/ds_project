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

def usage(program):
    print 'Usage: {} lowerLeague higherLeague'.format(program)


if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        usage(sys.argv[0])
        sys.exit(1)

    l1_table_ordinal = pd.read_csv('./data/' + sys.argv[1] + '_overlap_ordinal.csv')
    l1_table_numeric = pd.read_csv('./data/' + sys.argv[1] + '_overlap_numeric.csv')

    l2_table_ordinal = pd.read_csv('./data/' + sys.argv[2] + '_overlap_ordinal.csv')
    l2_table_numeric = pd.read_csv('./data/' + sys.argv[2] + '_overlap_numeric.csv')

    # Find players that appear in both
    numeric_overlap = pd.merge(l1_table_numeric, l2_table_numeric, how='inner', on=['Name'])
    oridinal_overlap = pd.merge(l1_table_ordinal, l2_table_ordinal, how='inner', on=['Name'])

    # Find player that appear only in the lower league
    ll_numeric = l1_table_numeric[~l1_table_numeric['Name'].isin(l2_table_numeric)].dropna()
    ll_ordinal = l1_table_ordinal[~l1_table_ordinal['Name'].isin(l2_table_ordinal)].dropna()

    print ll_numeric
	
    
