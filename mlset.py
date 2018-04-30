#!/usr/bin/python

from sklearn.tree import DecisionTreeClassifier, export_graphviz


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import os
import pydot
import sys

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# Usage message
def usage(program):
    print 'Usage: {} lowerLeague higherLeague'.format(program)


### Main Execution ###

if __name__ == '__main__':
    
    # Usage check
    if len(sys.argv) != 3:
        usage(sys.argv[0])
        sys.exit(1)

    # Read in tables
    l1_ordinal = pd.read_csv('./data/' + sys.argv[1] + '_overlap_ordinal.csv')
    l1_numeric = pd.read_csv('./data/' + sys.argv[1] + '_overlap_numeric.csv')

    l2_ordinal = pd.read_csv('./data/' + sys.argv[2] + '_overlap_ordinal.csv')
    l2_numeric = pd.read_csv('./data/' + sys.argv[2] + '_overlap_numeric.csv')

    # Find players that appear in both
    numeric_overlap = pd.merge(l1_numeric, l2_numeric, how='inner', on=['Name'])
    ordinal_overlap = pd.merge(l1_ordinal, l2_ordinal, how='inner', on=['Name'])
   
    # Find players that appear only in the lower league
    ll_numeric = pd.DataFrame(columns=['Name','PA',
					'BB%','K%',
					'AVG','OBP',
					'SLG','OPS',
					'BABIP','wRC+'])
    num_names = list(numeric_overlap['Name'])
    for index, row in l1_numeric.iterrows():
	if row['Name'] in num_names:
	    continue
	else:
	    ll_numeric.loc[len(ll_numeric)] = row

    ll_ordinal = pd.DataFrame(columns=['Name','PA',
					'BB%','K%',
					'AVG','OBP',
					'SLG','OPS',
					'BABIP','wRC+'])
    ord_names = list(ordinal_overlap['Name'])
    
    # lets keepy only columns from A for simplicity
    for column in ordinal_overlap: 
        if '_y' in column:
            del ordinal_overlap[column]
        elif column != 'Name':
            ordinal_overlap = ordinal_overlap.rename(columns={column : column[:-2]})
    del ordinal_overlap['WAR/PA']
    
    for index, row in l1_ordinal.iterrows():
	if row['Name'] in ord_names:
	    continue
	else:
	    ll_ordinal.loc[len(ll_ordinal)] = row
    

    ll_ordinal['success'] = len(ll_ordinal) * [0]
    ordinal_overlap['success'] = len(ordinal_overlap) * [1]

    combined = ll_ordinal 
    for index, row in ordinal_overlap.iterrows():
        combined = combined.append(row)
    
    features = list(combined.columns[2:-1])
    target = list(combined.columns[-1:])[0]


    X_train, X_test, y_train, y_test = train_test_split(combined[features], combined[target], random_state=1)

    dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=5)    
    dt_entropy = dt_entropy.fit(X_train, y_train)

    # output a PNG of the decision tree
    dotfile = 'ml_results/' + sys.argv[1] + '_entropy.dot'
    png     = 'ml_results/' + sys.argv[1] + '_entropy.png'

    export_graphviz(dt_entropy, out_file=dotfile, feature_names=features)
    (tree, ) = pydot.graph_from_dot_file(dotfile)
    tree.write_png(png)
    
    # test
    y_predict = dt_entropy.predict(X_test)
    
    print "Entropy " + str(accuracy_score(y_test, y_predict))[:5] + ' ' + str(precision_score(y_test, y_predict))[:5] + ' ' + str(recall_score(y_test, y_predict))[:5] + ' ' + str(f1_score(y_test, y_predict))[:5]


