#!/usr/bin/python


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
    for index, row in l1_ordinal.iterrows():
	if row['Name'] in ord_names:
	    continue
	else:
	    ll_ordinal.loc[len(ll_ordinal)] = row

    print numeric_overlap
    print ll_numeric

	
    
