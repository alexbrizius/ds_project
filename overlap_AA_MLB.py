#!/usr/bin/python

import os

import numpy as np
import pandas as pd
import scipy.stats as stats

data_path = './data/'
        
if __name__ == '__main__':
            
    AA_df = pd.read_csv(data_path + 'AA.csv')
    MLB_df = pd.read_csv(data_path + 'MLB.csv')

    df = pd.merge(AA_df, MLB_df, how='inner', on=['Name'])
    
    # rename correctly
    df = df.rename(columns=lambda x: x[:-2] + '_AA' if x[-2:] == '_x' else (x[:-2] + '_MLB' if x[-2:] == '_y' else x))

    df.to_csv(data_path + 'overlap_AA_MLB.csv', index=False)
    

