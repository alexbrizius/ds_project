#!/usr/bin/python

import matplotlib
# turn off XWindows for matplotlib
matplotlib.use('Agg')

import subprocess
import sys
import re

from StringIO import StringIO
import pandas as pd
import matplotlib.pyplot as plt

# Prints usage for script
def usage(program):
	print 'Usage: {} league'.format(program)

def mpl_reset():
    plt.clf()
    plt.cla()
    plt.close()

# Exit if usage incorrect
if len(sys.argv) != 2:
    usage(sys.argv[0])
    sys.exit(1)

# Capture analysis script output
outputCSV = subprocess.check_output(["./analyze.py", sys.argv[1]])
f = StringIO(outputCSV)
df = pd.read_csv(f, sep=' ')

        
# make a bar chart for each method 
for index, row in df.iterrows():
    #method_name = row['Method']
    print row
    row.plot(kind='bar')
    plt.savefig("hello.png")
    mpl_reset() 
