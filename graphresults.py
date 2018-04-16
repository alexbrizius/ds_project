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


delim = ' '
images_path = './images/'

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
df = pd.read_csv(f, sep=delim)

        
# make a bar chart for each method
for index, row in df.iterrows():
    row.plot(kind='bar', title=index, ylim=(0, 1.0))
    plt.tight_layout()
    plt.savefig(images_path + str(index) + '.png')
    mpl_reset() 
