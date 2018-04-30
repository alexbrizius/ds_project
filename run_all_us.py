#!/usr/bin/python


import os
import pydot
import sys

def usage(program):
    print 'Usage: {} league'.format(program)


if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        usage(sys.argv[0])
        sys.exit(1)

    os.system("./parse.py " + sys.argv[1])
    os.system("./overlap.py " + sys.argv[1])
    os.system("./graphresults_us.py " + sys.argv[1])
