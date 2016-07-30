#!/usr/bin/env python

import sys

"""
Expected data format ::  LABEL (or ?) wikipedia-ID example (features)+

"""

def strip_data(istream,outfile):
    """
    Strips comments and metadata from data set before running the algorithm
    """
    for line in istream:
        print >> outfile , "\t".join([line[0]]+line.split()[3:])


if __name__ ==  "__main__":
    infile = open(sys.argv[1])
    out = open('LP.in','w')
    strip_data(infile,out)
    infile.close()
    out.close()
    
    
