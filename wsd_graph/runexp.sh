#!/usr/bin/sh

OUT_DIR=analysis
INFILE=wikipedia.txt 

#recompile propagate with : g++ -std=c++11 -o propagate LP.cpp -lsqlite3 -lafcpu (or -lafcuda or -lafopencl)

mkdir -p $OUT_DIR
python preprocessLP.py $INFILE
./propagate -E embeddings-norm.db -w 1.0 -W 1.0 -k 3 $OUT_DIR
python analyzeLP.py $INFILE $OUT_DIR
cp -f index.html $OUT_DIR/graph.html
