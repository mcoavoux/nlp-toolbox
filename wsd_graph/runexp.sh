#!/usr/bin/sh

OUT_DIR=analysis
INFILE=sample_input.txt 

#recompile propagate with : g++ -std=c++11 -o propagate LP.cpp -lsqlite3 -lafcpu (or -lafcuda or -lafopencl)

mkdir -p $OUT_DIR
python preprocess.py $INFILE
./propagate -w 1.0 -W 1.0 -k 3 $OUT_DIR
#./propagate -E embeddings-norm.db-w 1.0 -W 1.0 -k 3 $OUT_DIR
python postprocess.py $INFILE $OUT_DIR
cp -f index.html $OUT_DIR/graph.html
