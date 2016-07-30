Label propagation for WSD
=========================

This is a label propagation utility script designed to handle reasonably large graphs with GPU computations. It is designed to easily encode NLP data sets and can use external word embeddings dictionaries to do so.

Input format 
------------
The input is a fixed size column format where columns are whitespace separated, with first column a Y label for seed example or '?' for unlabelled examples. Second column is a unique example identifier. Remaining columns are expected to be discrete X symbols.




