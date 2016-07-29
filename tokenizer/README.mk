README tokenizer
================

Usage
-----
`tokenizer -p <path to the param file dir> <input file> <output file>`

The tokenizer writes its output to output file, one paragraph per line. 
An additional flag `-c` tells the tokenizer to output one token per line.
paragraphs are separated by a blank line.

The tokenizer has been designed for French. It should be less accurate for other languages !

Compilation
-----------

`g++ -std=c++11 -O3 -o tokenizer tokenizer.cpp -lboost_regex-mt`

At the moment, for efficiency reasons, the `boost` headers and library should be made available to the compiler
which may require to add additional flags `-I` and `-L` before compiling.

The segmenter is reasonably efficient (around 100 000 toks/second on standard hardware) but takes some (reasonable) time at startup for compiling automata. Thus it is more efficient to use it on larger files.

Design Principle
----------------

The tokenizer does a job similar to PTB style tokenizers and it is largely compliant with French treebank parsers. First it splits the input into white space separated pseudo-tokens. Punctuation symbols are independent tokens except hyphens that are glued at the right of existing tokens. Then the segmenter progresses in the paragraph from left to right, greedily generating one token at a time. It generates tokens from pseudo tokens using the following rules in the given order:

1. Merges strong compounds, these are tokens such as _aujourd' hui_ or _arc - en -ciel_ that are getting merged as _aujourd'hui_ and _arc-en-ciel_ by suppressing whitespace between pseudo-tokens.
2. Merges weak compounds, these are tokens such as *c' est - à dire* or *d' autre part* that are getting merged as *c'_est_-_à_dire* or *d'_autre part*. For weak compounds the whitespaces are replaced by an _.
3. If no compound is found generate a token from the current point to the next white space.

The compounds are listed in dictionaries taken as parameters by the segmenter. The segmenter will in general not replace any symbol found in the input. For instance a number or a date will not be replaced by a special reserved symbol such as *NUM* or *DATE*. This is left for further processing modules (such as NER modules). There are few exceptions though : some typographic symbols (quotes) are normalized before segmentation.








