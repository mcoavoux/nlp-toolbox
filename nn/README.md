Neural Network lib
============
This is a lightweight neural network library designed to be used for Natural Language Processing
and implemented as a set of C++ headers.

It offers feedforward and recurrent network implementations (RNN, LSTM, Stack-LSTM) with the possibility of running it on multiple backends.
including CPU and GPU backends (cuda and opencl). For NLP it offers reasonably efficient lookup layers for encoding discrete symbols.
The whole implementation is designed for parallel execution with minibatches.

The directory provides simple examples of usage via the files `ffwd_test.cpp` and `rnn_test.cpp`

