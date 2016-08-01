CXX = g++
CFLAGS = -std=c++11 -I ../shared
LDFLAGS = -lafcpu #backend switch : -lafcuda -lafopencl
OLVL = -O3 


SRC = example_ffwd.cpp example_rnn.cpp
FFWD_OBJS = example_ffwd.o
RNN_OBJS = example_rnn.o
FFWD = ffwd_test
RNN = rnn_test
 
all : $(FFWD) $(RNN) 

$(FFWD) : $(FFWD_OBJS)
	$(CXX) $(LDFLAGS) $(OLVL) -o $(FFWD) $(FFWD_OBJS)

$(RNN) : $(RNN_OBJS)
	$(CXX) $(LDFLAGS) $(OLVL) -o $(RNN) $(RNN_OBJS)

%.o : %.cpp
	$(CXX) $(CFLAGS) $(OLVL) -o $@ -c $<

clean :
	rm -f *.o
	rm -f $(FFWD)
	rm -f $(RNN)
