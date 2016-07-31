CXX = g++
CFLAGS = -std=c++11 -I ../shared
LDFLAGS = -lsqlite3 -lafcpu #arrayfire lib -lafcuda for GPU 
OLVL = -O3 

SRC = lp.cpp lp_standalone.cpp
OBJS = lp.o lp_standalone.o
LP = propagate
 
all : $(LP) 

$(LP) : $(OBJS)
	$(CXX) $(LDFLAGS) $(OLVL) -o $(LP) $(OBJS)

%.o : %.cpp
	$(CXX) $(CFLAGS) $(OLVL) -o $@ -c $<

clean :
	rm -f *.o
