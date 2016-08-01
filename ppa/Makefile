CXX = g++
CFLAGS = -std=c++11 -I ../shared -I ../nn
LDFLAGS = -lafcpu #backend switch : -lafcuda -lafopencl
OLVL = -O3 


SRC = ppa.cpp example_ppa.cpp
OBJS = ppa.o example_ppa.o
PPA = ppa

all : $(PPA) 

$(PPA) : $(OBJS)
	$(CXX) $(LDFLAGS) $(OLVL) -o $(PPA) $(OBJS)

%.o : %.cpp
	$(CXX) $(CFLAGS) $(OLVL) -o $@ -c $<

clean :
	rm -f *.o
	rm -f $(PPA)
