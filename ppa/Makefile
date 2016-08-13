CXX = g++

CFLAGS = -fopenmp=libomp -std=c++11 -I ../shared -I ../nn -I/opt/local/include/libomp
LDFLAGS = -fopenmp=libomp -lafcpu -L/opt/local/lib/libomp #backend switch : -lafcuda -lafopencl
OLVL = -O3 

ifeq ($(shell uname -s),Darwin)
	CXX = clang++-mp-3.7   #requires clang >= 3.7 on macosx
endif

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
