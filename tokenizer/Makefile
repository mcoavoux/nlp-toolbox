CXX = clang++
CFLAGS = -std=c++11 -I /opt/local/include -I ../shared
LDFLAGS = -L /opt/local/lib -lboost_regex-mt
OLVL = -O3 


SRC = tokenizer.cpp tokenizer_standalone.cpp
OBJS = tokenizer.o tokenizer_standalone.o
TOK = tokenizer
 
all : $(TOK) 

$(TOK) : $(OBJS)
	$(CXX) $(LDFLAGS) $(OLVL) -o $(TOK) $(OBJS)

%.o : %.cpp
	$(CXX) $(CFLAGS) $(OLVL) -o $@ -c $<

clean :
	rm -f *.o
