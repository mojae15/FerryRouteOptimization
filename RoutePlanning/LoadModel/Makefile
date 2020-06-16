CPPFLAGS=
CXXFLAGS=-g -Wall -Wextra -O2 $(SANFLAGS)
LDFLAGS=$(SANFLAGS)
LIBS=

OBJECTS= LoadModel.cpp

# -fsanitize=undefined does not work for some reason
SANFLAGS=-fsanitize=address -fsanitize=leak

.PHONY: all
all: lm

lm: $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)


.PHONY: clean
clean:
	rm -f *.o lm
