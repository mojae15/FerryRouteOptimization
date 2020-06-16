CPPFLAGS=
CXXFLAGS=-g -Wall -Wextra -O2 $(SANFLAGS)
LDFLAGS=$(SANFLAGS)
LIBS=

OBJECTS= MinHeap.o RoutePlanning.o

# -fsanitize=undefined does not work for some reason
SANFLAGS=-fsanitize=address -fsanitize=leak

.PHONY: all
all: rp

rp: $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

$(OBJECTS): Vertex.hpp StoredEdge.hpp AdjacencyList.hpp MinHeap.hpp

.PHONY: clean
clean:
	rm -f *.o rp
