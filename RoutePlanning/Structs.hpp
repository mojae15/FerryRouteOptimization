#ifndef STRUCTS_HPP
#define STRUCTS_HPP

#include <vector>

struct StoredEdge {

public:
    StoredEdge( Vertex src, Vertex tar ) : src( src ), tar( tar ) {}
    Vertex src, tar;
};

struct Vertex {

    using EList = std::vector<StoredEdge>;
    

public:
    EList out;
    int cost;
    int id;
};


#endif