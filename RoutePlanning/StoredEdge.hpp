#ifndef STOREDEDGE_HPP
#define STOREDEDGE_HPP

struct StoredEdge {

public:
    StoredEdge( int src, int tar ) : src( src ), tar( tar ) {}
    int src, tar;
};

#endif