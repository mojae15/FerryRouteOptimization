#ifndef ADJACENCY_LIST_HPP
#define ADJACENCY_LIST_HPP

#include <iostream>
#include <vector>
#include <cassert>
#include <tuple>
#include <string>

#include "StoredEdge.hpp"
#include "Vertex.hpp"

struct AdjacencyList {

private:

    using EList = std::vector<StoredEdge>;
    using VList = std::vector<Vertex>;

    VList vList;
    EList eList;

    friend bool findEdge(StoredEdge &e, AdjacencyList &g){

        for (auto edge : g.eList){
            if (edge.src == e.src && edge.tar == e.tar){
                return true;
            }
        }
        return false;

    }

public:
    struct EdgeDescriptor {
        int src, tar;
    };


    friend int addVertex( AdjacencyList &g , std::tuple<std::string, std::string> coordinates, int i, int j, int dimensions, double distToEnd, double windSpeed, double windDegree, double currentvelocity, double currentAngle) {

        Vertex v;

        //Just for testing
        v.id = (i*dimensions) + j;
        v.coordinates = coordinates;
        v.distToEnd = distToEnd;
        v.windSpeed = windSpeed;
        v.windDegree = windDegree;
        v.currentVelocity = currentvelocity;
        v.currentAngle = currentAngle;
        g.vList.push_back( v );

        // std::cout << "Added vertex " << v.id << std::endl;
        return v.id;
    }

    friend EdgeDescriptor addEdge(int src, int tar, AdjacencyList &g){

        // std::cout << "Creating edge from " << src << " to " << tar << std::endl;

        // StoredEdge e1(src, tar);
        StoredEdge e2(tar, src);

        if (findEdge(e2, g)) assert(false);
        // g.eList.push_back(e1);
        g.eList.push_back(e2);

        // g.vList[src].out.push_back(e1);
        g.vList[tar].out.push_back(e2);

        return EdgeDescriptor{tar, src};
    }

    friend std::vector<Vertex> neighbors(int target, AdjacencyList &g){

        std::vector<Vertex> neighbors;
        Vertex v = g.vList[target];

        for (auto edge : v.out){
            // std::cout << edge.src << ", " << edge.tar << std::endl; 
            neighbors.push_back(g.vList[edge.tar]);
        }

        return neighbors;
    }

    int size(){
        return vList.size();
    }

    Vertex getVertex(int id){
        return vList[id];
    }
};

#endif