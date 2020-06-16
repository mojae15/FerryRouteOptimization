#ifndef VERTEX_HPP
#define VERTEX_HPP

#include <tuple>
#include "StoredEdge.hpp"
#include <vector>
#include <string>


struct Vertex {
    
    using EList = std::vector<StoredEdge>;
    
public:
    std::tuple<std::string, std::string> coordinates;
    EList out;
    double cost;
    double distToEnd;
    double windSpeed;
    double windDegree;
    double currentVelocity;
    double currentAngle;
    int id;
};

#endif