#include "AdjacencyList.hpp"
#include "MinHeap.hpp"
#include <array>
#include <climits>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <regex>

#include <fdeep/fdeep.hpp>


enum STATUS {
    LABELED,
    UNLABELED,
    DONE
};

// https://stackoverflow.com/questions/478898/how-do-i-execute-a-command-and-get-the-output-of-the-command-within-c-using-po
std::string exec( const char *cmd ) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype( &pclose )> pipe( popen( cmd, "r" ), pclose );
    if ( !pipe ) {
        throw std::runtime_error( "popen() failed!" );
    }
    while ( fgets( buffer.data(), buffer.size(), pipe.get() ) != nullptr ) {
        result += buffer.data();
    }
    return result;
}

// https://stackoverflow.com/questions/5888022/split-string-by-single-spaces
size_t split( const std::string &txt, std::vector<std::string> &strs, char ch ) {
    size_t pos = txt.find( ch );
    size_t initialPos = 0;
    strs.clear();

    // Decompose statement
    while ( pos != std::string::npos ) {
        strs.push_back( txt.substr( initialPos, pos - initialPos ) );
        initialPos = pos + 1;

        pos = txt.find( ch, initialPos );
    }

    // Add the last one
    strs.push_back( txt.substr( initialPos, std::min( pos, txt.size() ) - initialPos + 1 ) );

    return strs.size();
}

// https://stackoverflow.com/questions/3932502/calculate-angle-between-two-latitude-longitude-points
static const auto PI = 3.14159265358979323846, diameterOfEarthMeters = 6371.0 * 2 * 1000;

double degreeToRadian( double degree ) { return ( degree * PI / 180 ); };
double radianToDegree( double radian ) { return ( radian * 180 / PI ); };

double CoordinatesToAngle( const double latitude1,
                           const double longitude1,
                           const double latitude2,
                           const double longitude2 ) {
    const auto longitudeDifferenceRadians = degreeToRadian( longitude2 - longitude1 );
    auto latitude1Radian = degreeToRadian( latitude1 ),
         latitude2Radian = degreeToRadian( latitude2 );

    const auto x = std::cos( latitude1Radian ) * std::sin( latitude2Radian ) -
                   std::sin( latitude1Radian ) * std::cos( latitude2Radian ) *
                       std::cos( longitudeDifferenceRadians );
    const auto y = std::sin( longitudeDifferenceRadians ) * std::cos( latitude2Radian );


    double degrees = radianToDegree( std::atan2( y, x ) );
    if (degrees < 0){
        return 360 + degrees;
    } 
    return degrees;
}


long double toRadians(const long double val){
    long double one_deg = (M_PI) / 180;
    return (one_deg * val);
}

//Calc the distance between two vertices
long double distance(Vertex v, Vertex w){
    
    int earthRadiusMeters = 6371;

    long double lat1 = toRadians(std::stod(std::get<0>(v.coordinates)));
    long double lon1 = toRadians(std::stod(std::get<1>(v.coordinates)));

    long double lat2 = toRadians(std::stod(std::get<0>(w.coordinates)));
    long double lon2 = toRadians(std::stod(std::get<1>(w.coordinates)));

    //Haversine formula
    long double dLong = lon2 - lon1 ;
    long double dLat = lat2 - lat1;

    long double a = (sin(dLat / 2) * sin(dLat / 2)) + 
                        cos(lat1) * cos(lat2) *
                        (sin(dLong / 2) * sin(dLong / 2));

    long double c = 2 * atan2(sqrt(a), sqrt(1-a));

    long double d = earthRadiusMeters * c;

    return d;

}

//Calculate how long it will take to go from v to w given speed, in minutes
long double calcTime(Vertex v, Vertex w, double speed, double dist){

    float speedInKMH = speed * 1.85200;
    
    long double timeToCoverDist = (dist/speedInKMH) * 60;

    // std::cout << "Speed in knots: " << speed << std::endl;
    // std::cout << "Speed in meters/h: " << speedInKMH << std::endl;
    // std::cout << "Distance: " << dist << std::endl;

    return timeToCoverDist;
    
}

//Calculate the cost of going between two vertices
std::tuple<double, double, long double, std::tuple<float, float, float, float, float>> calcCost( Vertex u, Vertex v , const fdeep::model model) {

    std::string lat = std::get<0>( v.coordinates );
    std::string lon = std::get<1>( v.coordinates );

    // std::string cmd = "python3 api_calls.py " + lat + " " + lon;

    // std::string feature_res = exec( cmd.c_str() );




    // std::vector<std::string> features;

    // std::vector<std::string> features{"24", "321", "10", "210"};

    // split( feature_res, features, ' ' );

    // float wind_speed = stod( features[0] );
    // float wind_degrees = stod( features[1] );
    // float current_velocity = stod( features[2] );
    // float current_degree = stod( features[3] );
    // float depth = 30;

    float windSpeed = v.windSpeed;
    float windDegree = v.windDegree;
    float currentVelocity = v.currentVelocity;
    float currentAngle = v.currentAngle;
    float depth = 20;

    //Knots
    float speed = 10;


    float heading =  CoordinatesToAngle(std::stod(std::get<0>(u.coordinates)), std::stod(std::get<1>(u.coordinates)), std::stod(std::get<0>(v.coordinates)), std::stod(std::get<1>(v.coordinates)));


    // std::cout << heading << std::endl;

    // std::cout << std::get<0>(v.coordinates) << std::endl;
    // std::cout << std::get<1>(v.coordinates) << std::endl;
    


    // std::cout << prediction << std::endl;
    double cost = __DBL_MAX__;
    
    float best_speed = 0;
    double best_time = 0;


    long double dist = distance(u, v);

    while (speed < 14){

        double new_cost = 0;

        //Input format is:
        // Depth, current_velocity, speed, current_direction, heading, wind_speed, wind_direction

        const auto result = model.predict(
            {fdeep::tensor(fdeep::tensor_shape(static_cast<std::size_t>(7)),
            {depth, currentVelocity, speed, currentAngle, heading, windSpeed, windDegree})});

        std::string prediction = fdeep::show_tensors(result);

        std::smatch match;
        std::regex base(R"(([+-]{0,1}[0-9]+\.[0-9]+))");


        if(std::regex_search(prediction, match, base)) {
            std::ssub_match sub_match = match[1];
            std::string sub_string = sub_match.str();
            new_cost = stod(sub_string);
            // std::cout << sub_string << std::endl;
        }


        long double timeToReach = calcTime(u, v, speed, dist);
        new_cost = new_cost / (15/ (timeToReach*60) );

        if (new_cost < cost){
            cost = new_cost;
            best_speed = speed;
            best_time = timeToReach;
        }

        speed = speed + 0.5;

    }



    // std::cout << "Chosen speed: " << best_speed << std::endl;
    // std::cout << "Time to travel between points: " << calcTime(u, v, best_speed) << std::endl;

    auto data = std::make_tuple(windSpeed, windDegree, currentVelocity, currentAngle, heading);

    return std::make_tuple(cost, best_speed, best_time, data);

}


//Calculate the cost of going between two vertices with a certain speed
double calcCostWithSpeed( Vertex u, Vertex v , float speed, const fdeep::model model) {

    std::string lat = std::get<0>( v.coordinates );
    std::string lon = std::get<1>( v.coordinates );


    float windSpeed = v.windSpeed;
    float windDegree = v.windDegree;
    float currentVelocity = v.currentVelocity;
    float currentAngle = v.currentAngle;
    float depth = 20;


    float heading =  CoordinatesToAngle(std::stod(std::get<0>(u.coordinates)), std::stod(std::get<1>(u.coordinates)), std::stod(std::get<0>(v.coordinates)), std::stod(std::get<1>(v.coordinates)));
    

    double cost = 0;
    const auto result = model.predict(
            {fdeep::tensor(fdeep::tensor_shape(static_cast<std::size_t>(7)),
            {depth, currentVelocity, speed, currentAngle, heading, windSpeed, windDegree})});

    std::string prediction = fdeep::show_tensors(result);

    std::smatch match;
    std::regex base(R"(([+-]{0,1}[0-9]+\.[0-9]+))");


    if(std::regex_search(prediction, match, base)) {
        std::ssub_match sub_match = match[1];
        std::string sub_string = sub_match.str();
        cost = stod(sub_string);
        // std::cout << sub_string << std::endl;
    }
    long double dist = distance(u, v);

    long double timeToReach = calcTime(u, v, speed, dist);
    long double actualCost = cost / (15/ (timeToReach*60) );


    return actualCost;

}


//Dijkstra - if the heuristic if desired change "double distance = dist[u.id] + cost;" to "double distance = dist[u.id] + cost + distHeuristic;"
std::tuple<std::vector<double>, std::vector<Vertex>, std::vector<double>, std::vector<long double>, std::vector<std::tuple<float, float, float, float, float>>> dijkstra( AdjacencyList &g, Vertex src, Vertex tar, int time, const fdeep::model model) {


    //Features for the model
    //For some reason I can't get the depth right now, so it's the same for all predictions
    double DEPTH = 30.0;
    int opened = 0;
    int size = g.size();
    Vertex tmp;

    std::vector<double> dist( size );
    std::vector<double> costs( size );
    std::vector<bool> inserted( size );
    std::vector<double> speeds( size );
    std::vector<long double> times( size );
    std::vector<std::tuple<float, float, float, float, float>> data(size);
    

    std::vector<Vertex> prev( size, tmp );

    dist[src.id] = 0;
    costs[src.id] = 0;

    MinHeap mHeap( size );

    for ( int i = 0; i < size; i++ ) {
        if ( i != src.id ) {

            dist[i] = INT_MAX;
            costs[i] = INT_MAX;
            inserted[i] = false;
            // std::cout << "Set dist for " << i << std::endl;
        }
        // Vertex v = g.getVertex(i);
        // v.cost = dist[i];

        // mHeap.insertKey(v);
    }

    src.cost = dist[0];
    mHeap.insertKey( src );
    inserted[0] = true;

    // std::cout << "Initialized arrays" << std::endl;

    while ( mHeap.getSize() > 0 ) {
        Vertex u = mHeap.extractMin();
        if (inserted[u.id]){
            inserted[u.id] = false;
        }

        // std::cout << "Looking at vertex " << u.id << std::endl;
        if ( u.id == tar.id ) {
            std::cout << opened << std::endl;
            return std::make_tuple( costs, prev, speeds, times, data);
        }

        // std::cout << "Neighbors of " << u.id << std::endl;
        for ( Vertex v : neighbors( u.id, g ) ) {
            


            // std::cout << "src:  " << e.src->id << " target: " << e.target->Aid << std::endl;

            auto res = calcCost(u, v, model);
            double cost = std::get<0>(res);
            double speed = std::get<1>(res);
            long double traversalTime = std::get<2>(res);
            auto data_for_point = std::get<3>(res);

            long double distHeuristic = v.distToEnd;

            double actualCost = costs[u.id] + cost;
            

            double distance = dist[u.id] + cost;

            // std::cout << "Looking at: " << v.id << ", cost: " << cost << std::endl;

            // std::cout << "Dist to " << v.id << ": " << distance << std::endl;
            if ( distance < dist[v.id] ) {
                dist[v.id] = distance;
                costs[v.id] = actualCost; 


                // std::cout << "Settin prev of " << v.id << " to be " << u.id << std::endl;

                prev[v.id] = u;
                speeds[v.id] = speed;
                times[v.id] = times[u.id] + traversalTime;
                data[v.id] = data_for_point;
                v.cost = distance;

                // mHeap.decreaseKey(v.id, distance);

                if (!inserted[v.id]){
                    mHeap.insertKey( v );
                    inserted[v.id] = true;
                }
            }
        }

        // for (Edge e : u.edgeList){

        //     Vertex v = *e.target;
        //     // std::cout << "src:  " << e.src->id << " target: " << e.target->Aid << std::endl;

        //     int distance = dist[u.id] + e.cost;
        //     if (distance < dist[v.id]){
        //         dist[v.id] = distance;
        //         prev[v.id] = u;
        //         v.cost = distance;
        //         mHeap.insertKey(v);
        //     }
        // }
    }

    return std::make_tuple( costs, prev, speeds, times, data);
}


void addEdges( AdjacencyList &g, int i, int j, int dimensions ) {

    // std::cout << "i: " << i << ", j: " << j << ", dimensions: " << dimensions << std::endl;

    int converted_j = ( i * dimensions ) + j;
    int prev = ( ( i - 1 ) * dimensions ) + j;
    int prev2 = ( ( i - 2) * dimensions ) + j;

    // std::cout << "conv: " << converted_j << ", prev: " << prev << std::endl;

    if ( i > 0 ) {

        // std::cout << "Adding edge from " << converted_j << " to " << ((i-1)*dimensions) + j << std::endl;
        //Horizontal edge
        addEdge( converted_j, prev, g );

        //Add edges, two columns back
        if (i > 1){

            // std::cout << "Adding edge from " << converted_j << " to " << prev2 << std::endl;
            addEdge(converted_j, prev2, g);

            if (j > 0){
                
            // std::cout << "Adding edge from " << converted_j << " to " << prev2-1 << std::endl;
                addEdge(converted_j, prev2 - 1, g);

                if (j > 1){

                // std::cout << "Adding edge from " << converted_j << " to " << prev2-2 << std::endl;
                    addEdge(converted_j, prev2 - 2, g);
                }

            }



        }
    

        //Diagonal edges

        // if (j < dimensions-1){

        //     // std::cout << "Adding edge from " << converted_j << " to " << prev+1 << std::endl;
        //     addEdge(converted_j, prev+1, g);
        // }

        if (j > 0){

            // std::cout << "Adding edge from " << converted_j << " to " << prev-1 << std::endl;
            addEdge(converted_j, prev-1, g);
        }
    }

    // std::cout << "Adding edge from " << v.id << " to " << u.id << std::endl;

    if ( j > 0 ) {

        // std::cout << "Adding edge from " << converted_j << " to " << converted_j-1 << std::endl;

        addEdge( converted_j, converted_j - 1, g );

        //Add edges two rows back
        if (j > 1){
            
            // std::cout << "Adding edge from " << converted_j << " to " << converted_j - 2 << std::endl;
            addEdge(converted_j, converted_j - 2, g);

            if (i > 0){
                
            // std::cout << "Adding edge from " << converted_j << " to " << prev-2 << std::endl;
                addEdge(converted_j, prev - 2, g);

            }
        }
    }


}

std::tuple<double, double> increaseSpeeds(int endID, int maxTime, AdjacencyList graph, std::vector<Vertex> path, std::vector<double> speeds, std::vector<long double> times, std::vector<double> costs,  const fdeep::model model){

    long double timeTaken = times[endID];
    

    while (timeTaken > maxTime){

        int curPoint = endID;

        double lowestSpeed = INT_MAX;
        int lowestPoint = -1;

        // Find the lowest speed
        while (curPoint != 0 ){
            
            double curSpeed = speeds[curPoint];

            // std::cout << "Cur point: " << curPoint << ", speed: " << curSpeed << std::endl;

            if (curSpeed < lowestSpeed){
                lowestSpeed = curSpeed;
                lowestPoint = curPoint;

            }

            curPoint = path[curPoint].id;

        }
        
        // Increase the speed for the point with the lowest speed
        double increasedSpeed = speeds[lowestPoint] + 0.5;
        speeds[lowestPoint] = increasedSpeed;
        Vertex lowestVertex = graph.getVertex(lowestPoint);
        Vertex prevVertex = path[lowestPoint];
        // Calculate new time
        double myTime = times[lowestPoint];
        double prevTime = times[prevVertex.id];

        // std::cout << "Time to reach this point: " << myTime << ", time to reach prev point: " << prevTime << std::endl;


        double oldTime = myTime - prevTime;
        double dist = distance(lowestVertex, prevVertex);
        double newTime = calcTime(prevVertex, graph.getVertex(lowestPoint), increasedSpeed, dist);




        // std::cout << "Increasing speed to point " << lowestPoint << " to " << increasedSpeed << std::endl;

        double timeDifference = oldTime - newTime;

        timeTaken = timeTaken - timeDifference;


        times[lowestPoint] = prevTime + newTime;

        // std::cout << "Time before: " << oldTime << ", new time: " << newTime << ", Difference: "  << timeDifference << std::endl;


        // Increase the cost for the point with the lowest speed

        double myCost = costs[lowestPoint];
        double prevCost = costs[prevVertex.id];



        double oldCost = myCost - prevCost;
        double newCost = calcCostWithSpeed(prevVertex, lowestVertex, increasedSpeed, model);
        

        double costDifference = newCost - oldCost;




        costs[lowestPoint] = myCost + costDifference;

        //Update times after the point with the lowest speed

        int tempId = endID;

        while (tempId != lowestPoint){
            times[tempId] = times[tempId] - timeDifference;
            costs[tempId] = costs[tempId] + costDifference;


            tempId = path[tempId].id;
        }

        

    }

    return std::make_tuple(timeTaken, costs[endID]);

}

int main() {


    //Dimensions of vectors
    int x_dimensions = 0;
    int y_dimensions = 0;
    int i = 0;
    int j = 0;
    int maxTime = 25;

    AdjacencyList graph;

    //Load model

    const auto model = fdeep::load_model("fdeep_model.json");


    std::ifstream inFile("data_points.csv");
    std::string curLine;
    std::vector<std::string> splitLine;
    bool firstLine = true;


    std::tuple<std::string, std::string> tmp;

    std::cout.precision( 17 );



    //Read file with points
    while ( std::getline(inFile, curLine) ) {

        split(curLine, splitLine, ',');

        if (firstLine){
            //Get Dimensions
            x_dimensions = std::stoi(splitLine[0]);
            y_dimensions = std::stoi(splitLine[1]);
            firstLine = false;

        } else {
            std::get<0>(tmp) = splitLine[0];
            std::get<1>(tmp) = splitLine[1];
            double distToEnd = std::stod(splitLine[2]);
            double windSpeed = std::stod(splitLine[3]);
            double windDegree = std::stod(splitLine[4]);
            double currentVelocity = std::stod(splitLine[5]);
            double currentAngle = std::stod(splitLine[6]);


            // std::cout << "I: " << i << ", J: " << j << std::endl;

            addVertex( graph, tmp, i, j, y_dimensions, distToEnd, windSpeed, windDegree, currentVelocity, currentAngle);

            // std::cout << (double)std::get<0>(tmp) << ", " << (double)std::get<1>(tmp) << std::endl;

            // vec[i][j] = tmp;

            // Vertex v;
            // v.coordinates = tmp;
            // v.id = (j*dimensions) + i;

            // g.vertexList[v.id] = v;

            //Add edges
            addEdges( graph, i, j, y_dimensions );

            j++;
            if ( j % y_dimensions == 0 ) {
                j = 0;
                i++;
            }

        }

    }

    // std::cout << "Neighbors of 50" << std::endl;

    // for ( auto v : neighbors( 50, graph ) ) {
    //     std::cout << v.id << std::endl;
    // }
    // std::cout << g.vertexList[100].edgeList.size() << std::endl;



    auto res = dijkstra(graph, graph.getVertex(0) , graph.getVertex(graph.size() - 1), maxTime, model);

    auto costs = std::get<0>(res);

    auto path = std::get<1>(res);

    auto speeds = std::get<2>(res);

    auto times = std::get<3>(res);

    long double endTime = times[graph.getVertex(graph.size() - 1).id];


    Vertex end = graph.getVertex(graph.size() - 1);

    long double timeTaken = times[end.id];
 
    double cost = -1;


    //Time is larger than max time
    if (timeTaken > maxTime){

        auto ISRes = increaseSpeeds(end.id, maxTime, graph, path, speeds, times, costs, model);

        timeTaken = std::get<0>(ISRes);

        cost = std::get<1>(ISRes);

    }

    if (cost == -1){
        cost = costs[end.id];
    }




    std::ofstream outFile;
    outFile.open("path.csv");



    while (end.id != 0){
        std::cout << "Latitude: " << std::get<0>(end.coordinates) << ", Longitude: " << std::get<1>(end.coordinates) << ", Speed: " << speeds[end.id] << "\n";
        // std::cout << end.cost << std::endl;
        outFile << std::get<0>(end.coordinates) << "," << std::get<1>(end.coordinates) << ","<< end.id << "," << speeds[end.id] << "\n";

        

        end = path[end.id];
    }

    std::cout << "Latitude: " << std::get<0>(end.coordinates) << ", Longitude: " << std::get<1>(end.coordinates) << ", Speed: " << speeds[end.id] << "\n";
    outFile << std::get<0>(end.coordinates) << "," << std::get<1>(end.coordinates) << "," << end.id << "," << speeds[end.id] << "\n";
    outFile.close();


    inFile.close();


    

    return 0;
}
