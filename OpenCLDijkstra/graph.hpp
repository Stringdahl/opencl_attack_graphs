//
//  graph.hpp
//  OpenCLDijkstra
//
//  Created by Pontus Johnson on 2016-09-07.
//  Copyright © 2016 Pontus Johnson. All rights reserved.
//

#ifndef graph_hpp
#define graph_hpp

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <limits.h>
#include <float.h>


///
//  Types
//
//
//  This data structure and algorithm implementation is based on
//  Accelerating large graph algorithms on the GPU using CUDA by
//  Parwan Harish and P.J. Narayanan
//
typedef struct
{
    // This contains a pointer to the edge list for each vertex
    int *vertexArray;

    int *inverseVertexArray;

    // maxVerticeArray[i] is greater than 0 if vertex i is max, and -1 if min. It contains the highest value so far.
    int *maxVertexArray;
    
    // Vertex count
    int vertexCount;
    
    // This contains a pointer to the source array
    int *sourceArray;
    
    // Graph count
    int graphCount;
    
    // This contains pointers to the vertices that each edge is attached to
    int *edgeArray;

    int *inverseEdgeArray;

    // Edge count
    int edgeCount;
    
    // Cost array
    float *costArray;
    
    // Number of parents to each vertex
    int *parentCountArray;
    
    // Weight array
    float *weightArray;
    
    float *inverseWeightArray;
    
} GraphData;

void checkErrorFileLine(int errNum, int expected, const char* file, const int lineNumber);
void generateRandomGraph(GraphData *graph, int numVertices, int neighborsPerVertex, int numGraphs, float probOfMax);
float* dijkstra(GraphData *graph);

#endif /* graph_hpp */
