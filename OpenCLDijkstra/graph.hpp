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
    
    // maxVerticeArray[i] is 1 if vertex i is max, and 0 if min.
    // It is int rather than bool because some GPUs don't accept bool, I read somewhere.
    int *maxVertexArray;
    
    // Vertex count
    int vertexCount;
    
    // This contains a pointer to the source array
    int *sourceArray;
    
    // Graph count
    int graphCount;
    
    // This contains pointers to the vertices that each edge is attached to
    int *edgeArray;
    
    // Edge count
    int edgeCount;
    
    // Cost array
    float *costArray;
    
    // Number of parents to each vertex
    int *parentCountArray;
    
    // Weight array
    float *weightArray;
    
} GraphData;

void checkErrorFileLine(int errNum, int expected, const char* file, const int lineNumber);
void generateRandomGraph(GraphData *graph, int numVertices, int neighborsPerVertex, int numGraphs);

#endif /* graph_hpp */
