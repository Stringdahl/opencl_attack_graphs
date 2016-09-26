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
    // Graph count
    int graphCount;
    
    // Vertex count
    int vertexCount;
    
    // Edge count
    int edgeCount;
    
    // Source count
    int sourceCount;
    
    // This contains a pointer to the edge list for each vertex
    int *vertexArray;

    // maxVerticeArray[i] is greater than 0 if vertex i is max, and -1 if min. It contains the highest value so far.
    int *maxVertexArray;
    
    // This contains a pointer to the source array
    int *sourceArray;
    
    // This contains pointers to the vertices that each edge is attached to
    int *edgeArray;
    
    // Weight array
    int *weightArray;
    
    // Cost array
    int *costArray;
    
    // Sum cost array
    int *sumCostArray;
    
    // Number of parents to each vertex
    int *parentCountArray;
    
    int *inverseVertexArray;

    int *inverseEdgeArray;

    int *inverseWeightArray;
    
    int *shortestParentsArray;
    
} GraphData;

void checkErrorFileLine(int errNum, int expected, const char* file, const int lineNumber);
void generateRandomGraph(GraphData *graph, int vertexCount, int neighborsPerVertex, int graphCount, int sourceCount, float probOfMax);
void completeReadGraph(GraphData *graph);
void updateGraphWithNewRandomWeights(GraphData *graph);
int* dijkstra(GraphData *graph, int iGraph, bool verbose);

#endif /* graph_hpp */
