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
    // (V) This contains a pointer to the edge list for each vertex
    int *vertexArray;
    
    // Vertex count
    int vertexCount;
    
    // (V) This contains a pointer to the source array
    int *sourceArray;
    
    // Graph count
    int graphCount;
    
    // (E) This contains pointers to the vertices that each edge is attached to
    int *edgeArray;
    
    // Edge count
    int edgeCount;
    
    // (W) Weight array
    float *weightArray;
    
} GraphData;

void generateRandomGraph(GraphData *graph, int numVertices, int neighborsPerVertex, int numGraphs);

#endif /* graph_hpp */
