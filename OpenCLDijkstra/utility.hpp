//
//  utility.hpp
//  OpenCLDijkstra
//
//  Created by Pontus Johnson on 2016-09-07.
//  Copyright © 2016 Pontus Johnson. All rights reserved.
//

#ifndef utility_hpp
#define utility_hpp
#include <OpenCL/opencl.h>
#include <stdio.h>
#include "graph.hpp"

void checkErrorFileLine(int errNum, int expected, const char* file, const int lineNumber);

void printCostOfRandomVertices(float *costArrayHost, int verticesToPrint, int totalVerticeCount);
void printWeights(GraphData *graph);
void printGraph(GraphData *graph);
void printParents(GraphData *graph);
void printSources(GraphData *graph);
void printMaskArray(int *maskArrayHost, int totalVertexCount);
void printCostUpdating(GraphData *graph, cl_command_queue *commandQueue, cl_mem *maskArrayDevice, cl_mem *costArrayDevice, cl_mem *updatingCostArrayDevice, cl_mem *weightArrayDevice, cl_mem *parentCountArrayDevice);
void printMathematicaString(GraphData *graph, int iGraph);
void printTraversedEdges(cl_command_queue *commandQueue, GraphData *graph, cl_mem *traversedEdgeCountArrayDevice);
void printVisitedParents(cl_command_queue *commandQueue, GraphData *graph, cl_mem *parentCountArrayDevice);



#endif /* utility_hpp */
