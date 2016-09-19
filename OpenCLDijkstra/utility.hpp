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
#include <math.h>

void checkErrorFileLine(int errNum, int expected, const char* file, const int lineNumber);

void printCostOfRandomVertices(int *costArrayHost, int verticesToPrint, int totalVerticeCount);
void printWeights(GraphData *graph);
void printInverseWeights(GraphData *graph);
void printGraph(GraphData *graph);
void printInverseGraph(GraphData *graph);
void printParents(GraphData *graph);
void printSources(GraphData *graph);
void printMaskArray(int *maskArrayHost, int totalVertexCount);
void dumpBuffers(GraphData *graph, cl_command_queue *commandQueue, cl_mem *maskArrayDevice, cl_mem *costArrayDevice, cl_mem *updatingCostArrayDevice, cl_mem *weightArrayDevice, cl_mem *parentCountArrayDevice, cl_mem *maxVerticeArrayDevice,int iVertex);
void printAfterUpdating(GraphData *graph, cl_command_queue *commandQueue, int *maskArrayHost, cl_mem *costArrayDevice, cl_mem *updatingCostArrayDevice, cl_mem *weightArrayDevice, cl_mem *parentCountArrayDevice, cl_mem *maxVerticeArrayDevice);
void printMathematicaString(GraphData *graph, int iGraph);
void printTraversedEdges(cl_command_queue *commandQueue, GraphData *graph, cl_mem *traversedEdgeCountArrayDevice);
void printVisitedParents(cl_command_queue *commandQueue, GraphData *graph, cl_mem *parentCountArrayDevice);
void printMaxVertices(cl_command_queue *commandQueue, GraphData *graph, cl_mem *maxVertexArrayDevice);
void printSolution(int *dist, int n);
int getEdgeEnd(int iVertex, int vertexCount, int *vertexArray, int edgeCount);
void compareToCPUComputation(GraphData *graph, bool verbose, int nGraphsToCheck);
void shadowKernel1(int graphCount, int vertexCount, int edgeCount, cl_mem *vertexArrayDevice, cl_mem *inverseVertexArrayDevice, cl_mem *edgeArrayDevice, cl_mem *inverseEdgeArrayDevice, cl_mem *weightArrayDevice, cl_mem *inverseWeightArrayDevice, cl_command_queue *commandQueue, cl_mem *maskArrayDevice, cl_mem *costArrayDevice, cl_mem *updatingCostArrayDevice, cl_mem *parentCountArrayDevice, cl_mem *maxVerticeArrayDevice, cl_mem *traversedEdgeCountArrayDevice, cl_mem *intUpdateCostArrayDevice);


#endif /* utility_hpp */
