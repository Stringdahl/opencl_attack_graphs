//
//  utility.cpp
//  OpenCLDijkstra
//
//  Created by Pontus Johnson on 2016-09-07.
//  Copyright © 2016 Pontus Johnson. All rights reserved.
//

#include "utility.hpp"
#include "graph.hpp"

///
//  Macros
//
#define checkError(a, b) checkErrorFileLine(a, b, __FILE__ , __LINE__)


///
//  Utility functions adapted from NVIDIA GPU Computing SDK
//
void checkErrorFileLine(int errNum, int expected, const char* file, const int lineNumber);


void printCostOfRandomVertices(float *costArrayHost, int verticesToPrint, int totalVerticeCount) {
    for(int i = 0; i < verticesToPrint; i++)
    {
        int iVertice = rand() % totalVerticeCount;
        printf("Cost of node %i is %f\n", iVertice, costArrayHost[iVertice]);
    }
}

void printGraph(GraphData *graph) {
    int nChildren;
    for (int iNode=0; iNode<graph->vertexCount; iNode++) {
        if (iNode<graph->vertexCount-1) {
            nChildren = graph->vertexArray[iNode+1]-graph->vertexArray[iNode];
        }
        else {
            nChildren = graph->edgeCount-graph->vertexArray[iNode];
        }
        printf("Vertex %i has %i children\n", iNode, nChildren);
        for (int iChild=0; iChild<nChildren; iChild++) {
            printf("Vertex %i is parent to vertex %i with edge weight of %f\n", iNode, graph->edgeArray[graph->vertexArray[iNode]+iChild], graph->weightArray[graph->vertexArray[iNode]+iChild]);
        }
    }
}

void printParents(GraphData *graph) {
    for (int i = 0; i < graph->vertexCount; i++) {
        printf("Vertex %i has %i parents.\n", i, graph->parentCountArray[i]);
    }
}

void printWeights(GraphData *graph) {
    for (int i = 0; i < graph->graphCount*graph->edgeCount; i++) {
        printf("weightArray[%i] = %f\n", i, graph->weightArray[i]);
    }
}

void printSources(GraphData *graph) {
    printf("Nodes ");
    for (int i = 0; i < graph->graphCount; i++) {
        printf("%i, ", graph->sourceArray[i]);
    }
    printf("are sources.\n");
    
}

void printMaskArray(int *maskArrayHost, int totalVertexCount) {
    for (int i = 0; i < totalVertexCount; i++) {
        printf("%i", maskArrayHost[i]);
    }
    printf("\n");
}


void printCostUpdating(GraphData *graph, cl_command_queue *commandQueue, cl_mem *maskArrayDevice, cl_mem *costArrayDevice, cl_mem *updatingCostArrayDevice, cl_mem *weightArrayDevice) {
    
    int errNum = 0;
    cl_event readDone;
    
    int totalVertexCount = graph->graphCount * graph->vertexCount;
    int totalEdgeCount = graph->graphCount * graph->edgeCount;
    
    float *costArrayHost = (float*) malloc(sizeof(float) * totalVertexCount);
    float *updatingCostArrayHost = (float*) malloc(sizeof(float) * totalVertexCount);
    float *weightArrayHost = (float*) malloc(sizeof(float) * totalEdgeCount);
    int *maskArrayHost = (int*) malloc(sizeof(int) * totalVertexCount);
    
    errNum = clEnqueueReadBuffer(*commandQueue, *maskArrayDevice, CL_FALSE, 0, sizeof(int) * totalVertexCount, maskArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    
    errNum = clEnqueueReadBuffer(*commandQueue, *costArrayDevice, CL_FALSE, 0, sizeof(float) * totalVertexCount, costArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *updatingCostArrayDevice, CL_FALSE, 0, sizeof(float) * totalVertexCount, updatingCostArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *weightArrayDevice, CL_FALSE, 0, sizeof(float) * totalEdgeCount, weightArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    clWaitForEvents(1, &readDone);
    
    for (int i = 0; i < totalVertexCount; i++) {
        if ( maskArrayHost[i] != 0 )
        {
            
            int iGraph = i / graph->vertexCount;
            int localTid = i % graph->vertexCount;
            
            int edgeStart = graph->vertexArray[localTid];
            int edgeEnd;
            if (localTid + 1 < (graph->vertexCount))
            {
                edgeEnd = graph->vertexArray[localTid + 1];
            }
            else
            {
                edgeEnd = graph->edgeCount;
            }
            
            for(int edge = edgeStart; edge < edgeEnd; edge++)
            {
                int nid = iGraph*graph->vertexCount + graph->edgeArray[edge];
                int eid = iGraph*graph->edgeCount + edge;
                printf("Node %i (of cost %f) updated node %i (of cost %f) by edge %i with weight %f\n", i, costArrayHost[i], nid, costArrayHost[nid], edge, graph->weightArray[eid]);
            }
        }
    }
}

const char* costToString(float cost) {
    static char str[8];
    if (cost > 99999) {
        sprintf(str, "inf");
    }
    else {
        sprintf(str, "%.2f", cost);
    }
    return str;
}

bool contains(int *array, int arrayLength, int value) {
    for (int i = 0; i < arrayLength; i++) {
        if (array[i]==value) {
            return true;
        }
    }
    return false;
}

void printMathematicaString(GraphData *graph, int iGraph) {
    char str[64*graph->edgeCount];
    
    sprintf(str, "Graph[{");
    for (int localSource = 0; localSource < graph->vertexCount; localSource++) {
        int edgeStart = graph->vertexArray[localSource];
        int edgeEnd;
        if (localSource + 1 < (graph->vertexCount)) {
            edgeEnd = graph->vertexArray[localSource + 1];
        }
        else {
            edgeEnd = graph->edgeCount;
        }
        
        for(int edge = edgeStart; edge < edgeEnd; edge++) {
            int localTarget = graph->edgeArray[edge];
            sprintf(str + strlen(str), "%i \\[DirectedEdge] %i, ", localSource, localTarget);
        }
    }
    sprintf(str + strlen(str)-2, "}, VertexLabels -> {");
    for (int vertex = 0; vertex < graph->vertexCount; vertex++) {
        const char* sourceString = costToString(graph->costArray[vertex]);
        sprintf(str + strlen(str), "%i -> %s, ", vertex, sourceString);
    }
    sprintf(str + strlen(str)-2, "}, VertexShapeFunction -> {");
    for (int vertex = 0; vertex < graph->vertexCount; vertex++) {
        if (contains(graph->sourceArray, graph->graphCount, vertex)) {
            sprintf(str + strlen(str), "%i -> \"Star\", ", vertex);
        }
        else {
            if (graph->maxVertexArray[vertex]==1) {
                sprintf(str + strlen(str), "%i -> \"Square\", ", vertex);
            }
        }
    }
    sprintf(str + strlen(str) - 2, "}, VertexSize -> Large, EdgeShapeFunction -> GraphElementData[{\"CarvedArrow\", \"ArrowSize\" -> .02}]]\n");
    printf("%s", str);
}

void printTraversedEdges(cl_command_queue *commandQueue, GraphData *graph, cl_mem *traversedEdgeCountArrayDevice) {
    int *traversedEdgeCountArrayHost = (int*) malloc(sizeof(int) * graph->graphCount*graph->edgeCount);
    cl_event readDone;
    
    int errNum = clEnqueueReadBuffer(*commandQueue, *traversedEdgeCountArrayDevice, CL_FALSE, 0, sizeof(int) * graph->graphCount*graph->edgeCount, traversedEdgeCountArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    clWaitForEvents(1, &readDone);
    
    for (int iGraph=0; iGraph < graph->graphCount; iGraph++) {
        for (int iEdge=0; iEdge<graph->edgeCount; iEdge++) {
            printf("%i",traversedEdgeCountArrayHost[iGraph * graph->edgeCount + iEdge]);
        }
        printf("\n");
    }
}

