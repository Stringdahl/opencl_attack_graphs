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

void printInverseGraph(GraphData *graph) {
    int nParents;
    for (int iNode=0; iNode<graph->vertexCount; iNode++) {
        if (iNode<graph->vertexCount-1) {
            nParents = graph->inverseVertexArray[iNode+1]-graph->inverseVertexArray[iNode];
        }
        else {
            nParents = graph->edgeCount-graph->inverseVertexArray[iNode];
        }
        printf("Vertex %i has %i parents\n", iNode, nParents);
        for (int iParent=0; iParent<nParents; iParent++) {
            printf("Vertex %i is child to vertex %i with edge weight of %f\n", iNode, graph->inverseEdgeArray[graph->inverseVertexArray[iNode]+iParent], graph->inverseWeightArray[graph->inverseVertexArray[iNode]+iParent]);
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
        printf("weightArray[%i] = %.3f\n", i, graph->weightArray[i]);
    }
}

void printInverseWeights(GraphData *graph) {
    for (int i = 0; i < graph->graphCount*graph->edgeCount; i++) {
        printf("inverseWeightArray[%i] = %.3f\n", i, graph->inverseWeightArray[i]);
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

void dumpBuffers(GraphData *graph, cl_command_queue *commandQueue, cl_mem *maskArrayDevice, cl_mem *costArrayDevice, cl_mem *updatingCostArrayDevice, cl_mem *weightArrayDevice, cl_mem *parentCountArrayDevice, cl_mem *maxVerticeArrayDevice, int iVertex) {
    
    int errNum = 0;
    cl_event readDone;
    
    int totalVertexCount = graph->graphCount * graph->vertexCount;
    int totalEdgeCount = graph->graphCount * graph->edgeCount;
    
    float *costArrayHost = (float*) malloc(sizeof(float) * totalVertexCount);
    float *updatingCostArrayHost = (float*) malloc(sizeof(float) * totalVertexCount);
    float *weightArrayHost = (float*) malloc(sizeof(float) * totalEdgeCount);
    float *maxVertexArrayHost = (float*) malloc(sizeof(float) * totalVertexCount);
    int *parentCountArrayHost = (int*) malloc(sizeof(int) * totalVertexCount);
    int *maskArrayHost = (int*) malloc(sizeof(int) * totalVertexCount);
    
    
    errNum = clEnqueueReadBuffer(*commandQueue, *costArrayDevice, CL_FALSE, 0, sizeof(float) * totalVertexCount, costArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *maxVerticeArrayDevice, CL_FALSE, 0, sizeof(float) * totalVertexCount, maxVertexArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *weightArrayDevice, CL_FALSE, 0, sizeof(float) * totalEdgeCount, weightArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *parentCountArrayDevice, CL_FALSE, 0, sizeof(int) * graph->graphCount*graph->vertexCount, parentCountArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *maskArrayDevice, CL_FALSE, 0, sizeof(int) * graph->graphCount*graph->vertexCount, maskArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *updatingCostArrayDevice, CL_FALSE, 0, sizeof(float) * totalVertexCount, updatingCostArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    clWaitForEvents(1, &readDone);
    
    for (int tid = 0; tid < totalVertexCount; tid++) {
        int localTid = tid % graph -> vertexCount;
        int iGraph = tid / graph -> vertexCount;
        
        if (tid == iVertex || iVertex == -1) {
            printf("Node %i: Mask: %i, Cost: %.2f, updatingCost: %.2f, max: %.2f, parentCount: %i.\n", tid, maskArrayHost[tid], costArrayHost[tid], updatingCostArrayHost[tid], maxVertexArrayHost[tid], parentCountArrayHost[tid]);
            
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
                int nid = graph->edgeArray[edge];
                printf("Influences node %i through edge %i by weight %.2f.\n", (iGraph*graph->vertexCount + nid), edge, graph->weightArray[iGraph*graph->vertexCount + edge]);
            }
        }
    }
}

void printAfterUpdating(GraphData *graph, cl_command_queue *commandQueue, int *maskArrayHost, cl_mem *costArrayDevice, cl_mem *updatingCostArrayDevice, cl_mem *weightArrayDevice, cl_mem *parentCountArrayDevice, cl_mem *maxVerticeArrayDevice) {
    
    int errNum = 0;
    cl_event readDone;
    
    int totalVertexCount = graph->graphCount * graph->vertexCount;
    int totalEdgeCount = graph->graphCount * graph->edgeCount;
    
    float *costArrayHost = (float*) malloc(sizeof(float) * totalVertexCount);
    float *updatingCostArrayHost = (float*) malloc(sizeof(float) * totalVertexCount);
    float *weightArrayHost = (float*) malloc(sizeof(float) * totalEdgeCount);
    float *maxVertexArrayHost = (float*) malloc(sizeof(float) * totalVertexCount);
    int *parentCountArrayHost = (int*) malloc(sizeof(int) * graph->graphCount*graph->vertexCount);
    
    
    errNum = clEnqueueReadBuffer(*commandQueue, *costArrayDevice, CL_FALSE, 0, sizeof(float) * totalVertexCount, costArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *updatingCostArrayDevice, CL_FALSE, 0, sizeof(float) * totalVertexCount, updatingCostArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *maxVerticeArrayDevice, CL_FALSE, 0, sizeof(float) * totalVertexCount, maxVertexArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *weightArrayDevice, CL_FALSE, 0, sizeof(float) * totalEdgeCount, weightArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *parentCountArrayDevice, CL_FALSE, 0, sizeof(int) * graph->graphCount*graph->vertexCount, parentCountArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    clWaitForEvents(1, &readDone);
    
    printf("%i vertices.\n", totalVertexCount);
    for (int tid = 0; tid < totalVertexCount; tid++) {
        if ( maskArrayHost[tid] != 0 ) {
            printf("Vertex %i, (max: %.2f, %i remaining parents) is considered for updating.\n", tid, maxVertexArrayHost[tid], parentCountArrayHost[tid]);
            if (maxVertexArrayHost[tid]<0 || parentCountArrayHost[tid]==0) {
                
                int iGraph = tid / graph->vertexCount;
                int localTid = tid % graph->vertexCount;
                
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
                    
                    printf("Node %i (of cost %.2f and updatingCost %.2f) updated node %i (of max %.2f with %i remaining parents) by edge %i with weight %.2f. Node %i now has cost %.2f and updatingCost %.2f.\n", tid, costArrayHost[tid], updatingCostArrayHost[tid], nid, maxVertexArrayHost[nid], parentCountArrayHost[nid], edge, graph->weightArray[eid], nid, costArrayHost[nid], updatingCostArrayHost[nid]);
                    
                    if (maxVertexArrayHost[nid]>=0)
                    {
                        printf("Updated a max node.\n");
                        printf("Perhaps maxVertexArray was increased. It is now %.2f.\n", maxVertexArrayHost[nid]);
                        if (parentCountArrayHost[nid]==0) {
                            printf("All parents visited. Set updatingCostArray[nid] (%.2f) to maxVertexArray[nid] (%.2f).\n", updatingCostArrayHost[nid], maxVertexArrayHost[nid]);
                        }
                    }
                }
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
        int globalSource = iGraph*graph->vertexCount + localSource;
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
            int globalTarget = iGraph*graph->vertexCount + localTarget;
            sprintf(str + strlen(str), "%i \\[DirectedEdge] %i, ", globalSource, globalTarget);
        }
    }
    sprintf(str + strlen(str)-2, "}, VertexLabels -> {");
    for (int vertex = 0; vertex < graph->vertexCount; vertex++) {
        int globalVertex = iGraph*graph->vertexCount + vertex;
        const char* sourceString = costToString(graph->costArray[globalVertex]);
        sprintf(str + strlen(str), "%i -> %i [%s], ", globalVertex, globalVertex, sourceString);
    }
    sprintf(str + strlen(str)-2, "}, VertexShapeFunction -> {");
    for (int localVertex = 0; localVertex < graph->vertexCount; localVertex++) {
        int globalVertex = iGraph*graph->vertexCount + localVertex;
        if (graph->sourceArray[iGraph]==localVertex) {
            sprintf(str + strlen(str), "%i -> \"Star\", ", globalVertex);
        }
        else {
            if (graph->maxVertexArray[localVertex]>=0) {
                sprintf(str + strlen(str), "%i -> \"Square\", ", globalVertex);
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

void printVisitedParents(cl_command_queue *commandQueue, GraphData *graph, cl_mem *parentCountArrayDevice) {
    int *parentCountArrayHost = (int*) malloc(sizeof(int) * graph->graphCount*graph->vertexCount);
    cl_event readDone;
    
    int errNum = clEnqueueReadBuffer(*commandQueue, *parentCountArrayDevice, CL_FALSE, 0, sizeof(int) * graph->graphCount*graph->vertexCount, parentCountArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    clWaitForEvents(1, &readDone);
    
    for (int iGraph=0; iGraph < graph->graphCount; iGraph++) {
        for (int iVertex=0; iVertex<graph->vertexCount; iVertex++) {
            printf("Vertex %i has %i remaining parents\n",iGraph * graph->vertexCount + iVertex, parentCountArrayHost[iGraph * graph->vertexCount + iVertex]);
        }
    }
}

void printMaxVertices(cl_command_queue *commandQueue, GraphData *graph, cl_mem *maxVertexArrayDevice) {
    float *maxVertexArrayHost = (float*) malloc(sizeof(float) * graph->graphCount*graph->vertexCount);
    cl_event readDone;
    
    int errNum = clEnqueueReadBuffer(*commandQueue, *maxVertexArrayDevice, CL_FALSE, 0, sizeof(int) * graph->graphCount*graph->vertexCount, maxVertexArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    clWaitForEvents(1, &readDone);
    
    for (int iGraph=0; iGraph < graph->graphCount; iGraph++) {
        for (int iVertex=0; iVertex<graph->vertexCount; iVertex++) {
            int iGlobalVertex =iGraph * graph->vertexCount + iVertex;
            printf("Max of vertex %i is %.2f.\n",iGlobalVertex, maxVertexArrayHost[iGlobalVertex]);
        }
    }
}

// A utility function to print the constructed distance array
void printSolution(float *dist, int n)
{
    printf("Vertex   Distance from Source\n");
    for (int i = 0; i < n; i++)
        printf("%d \t\t %.2f\n", i, dist[i]);
}

void compareToCPUComputation(GraphData *graph, bool verbose, int nGraphsToCheck) {
    if (nGraphsToCheck>graph->graphCount) {
        nGraphsToCheck=graph->graphCount;
    }
    printf("Checking correctness against sequential implementation.\n");
    int nInfinite = 0;
    int iErrors = 0;
    for (int iGraph = 0; iGraph<nGraphsToCheck; iGraph++) {
        float *dist = dijkstra(graph, iGraph, verbose);
        if (verbose) {
            printf("Source is %i.\n", graph->sourceArray[iGraph]);
        }
        for (int iVertex = 0; iVertex < graph->vertexCount; iVertex++) {
            if (verbose) {
                printf("%i: CPU=%.2f, GPU=%.2f\n", iVertex, dist[iVertex], graph->costArray[iGraph*graph->vertexCount + iVertex]);
            }
            if (dist[iVertex]-graph->costArray[iGraph*graph->vertexCount + iVertex]>0.01 || graph->costArray[iGraph*graph->vertexCount + iVertex]-dist[iVertex]>0.01) {
                printf("CPU computed %.2f for vertex %i while GPU computed %.2f\n", dist[iVertex], iVertex, graph->costArray[iGraph*graph->vertexCount + iVertex]);
                iErrors++;
               // exit(1);
            }
            if (graph->costArray[iGraph*graph->vertexCount + iVertex] == FLT_MAX) {
                nInfinite++;
            }
        }
    }
    printf("%i errors.\n", iErrors);
    printf("%i infinite-time attack steps.\n", nInfinite);
}

#define PRECISION 1000


int getMilliInteger(float floatValue) {
    if (floatValue*PRECISION < INT_MAX)
        return (int)((floatValue*PRECISION)+0.5);
    else
        return INT_MAX;
    
}

int getEdgeEnd(int iVertex, int vertexCount, int *vertexArray, int edgeCount) {
    if (iVertex + 1 < (vertexCount))
        return vertexArray[iVertex + 1];
    else
        return edgeCount;
}


void shadowKernel1(int graphCount, int vertexCount, int edgeCount, cl_mem *vertexArrayDevice, cl_mem *inverseVertexArrayDevice, cl_mem *edgeArrayDevice, cl_mem *inverseEdgeArrayDevice, cl_mem *weightArrayDevice, cl_mem *inverseWeightArrayDevice, cl_command_queue *commandQueue, cl_mem *maskArrayDevice, cl_mem *costArrayDevice, cl_mem *updatingCostArrayDevice, cl_mem *parentCountArrayDevice, cl_mem *maxVerticeArrayDevice, cl_mem *traversedEdgeCountArrayDevice, cl_mem *intUpdateCostArrayDevice) {
    
    int errNum = 0;
    cl_event readDone;
    
    int totalVertexCount = graphCount*vertexCount;
    int totalEdgeCount = graphCount*edgeCount;
    
    int *vertexArray = (int*) malloc(sizeof(int) * totalVertexCount);
    int *inverseVertexArray = (int*) malloc(sizeof(int) * totalVertexCount);
    int *edgeArray = (int*) malloc(sizeof(int) * totalEdgeCount);
    float *weightArray = (float*) malloc(sizeof(float) * totalEdgeCount);
    float *inverseWeightArray = (float*) malloc(sizeof(float) * totalEdgeCount);
    int *inverseEdgeArray = (int*) malloc(sizeof(int) * totalEdgeCount);
    float *costArray = (float*) malloc(sizeof(float) * totalVertexCount);
    float *updatingCostArray = (float*) malloc(sizeof(float) * totalVertexCount);
    float *maxVertexArray = (float*) malloc(sizeof(float) * totalVertexCount);
    int *parentCountArray = (int*) malloc(sizeof(int) * totalVertexCount);
    int *maskArray = (int*) malloc(sizeof(int) * totalVertexCount);
    int *traversedEdgeCountArray = (int*) malloc(sizeof(int) * totalEdgeCount);
    int *intUpdateCostArray = (int*) malloc(sizeof(int) * totalVertexCount);
    
    
    errNum = clEnqueueReadBuffer(*commandQueue, *vertexArrayDevice, CL_FALSE, 0, sizeof(int) * vertexCount, vertexArray, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *inverseVertexArrayDevice, CL_FALSE, 0, sizeof(int) * vertexCount, inverseVertexArray, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *edgeArrayDevice, CL_FALSE, 0, sizeof(int) * edgeCount, edgeArray, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *inverseEdgeArrayDevice, CL_FALSE, 0, sizeof(int) * edgeCount, inverseEdgeArray, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *weightArrayDevice, CL_FALSE, 0, sizeof(float) * totalEdgeCount, weightArray, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *inverseWeightArrayDevice, CL_FALSE, 0, sizeof(float) * totalEdgeCount, inverseWeightArray, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *costArrayDevice, CL_FALSE, 0, sizeof(float) * totalVertexCount, costArray, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *updatingCostArrayDevice, CL_FALSE, 0, sizeof(float) * totalVertexCount, updatingCostArray, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *maxVerticeArrayDevice, CL_FALSE, 0, sizeof(float) * totalVertexCount, maxVertexArray, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *parentCountArrayDevice, CL_FALSE, 0, sizeof(int) * totalVertexCount, parentCountArray, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *maskArrayDevice, CL_FALSE, 0, sizeof(int) * totalVertexCount, maskArray, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *traversedEdgeCountArrayDevice, CL_FALSE, 0, sizeof(int) * totalEdgeCount, traversedEdgeCountArray, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *intUpdateCostArrayDevice, CL_FALSE, 0, sizeof(int) * totalVertexCount, intUpdateCostArray, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    clWaitForEvents(1, &readDone);
    
    printf("\n\n\nIn the shadow\n\n");
    for (int globalSource = 0; globalSource < totalVertexCount; globalSource++) {
        
        int iGraph = globalSource / vertexCount;
        int localSource = globalSource % vertexCount;
        
        printf("globalSource = %i, iGraph = %i, localSource = %i, maskArray[%i] = %i.\n", globalSource, iGraph, localSource, globalSource, maskArray[globalSource]);
        // Only consider vertices that are marked for update
        if ( maskArray[globalSource] != 0 ) {
            // After attempting to update, don't do it again unless (i) a parent updated this, or (ii) recalculation is required due to kernel 2.
            maskArray[globalSource] = 0;
            // Only update if (i) this is a min node, or (ii) this is a max node and all parents have been visited.
            printf("maxVertexArray[%i] = %.2f, parentCountArray[%i] = %i\n", globalSource, maxVertexArray[globalSource], globalSource, parentCountArray[globalSource]);
            if (maxVertexArray[globalSource]<0 || parentCountArray[globalSource]==0) {
                {
                    // Get the edges
                    int edgeStart = vertexArray[localSource];
                    int edgeEnd = getEdgeEnd(localSource, vertexCount, vertexArray, edgeCount);
                    
                    printf("edgeStart = %i, edgeEnd = %i.\n", edgeStart, edgeEnd);
                    // Iterate over the edges
                    for(int localEdge = edgeStart; localEdge < edgeEnd; localEdge++)
                    {
                        // nid is the (globally indexed) target node
                        int localTarget = edgeArray[localEdge];
                        int globalTarget = iGraph*vertexCount + edgeArray[localEdge];
                        // eid is the globally indexed edge
                        int globalEdge = iGraph*edgeCount + localEdge;
                        
                        printf("localEdge = %i, globalTarget = %i, globalEdge = %i, traversedEdgeCountArray[%i] = %i.\n", localEdge, globalTarget, globalEdge, globalEdge, traversedEdgeCountArray[globalEdge]);
                        // If this edge has never been traversed, reduce the remaining parents of the target by one, so that they reach zero when all incoming edges have been visited.
                        if (traversedEdgeCountArray[globalEdge] == 0) {
                            printf("Reducing parentCountArray[%i] by one from %i ", globalTarget, parentCountArray[globalTarget]);
                            parentCountArray[globalTarget]--;
                            printf(" to %i.\n", parentCountArray[globalTarget]);
                        }
                        // Mark that this edge has been traversed.
                        printf("Augmenting traversedEdgeCountArray[%i] by one from %i.\n", globalEdge, traversedEdgeCountArray[globalEdge]);
                        traversedEdgeCountArray[globalEdge] ++;
                        int inverseEdgeStart = inverseVertexArray[localTarget];
                        int inverseEdgeEnd = getEdgeEnd(localTarget, vertexCount, inverseVertexArray, edgeCount);
                        // If this is a min node ...
                        printf("updatingCostArray[%i] = %.2f, inverseEdgeStart = %i, inverseEdgeEnd = %i, maxVertexArray[%i] = %.2f.\n", globalTarget, updatingCostArray[globalTarget], inverseEdgeStart, inverseEdgeEnd, globalTarget, maxVertexArray[globalTarget]);
                        if (maxVertexArray[globalTarget]<0) {
                            // ...atomically choose the lesser of the current and candidate updatingCost
                            //atomic_min(&intUpdateCostArrayDevice[nid], candidateMilliCostInt);
                            // Reconvert the integer representation to float and store in updatingCostArray
                            //updatingCostArray[nid] = (float)(intUpdateCostArrayDevice[nid])/PRECISION;
                            // Iterate over the edges
                            float minEdgeVal = FLT_MAX;
                            printf("minEdgeVal = %.2f.\n", minEdgeVal);
                            for(int localInverseEdge = inverseEdgeStart; localInverseEdge < inverseEdgeEnd; localInverseEdge++) {
                                int localInverseTarget = inverseEdgeArray[localInverseEdge];
                                int globalInverseTarget = iGraph*vertexCount + localInverseTarget;
                                int globalInverseEdge = iGraph*edgeCount + localInverseEdge;
                                float currEdgeVal = costArray[globalInverseTarget] + inverseWeightArray[globalInverseEdge];
                                printf("inverseEdge = %i, currEdgeVal = %.2f, inverseEdgeArray[%i] = %i, costArray[%i] = %.2f, inverseWeightArray[%i] = %.2f.\n",localInverseEdge, currEdgeVal, localInverseEdge, inverseEdgeArray[localInverseEdge], globalInverseTarget, costArray[globalInverseTarget], localInverseEdge, inverseWeightArray[globalInverseEdge]);
                                if (currEdgeVal<minEdgeVal) {
                                    minEdgeVal = currEdgeVal;
                                    printf("Updated minEdgeVal to %.2f.\n", minEdgeVal);
                                }
                            }
                            updatingCostArray[globalTarget] = minEdgeVal;
                            printf("Updated updatingCostArray[%i] to %.2f.\n", globalTarget, updatingCostArray[globalTarget]);
                            // Mark the target for update
                            //maskArray[nid] = 1;
                            
                        }
                        
                        // If this is a max node...
                        else {
                            printf("In max node.\n");
                            if (parentCountArray[globalTarget]==0) {
                                // If all parents have been visited ...
                                // Iterate over the edges
                                float maxEdgeVal = 0;
                                for(int inverseEdge = inverseEdgeStart; inverseEdge < inverseEdgeEnd; inverseEdge++) {
                                    float currEdgeVal = costArray[inverseEdgeArray[inverseEdge]] + inverseWeightArray[inverseEdge];
                                    if (currEdgeVal>maxEdgeVal) {
                                        maxEdgeVal = currEdgeVal;
                                    }
                                }
                                
                                costArray[globalTarget] = maxEdgeVal;
                                updatingCostArray[globalTarget] = maxEdgeVal;
                                // Mark the target for update
                                maskArray[globalTarget] = 1;
                            }
                        }
                    }
                }
            }
        }
    }
}


