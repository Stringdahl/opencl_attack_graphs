//
//  graph.cpp
//  OpenCLDijkstra
//
//  Created by Pontus Johnson on 2016-09-07.
//  Copyright © 2016 Pontus Johnson. All rights reserved.
//

#include "graph.hpp"


///
//  Namespaces
//
using namespace std;



bool containes(int *array, int arrayLength, int value) {
    for (int i = 0; i < arrayLength; i++) {
        if (array[i]==value) {
            return true;
        }
    }
    return false;
}


///
/// Check for error condition and exit if found.  Print file and line number
/// of error. (from NVIDIA SDK)
///
void checkErrorFileLine(int errNum, int expected, const char* file, const int lineNumber)
{
    if (errNum != expected)
    {
        cerr << "Line " << lineNumber << " in File " << file << endl;
        exit(1);
    }
}


///
//  Generate a random graph
//
void generateRandomGraph(GraphData *graph, int vertexCount, int neighborsPerVertex, int graphCount, int sourceCount, float probOfMax)
{
    graph->vertexCount = vertexCount;
    graph->graphCount = graphCount;
    graph->sourceCount = sourceCount;
    graph->vertexArray = (int*) malloc(graph->vertexCount * sizeof(int));
    graph->inverseVertexArray = (int*) malloc(graph->vertexCount * sizeof(int));
    graph->maxVertexArray = (int*) malloc(graph->vertexCount * sizeof(int));
    graph->costArray = (int*) malloc(graph->graphCount * graph->vertexCount * sizeof(int));
    graph->sumCostArray = (int*) malloc(graph->graphCount * graph->vertexCount * sizeof(int));
    graph->sourceArray = (int*) malloc(graph->sourceCount * sizeof(int));
    graph->edgeCount = vertexCount * neighborsPerVertex;
    graph->edgeArray = (int*)malloc(graph->edgeCount * sizeof(int));
    graph->inverseEdgeArray = (int*)malloc(graph->edgeCount * sizeof(int));
    graph->parentCountArray = (int*)malloc(graph->edgeCount * sizeof(int));
    graph->weightArray = (int*)malloc(graphCount * graph->edgeCount * sizeof(int));
    graph->inverseWeightArray = (int*)malloc(graphCount * graph->edgeCount * sizeof(int));
    graph->shortestParentsArray = (int*)malloc(graphCount * graph->edgeCount * sizeof(int));
    
    
    
    for(int i = 0; i < graph->vertexCount; i++)
    {
        graph->vertexArray[i] = i * neighborsPerVertex;
        if ((rand() % 100) < 100*probOfMax) {
            graph->maxVertexArray[i]=0;
        }
        else {
            graph->maxVertexArray[i]=-1;
        }
        graph->parentCountArray[i] = 0;
    }
    
    for(int i = 0; i < graph->edgeCount; i++)
    {
        int targetVertex = (rand() % graph->vertexCount);
        graph->edgeArray[i] = targetVertex;
        graph->parentCountArray[targetVertex]++;
        
    }
    for(int i = 0; i < graphCount * graph->edgeCount; i++)
    {
        graph->weightArray[i] = (rand() % 1000);
    }
    
    for(int iSource = 0; iSource < graph->sourceCount; iSource++) {
        int tempSource = (rand() % graph->vertexCount);
        while (containes(graph->sourceArray, graph->sourceCount, tempSource)) {
            tempSource = (rand() % graph->vertexCount);
        }
        graph->sourceArray[iSource] = tempSource;
        // The source should be min
        graph->maxVertexArray[graph->sourceArray[iSource]]=-1;
    }
    
    int iEdge = 0;
    for (int iChild = 0; iChild < graph->vertexCount; iChild++) {
        graph->inverseVertexArray[iChild] = iEdge;
        for (int iParent = 0; iParent < graph->vertexCount; iParent++) {
            // Get the edges
            int edgeStart = graph->vertexArray[iParent];
            int edgeEnd;
            if (iParent + 1 < (graph->vertexCount))
            {
                edgeEnd = graph->vertexArray[iParent + 1];
            }
            else
            {
                edgeEnd = graph->edgeCount;
            }
            for(int edge = edgeStart; edge < edgeEnd; edge++){
                if (graph->edgeArray[edge]==iChild) {
                    graph->inverseEdgeArray[iEdge]=iParent;
                    for (int iGraph = 0; iGraph < graphCount; iGraph++) {
                        graph->inverseWeightArray[iGraph * graph->edgeCount + iEdge]=graph->weightArray[iGraph * graph->edgeCount + edge];
                    }
                    iEdge ++;
                }
            }
        }
    }
}

///
//  Compute all values that can be derived from a graph as read by readGraphFromFile.
//
void completeReadGraph(GraphData *graph)
{
    graph->inverseVertexArray = (int*) malloc(graph->vertexCount * sizeof(int));
    graph->costArray = (int*) malloc(graph->graphCount * graph->vertexCount * sizeof(int));
    graph->sumCostArray = (int*) malloc(graph->graphCount * graph->vertexCount * sizeof(int));
    graph->inverseEdgeArray = (int*)malloc(graph->edgeCount * sizeof(int));
    graph->parentCountArray = (int*)malloc(graph->edgeCount * sizeof(int));
    graph->inverseWeightArray = (int*)malloc(graph->graphCount * graph->edgeCount * sizeof(int));
    graph->shortestParentsArray = (int*)malloc(graph->graphCount * graph->edgeCount * sizeof(int));
    
    
    
    for(int i = 0; i < graph->vertexCount; i++)
    {
        graph->parentCountArray[i] = 0;
    }
    
    for(int i = 0; i < graph->edgeCount; i++)
    {
        graph->parentCountArray[graph->edgeArray[i]]++;
    }
    
    for(int iSource = 0; iSource < graph->sourceCount; iSource++) {
        // The source should be min
        graph->maxVertexArray[graph->sourceArray[iSource]]=-1;
    }
    
    int iEdge = 0;
    for (int iChild = 0; iChild < graph->vertexCount; iChild++) {
        graph->inverseVertexArray[iChild] = iEdge;
        for (int iParent = 0; iParent < graph->vertexCount; iParent++) {
            // Get the edges
            int edgeStart = graph->vertexArray[iParent];
            int edgeEnd;
            if (iParent + 1 < (graph->vertexCount))
            {
                edgeEnd = graph->vertexArray[iParent + 1];
            }
            else
            {
                edgeEnd = graph->edgeCount;
            }
            for(int edge = edgeStart; edge < edgeEnd; edge++){
                if (graph->edgeArray[edge]==iChild) {
                    graph->inverseEdgeArray[iEdge]=iParent;
                    for (int iGraph = 0; iGraph < graph->graphCount; iGraph++) {
                        graph->inverseWeightArray[iGraph * graph->edgeCount + iEdge]=graph->weightArray[iGraph * graph->edgeCount + edge];
                    }
                    iEdge ++;
                }
            }
        }
    }
}



void updateGraphWithNewRandomWeights(GraphData *graph) {
    for(int i = 0; i < graph->graphCount * graph->edgeCount; i++)
    {
        graph->weightArray[i] = (rand() % 1000);
    }
    int iEdge = 0;
    for (int iChild = 0; iChild < graph->vertexCount; iChild++) {
        for (int iParent = 0; iParent < graph->vertexCount; iParent++) {
            // Get the edges
            int edgeStart = graph->vertexArray[iParent];
            int edgeEnd;
            if (iParent + 1 < (graph->vertexCount))
            {
                edgeEnd = graph->vertexArray[iParent + 1];
            }
            else
            {
                edgeEnd = graph->edgeCount;
            }
            for(int edge = edgeStart; edge < edgeEnd; edge++){
                if (graph->edgeArray[edge]==iChild) {
                    for (int iGraph = 0; iGraph < graph->graphCount; iGraph++) {
                        graph->inverseWeightArray[iGraph * graph->edgeCount + iEdge]=graph->weightArray[iGraph * graph->edgeCount + edge];
                    }
                    iEdge ++;
                }
            }
        }
    }
}


void reorderEdgeAttributeArrayFromInverse(int *edgeAttributeArray, GraphData graph) {
    
}


// A utility function to find the vertex with minimum distance value, from
// the set of vertices not yet included in shortest path tree
int minDistance(int *dist, bool *sptSet, int vertexCount)
{
    // Initialize min value
    int min = INT_MAX;
    int min_index = 0;
    
    for (int v = 0; v < vertexCount; v++) {
        //        printf("Node %i costs %i ", v, dist[v]);
        //        if (sptSet[v] == false) {
        //            printf("and is unprocessed. ");
        //        }
        //        else {
        //            printf("but is already proceessed. ");
        //        }
        if (sptSet[v] == false && dist[v] <= min) {
            min = dist[v], min_index = v;
        }
        //        printf("Lowest so far is %i with %i\n", min_index, min);
    }
    return min_index;
}



bool atLeastOneUnprocessedIsFinite(bool *sptSet, int vertexCount, int *dist) {
    for (int i = 0; i < vertexCount; i++) {
        if (!sptSet[i] && dist[i]<INT_MAX) {
            return true;
        }
    }
    return false;
}


// Funtion that implements Dijkstra's single source shortest path algorithm
// for a graph represented using adjacency matrix representation
int* dijkstra(GraphData *graph, int iGraph, bool verbose){
    int *dist = (int*) malloc(sizeof(int) * graph->vertexCount);     // The output array.  dist[i] will hold the shortest
    // distance from src to i
    bool *sptSet = (bool*) malloc(sizeof(bool) * graph->vertexCount); // sptSet[i] will true if vertex i is included in shortest
    // path tree or shortest distance from src to i is finalized
    
    int vertexCount = graph->vertexCount;
    int edgeCount = graph->edgeCount;
    
    // Copy the appropriate graph from GraphData
    int *vertexArray = (int*)malloc(vertexCount * sizeof(int));
    int *parentCountArray = (int*)malloc(vertexCount * sizeof(int));
    int *maxVertexArray = (int*)malloc(vertexCount * sizeof(int));
    
    for (int iVertex=0; iVertex<vertexCount; iVertex++) {
        vertexArray[iVertex]=graph->vertexArray[iVertex];
        parentCountArray[iVertex]=graph->parentCountArray[iVertex];
        maxVertexArray[iVertex]=graph->maxVertexArray[iVertex];
    }
    
    int *edgeArray = (int*)malloc(edgeCount * sizeof(int));
    int *weightArray = (int*)malloc(edgeCount * sizeof(int));
    int *traversedEdgeCountArray = (int*)malloc(edgeCount * sizeof(int));
    
    for (int iEdge=0; iEdge<edgeCount; iEdge++) {
        edgeArray[iEdge] = graph->edgeArray[iEdge];
        weightArray[iEdge] = graph->weightArray[iGraph*edgeCount + iEdge];
        traversedEdgeCountArray[iEdge]=0;
    }
    
    
    // Initialize all distances as INFINITE and stpSet[] as false
    for (int i = 0; i < vertexCount; i++)
        dist[i] = INT_MAX, sptSet[i] = false;
    
    // Distance of  vertex from itself is always 0
        for (int iSource = 0; iSource < graph->sourceCount; iSource++) {
            dist[graph->sourceArray[iSource]] = 0;
            if (verbose) {
                printf("Source vertex = %i in CPU.\n", iGraph*graph->vertexCount + graph->sourceArray[iSource]);
            }
        }
    
    // Find shortest path for all vertices
    while (atLeastOneUnprocessedIsFinite(sptSet, vertexCount, dist))
    {
        
        // Pick the minimum distance vertex from the set of vertices not
        // yet processed. u is always equal to src in first iteration.
        int source = minDistance(dist, sptSet, vertexCount);
        if (verbose) {
            printf("Node %i (of cost %i) ...", source, dist[source]);
        }
        if (maxVertexArray[source]<0 || parentCountArray[source]==0) {
            // Mark the picked vertex as processed
            sptSet[source] = true;
            
            // Get the edges
            int edgeStart = vertexArray[source];
            int edgeEnd;
            if (source + 1 < (vertexCount))
            {
                edgeEnd = vertexArray[source + 1];
            }
            else
            {
                edgeEnd = edgeCount;
            }
            // Iterate over the edges
            
            for(int edge = edgeStart; edge < edgeEnd; edge++) {
                int target = edgeArray[edge];
                
                if (traversedEdgeCountArray[edge]==0) {
                    parentCountArray[target]--;
                }
                traversedEdgeCountArray[edge]++;
                
                if (dist[source] != INT_MAX) {
                    // If min node
                    if (maxVertexArray[target]<0) {
                        if (!sptSet[target]) {
                            if (verbose) {
                                printf(" looking at min node %i (with %i remainaing parents) by edge with weight %i.", target, parentCountArray[target], graph->weightArray[edge]);
                            }
                            long longDist = dist[source];
                            longDist = longDist + weightArray[edge];
                            if (longDist > INT_MAX)
                                longDist = INT_MAX;
                            
                            if (longDist < dist[target]) {
                                if (verbose) {
                                    printf(".. updated from %i ", dist[target]);
                                }
                                dist[target] = (int)longDist;
                                if (verbose) {
                                    printf("to %i", dist[target]);
                                }
                            }
                            if (verbose) {
                                printf("\n");
                            }
                        }
                    }
                    
                    // If max node
                    else {
                        if (verbose) {
                            printf(" looking at max node %i (with %i remainaing parents, max: %i) by edge with weight %i.", target, parentCountArray[target], maxVertexArray[target], graph->weightArray[edge]);
                        }
                        long longDist = dist[source];
                        longDist = longDist + weightArray[edge];
                        if (longDist > INT_MAX)
                            longDist = INT_MAX;
                        if (maxVertexArray[target] < longDist) {
                            maxVertexArray[target] = (int)longDist;
                        }
                        if (parentCountArray[target]==0) {
                            if (verbose) {
                                printf(".. updated from %i ", dist[target]);
                            }
                            dist[target] = maxVertexArray[target];
                            if (verbose) {
                                printf("to %i", dist[target]);
                            }
                        }
                        if (verbose) {
                            printf("\n");
                        }
                    }
                }
            }
            
        }
        if (verbose) {
            printf("\n");
        }
    }
    return dist;
}