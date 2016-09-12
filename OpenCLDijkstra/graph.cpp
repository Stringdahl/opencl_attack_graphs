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
void generateRandomGraph(GraphData *graph, int vertexCount, int neighborsPerVertex, int graphCount, float probOfMax)
{
    graph->vertexCount = vertexCount;
    graph->graphCount = graphCount;
    graph->vertexArray = (int*) malloc(graph->vertexCount * sizeof(int));
    graph->inverseVertexArray = (int*) malloc(graph->vertexCount * sizeof(int));
    graph->maxVertexArray = (int*) malloc(graph->vertexCount * sizeof(int));
    graph->costArray = (float*) malloc(graphCount * graph->vertexCount * sizeof(float));
    graph->sourceArray = (int*) malloc(graph->graphCount * sizeof(int));
    graph->edgeCount = vertexCount * neighborsPerVertex;
    graph->edgeArray = (int*)malloc(graph->edgeCount * sizeof(int));
    graph->inverseEdgeArray = (int*)malloc(graph->edgeCount * sizeof(int));
    graph->parentCountArray = (int*)malloc(graph->edgeCount * sizeof(int));
    graph->weightArray = (float*)malloc(graphCount * graph->edgeCount * sizeof(float));
    graph->inverseWeightArray = (float*)malloc(graphCount * graph->edgeCount * sizeof(float));
    
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
        graph->weightArray[i] = (float)(rand() % 1000) / 1000.0f;
    }
    int tempSource = (rand() % graph->vertexCount);
    for(int i = 0; i < graphCount; i++)
    {
        graph->sourceArray[i] = tempSource;
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
                    graph->inverseWeightArray[iEdge]=graph->weightArray[edge];
                    iEdge ++;
                }
            }
        }
    }
}

// A utility function to find the vertex with minimum distance value, from
// the set of vertices not yet included in shortest path tree
int minDistance(float *dist, bool *sptSet, int vertexCount)
{
    // Initialize min value
    float min = FLT_MAX;
    int min_index = 0;
    
    for (int v = 0; v < vertexCount; v++) {
        //        printf("Node %i costs %.2f ", v, dist[v]);
        //        if (sptSet[v] == false) {
        //            printf("and is unprocessed. ");
        //        }
        //        else {
        //            printf("but is already proceessed. ");
        //        }
        if (sptSet[v] == false && dist[v] <= min) {
            min = dist[v], min_index = v;
        }
        //        printf("Lowest so far is %i with %.2f\n", min_index, min);
    }
    return min_index;
}



bool atLeastOneUnprocessedIsFinite(bool *sptSet, int vertexCount, float *dist) {
    for (int i = 0; i < vertexCount; i++) {
        if (!sptSet[i] && dist[i]<FLT_MAX)
            return true;
    }
    return false;
}


// Funtion that implements Dijkstra's single source shortest path algorithm
// for a graph represented using adjacency matrix representation
float* dijkstra(GraphData *graph, int iGraph, bool verbose){
    float *dist = (float*) malloc(sizeof(float) * graph->vertexCount);     // The output array.  dist[i] will hold the shortest
    // distance from src to i
    bool *sptSet = (bool*) malloc(sizeof(bool) * graph->vertexCount); // sptSet[i] will true if vertex i is included in shortest
    // path tree or shortest distance from src to i is finalized
    
    int vertexCount = graph->vertexCount;
    int edgeCount = graph->edgeCount;
    int sourceVertex = graph->sourceArray[iGraph];
    
    // Copy the appropriate graph from GraphData
    int *vertexArray = (int*)malloc(vertexCount * sizeof(int));
    int *parentCountArray = (int*)malloc(vertexCount * sizeof(int));
    float *maxVertexArray = (float*)malloc(vertexCount * sizeof(float));
    
    for (int iVertex=0; iVertex<vertexCount; iVertex++) {
        vertexArray[iVertex]=graph->vertexArray[iVertex];
        parentCountArray[iVertex]=graph->parentCountArray[iVertex];
        maxVertexArray[iVertex]=graph->maxVertexArray[iVertex];
    }
    
    int *edgeArray = (int*)malloc(edgeCount * sizeof(int));
    float *weightArray = (float*)malloc(edgeCount * sizeof(float));
    int *traversedEdgeCountArray = (int*)malloc(edgeCount * sizeof(int));
    
    for (int iEdge=0; iEdge<edgeCount; iEdge++) {
        edgeArray[iEdge] = graph->edgeArray[iEdge];
        weightArray[iEdge] = graph->weightArray[iGraph*edgeCount + iEdge];
        traversedEdgeCountArray[iEdge]=0;
    }
    
    
    // Initialize all distances as INFINITE and stpSet[] as false
    for (int i = 0; i < vertexCount; i++)
        dist[i] = FLT_MAX, sptSet[i] = false;
    
    // Distance of source vertex from itself is always 0
    dist[sourceVertex] = 0;
    if (verbose) {
        printf("Source vertex = %i.\n", sourceVertex);
    }
    // Find shortest path for all vertices
    while (atLeastOneUnprocessedIsFinite(sptSet, vertexCount, dist))
    {
        
        // Pick the minimum distance vertex from the set of vertices not
        // yet processed. u is always equal to src in first iteration.
        int source = minDistance(dist, sptSet, vertexCount);
        if (verbose) {
            printf("Node %i (of cost %.2f) ...", source, dist[source]);
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
                
                if (dist[source] != FLT_MAX) {
                    // If min node
                    if (maxVertexArray[target]<0) {
                        if (!sptSet[target]) {
                            if (verbose) {
                                printf(" looking at min node %i (with %i remainaing parents) by edge with weight %.2f.", target, parentCountArray[target], graph->weightArray[edge]);
                            }
                            if (dist[source]+weightArray[edge] < dist[target]) {
                                if (verbose) {
                                    printf(".. updated from %.2f ", dist[target]);
                                }
                                dist[target] = dist[source] + weightArray[edge];
                                if (verbose) {
                                    printf("to %.2f", dist[target]);
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
                            printf(" looking at max node %i (with %i remainaing parents, max: %.2f) by edge with weight %.2f.", target, parentCountArray[target], maxVertexArray[target], graph->weightArray[edge]);
                        }
                        if (maxVertexArray[target] < dist[source]+weightArray[edge]) {
                            maxVertexArray[target] = dist[source]+weightArray[edge];
                        }
                        if (parentCountArray[target]==0) {
                            if (verbose) {
                                printf(".. updated from %.2f ", dist[target]);
                            }
                            dist[target] = maxVertexArray[target];
                            if (verbose) {
                                printf("to %.2f", dist[target]);
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