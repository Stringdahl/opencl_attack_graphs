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
void generateRandomGraph(GraphData *graph, int numVertices, int neighborsPerVertex, int numGraphs, float probOfMax)
{
    graph->vertexCount = numVertices;
    graph->graphCount = numGraphs;
    graph->vertexArray = (int*) malloc(graph->vertexCount * sizeof(int));
    graph->inverseVertexArray = (int*) malloc(graph->vertexCount * sizeof(int));
    graph->maxVertexArray = (int*) malloc(graph->vertexCount * sizeof(int));
    graph->costArray = (float*) malloc(numGraphs * graph->vertexCount * sizeof(float));
    graph->sourceArray = (int*) malloc(graph->graphCount * sizeof(int));
    graph->edgeCount = numVertices * neighborsPerVertex;
    graph->edgeArray = (int*)malloc(graph->edgeCount * sizeof(int));
    graph->inverseEdgeArray = (int*)malloc(graph->edgeCount * sizeof(int));
    graph->parentCountArray = (int*)malloc(graph->edgeCount * sizeof(int));
    graph->weightArray = (float*)malloc(numGraphs * graph->edgeCount * sizeof(float));
    graph->inverseWeightArray = (float*)malloc(numGraphs * graph->edgeCount * sizeof(float));
    
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
    for(int i = 0; i < numGraphs * graph->edgeCount; i++)
    {
        graph->weightArray[i] = (float)(rand() % 1000) / 1000.0f;
    }
    for(int i = 0; i < numGraphs; i++)
    {
        graph->sourceArray[i] = (graph->vertexCount*i + rand() % graph->vertexCount);
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
    int min = INT_MAX, min_index = 0;
    
    for (int v = 0; v < vertexCount; v++)
        if (sptSet[v] == false && dist[v] <= min)
            min = dist[v], min_index = v;
    return min_index;
}

void updateMinVertex(int u, bool *sptSet, float *dist, GraphData *graph, int edge) {
    int v = graph->edgeArray[edge];
    if (dist[u] != FLT_MAX) {
        if (dist[u]+graph->weightArray[edge] < dist[v]) {
            dist[v] = dist[u] + graph->weightArray[edge];
        }
    }
    
}


// Funtion that implements Dijkstra's single source shortest path algorithm
// for a graph represented using adjacency matrix representation
float* dijkstra(GraphData *graph){
    float *dist = (float*) malloc(sizeof(float) * graph->vertexCount);     // The output array.  dist[i] will hold the shortest
    // distance from src to i
    bool *sptSet = (bool*) malloc(sizeof(bool) * graph->vertexCount); // sptSet[i] will true if vertex i is included in shortest
    // path tree or shortest distance from src to i is finalized
    
    // Initially, no edges have been travelled
    int *traversedEdgeCountArray = (int*)malloc(graph->edgeCount * sizeof(int));
    for (int iEdge=0; iEdge<graph->edgeCount; iEdge++) {
        traversedEdgeCountArray[iEdge]=0;
    }
    
    int *parentCountArray = (int*)malloc(graph->vertexCount * sizeof(int));
    for (int iVertex=0; iVertex<graph->vertexCount; iVertex++) {
        parentCountArray[iVertex]=graph->parentCountArray[iVertex];
    }
    
    float *maxVertexArray = (float*)malloc(graph->vertexCount * sizeof(float));
    for (int iVertex=0; iVertex<graph->vertexCount; iVertex++) {
        maxVertexArray[iVertex]=graph->maxVertexArray[iVertex];
    }
    
    
    // Initialize all distances as INFINITE and stpSet[] as false
    for (int i = 0; i < graph->vertexCount; i++)
        dist[i] = FLT_MAX, sptSet[i] = false;
    
    // Distance of source vertex from itself is always 0
    dist[graph->sourceArray[0]] = 0;
    
    // Find shortest path for all vertices
    for (int count = 0; count < graph->vertexCount-1; count++)
    {
        // Pick the minimum distance vertex from the set of vertices not
        // yet processed. u is always equal to src in first iteration.
        int u = minDistance(dist, sptSet, graph->vertexCount);
        
        if (maxVertexArray[u]<0 || parentCountArray[u]==0) {
            
            // Mark the picked vertex as processed
            sptSet[u] = true;
            
            // Get the edges
            int edgeStart = graph->vertexArray[u];
            int edgeEnd;
            if (u + 1 < (graph->vertexCount))
            {
                edgeEnd = graph->vertexArray[u + 1];
            }
            else
            {
                edgeEnd = graph->edgeCount;
            }
            // Iterate over the edges
            
            for(int edge = edgeStart; edge < edgeEnd; edge++) {
                int v = graph->edgeArray[edge];
                if (!sptSet[v]) {
                    
                    if (traversedEdgeCountArray[edge]==0) {
                        parentCountArray[v]--;
                    }
                    traversedEdgeCountArray[edge]++;
                    // If min node
                    if (maxVertexArray[v]<0) {
                        if (dist[u] != FLT_MAX) {
                            if (dist[u]+graph->weightArray[edge] < dist[v]) {
                                dist[v] = dist[u] + graph->weightArray[edge];
                            }
                        }
                    }
                    // If max node
                    else {
                        if (maxVertexArray[v] < dist[u]+graph->weightArray[edge]) {
                            maxVertexArray[v] = dist[u]+graph->weightArray[edge];
                        }
                        if (parentCountArray[v]==0) {
                            dist[v] = maxVertexArray[v];
                        }
                    }
                }
            }
        }
    }
    return dist;
}