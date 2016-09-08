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
void generateRandomGraph(GraphData *graph, int numVertices, int neighborsPerVertex, int numGraphs)
{
    graph->vertexCount = numVertices;
    graph->graphCount = numGraphs;
    graph->vertexArray = (int*) malloc(graph->vertexCount * sizeof(int));
    graph->maxVertexArray = (int*) malloc(graph->vertexCount * sizeof(int));
    graph->costArray = (float*) malloc(numGraphs * graph->vertexCount * sizeof(float));
    graph->sourceArray = (int*) malloc(graph->graphCount * sizeof(int));
    graph->edgeCount = numVertices * neighborsPerVertex;
    graph->edgeArray = (int*)malloc(graph->edgeCount * sizeof(int));
    graph->parentCountArray = (int*)malloc(graph->edgeCount * sizeof(int));
    graph->weightArray = (float*)malloc(numGraphs * graph->edgeCount * sizeof(float));
    
    for(int i = 0; i < graph->vertexCount; i++)
    {
        graph->vertexArray[i] = i * neighborsPerVertex;
        if (rand() % 100 < 34) {
            graph->maxVertexArray[i]=1;
        }
        else {
            graph->maxVertexArray[i]=0;
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
    
    
    
}