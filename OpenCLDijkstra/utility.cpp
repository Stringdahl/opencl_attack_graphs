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
//  Namespaces
//
using namespace std;


///
//  Utility functions adapted from NVIDIA GPU Computing SDK
//
void checkErrorFileLine(int errNum, int expected, const char* file, const int lineNumber);


void printCostOfRandomVertices(int *costArrayHost, int verticesToPrint, int totalVerticeCount) {
    for(int i = 0; i < verticesToPrint; i++)
    {
        int iVertice = rand() % totalVerticeCount;
        printf("Cost of node %i is %i\n", iVertice, costArrayHost[iVertice]);
    }
}

void printCostOfVertex(GraphData *graph, int vertexToPrint) {
    
    for(int i = 0; i < graph->graphCount; i++)
    {
        if (graph->costArray[i*graph->vertexCount + vertexToPrint] == 2147482646)
            printf("TTC of attack step %i is infinite.\n", vertexToPrint);
        else
            printf("TTC of attack step %i is %i.\n", vertexToPrint, graph->costArray[i*graph->vertexCount + vertexToPrint]);
    }
    printf("\n");
}

bool contains(int *array, int arrayLength, int value) {
    for (int i = 0; i < arrayLength; i++) {
        if (array[i]==value) {
            return true;
        }
    }
    return false;
}


void printGraph(GraphData *graph, char **verticeNameArray, int iGraph) {
    int nChildren;
    for (int localSource=0; localSource<graph->vertexCount; localSource++) {
        if (localSource<graph->vertexCount-1) {
            nChildren = graph->vertexArray[localSource+1]-graph->vertexArray[localSource];
        }
        else {
            nChildren = graph->edgeCount-graph->vertexArray[localSource];
        }
        printf("Vertex %s has %i children\n", verticeNameArray[localSource], nChildren);
        if (contains(graph->sourceArray, graph->sourceCount, localSource)) {
            printf("Vertex %i is source.\n", localSource);
        }
        for (int iChild=0; iChild<nChildren; iChild++) {
            int localEdge = graph->vertexArray[localSource]+iChild;
            int localTarget = graph->edgeArray[localEdge];
            int globalSource = iGraph*graph->vertexCount + localSource;
            int globalTarget = iGraph*graph->vertexCount + localTarget;
            int globalEdge = iGraph*graph->edgeCount + localEdge;
            printf("Vertex %s (%i) is parent to vertex %s (%i) with edge weight of %i\n", verticeNameArray[localSource], graph->costArray[globalSource], verticeNameArray[localTarget], graph->costArray[globalTarget], graph->weightArray[globalEdge]);
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
            printf("Vertex %i is child to vertex %i with edge weight of %i\n", iNode, graph->inverseEdgeArray[graph->inverseVertexArray[iNode]+iParent], graph->inverseWeightArray[graph->inverseVertexArray[iNode]+iParent]);
        }
    }
}

void printParents(GraphData *graph) {
    for (int i = 0; i < graph->vertexCount; i++) {
        printf("Vertex %i has %i parents.\n", i, graph->parentCountArray[i]);
    }
}

void printMax(GraphData *graph) {
    for (int i = 0; i < graph->vertexCount; i++) {
        printf("Vertex %i has max value %i.\n", i, graph->maxVertexArray[i]);
    }
}

void printWeights(GraphData *graph) {
    for (int i = 0; i < graph->graphCount*graph->edgeCount; i++) {
        printf("weightArray[%i] = %i\n", i, graph->weightArray[i]);
    }
}

void printInverseWeights(GraphData *graph) {
    for (int i = 0; i < graph->graphCount*graph->edgeCount; i++) {
        printf("inverseWeightArray[%i] = %i\n", i, graph->inverseWeightArray[i]);
    }
}

void printSources(GraphData *graph, char **verticeNameArray) {
    printf("Nodes ");
    for (int i = 0; i < graph->sourceCount; i++) {
        printf("%s, ", verticeNameArray[graph->sourceArray[i]]);
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
    
    int *costArrayHost = (int*) malloc(sizeof(int) * totalVertexCount);
    int *updatingCostArrayHost = (int*) malloc(sizeof(int) * totalVertexCount);
    int *weightArrayHost = (int*) malloc(sizeof(int) * totalEdgeCount);
    int *maxVertexArrayHost = (int*) malloc(sizeof(int) * totalVertexCount);
    int *parentCountArrayHost = (int*) malloc(sizeof(int) * totalVertexCount);
    int *maskArrayHost = (int*) malloc(sizeof(int) * totalVertexCount);
    
    
    errNum = clEnqueueReadBuffer(*commandQueue, *costArrayDevice, CL_FALSE, 0, sizeof(int) * totalVertexCount, costArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *maxVerticeArrayDevice, CL_FALSE, 0, sizeof(int) * totalVertexCount, maxVertexArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *weightArrayDevice, CL_FALSE, 0, sizeof(int) * totalEdgeCount, weightArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *parentCountArrayDevice, CL_FALSE, 0, sizeof(int) * graph->graphCount*graph->vertexCount, parentCountArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *maskArrayDevice, CL_FALSE, 0, sizeof(int) * graph->graphCount*graph->vertexCount, maskArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *updatingCostArrayDevice, CL_FALSE, 0, sizeof(int) * totalVertexCount, updatingCostArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    clWaitForEvents(1, &readDone);
    
    for (int tid = 0; tid < totalVertexCount; tid++) {
        int localTid = tid % graph -> vertexCount;
        int iGraph = tid / graph -> vertexCount;
        
        if (tid == iVertex || iVertex == -1) {
            printf("Node %i: Mask: %i, Cost: %i, updatingCost: %i, max: %i, parentCount: %i.\n", tid, maskArrayHost[tid], costArrayHost[tid], updatingCostArrayHost[tid], maxVertexArrayHost[tid], parentCountArrayHost[tid]);
            
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
                printf("Influences node %i through edge %i by weight %i.\n", (iGraph*graph->vertexCount + nid), edge, graph->weightArray[iGraph*graph->vertexCount + edge]);
            }
        }
    }
}

void printAfterUpdating(GraphData *graph, cl_command_queue *commandQueue, int *maskArrayHost, cl_mem *costArrayDevice, cl_mem *updatingCostArrayDevice, cl_mem *weightArrayDevice, cl_mem *parentCountArrayDevice, cl_mem *maxVerticeArrayDevice) {
    
    int errNum = 0;
    cl_event readDone;
    
    int totalVertexCount = graph->graphCount * graph->vertexCount;
    int totalEdgeCount = graph->graphCount * graph->edgeCount;
    
    int *costArrayHost = (int*) malloc(sizeof(int) * totalVertexCount);
    int *updatingCostArrayHost = (int*) malloc(sizeof(int) * totalVertexCount);
    int *weightArrayHost = (int*) malloc(sizeof(int) * totalEdgeCount);
    int *maxVertexArrayHost = (int*) malloc(sizeof(int) * totalVertexCount);
    int *parentCountArrayHost = (int*) malloc(sizeof(int) * graph->graphCount*graph->vertexCount);
    
    
    errNum = clEnqueueReadBuffer(*commandQueue, *costArrayDevice, CL_FALSE, 0, sizeof(int) * totalVertexCount, costArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *updatingCostArrayDevice, CL_FALSE, 0, sizeof(int) * totalVertexCount, updatingCostArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *maxVerticeArrayDevice, CL_FALSE, 0, sizeof(int) * totalVertexCount, maxVertexArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *weightArrayDevice, CL_FALSE, 0, sizeof(int) * totalEdgeCount, weightArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(*commandQueue, *parentCountArrayDevice, CL_FALSE, 0, sizeof(int) * graph->graphCount*graph->vertexCount, parentCountArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    clWaitForEvents(1, &readDone);
    
    printf("%i vertices.\n", totalVertexCount);
    for (int tid = 0; tid < totalVertexCount; tid++) {
        if ( maskArrayHost[tid] != 0 ) {
            printf("Vertex %i, (max: %i, %i remaining parents) is considered for updating.\n", tid, maxVertexArrayHost[tid], parentCountArrayHost[tid]);
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
                    
                    printf("Node %i (of cost %i and updatingCost %i) updated node %i (of max %i with %i remaining parents) by edge %i with weight %i. Node %i now has cost %i and updatingCost %i.\n", tid, costArrayHost[tid], updatingCostArrayHost[tid], nid, maxVertexArrayHost[nid], parentCountArrayHost[nid], edge, graph->weightArray[eid], nid, costArrayHost[nid], updatingCostArrayHost[nid]);
                    
                    if (maxVertexArrayHost[nid]>=0)
                    {
                        printf("Updated a max node.\n");
                        printf("Perhaps maxVertexArray was increased. It is now %i.\n", maxVertexArrayHost[nid]);
                        if (parentCountArrayHost[nid]==0) {
                            printf("All parents visited. Set updatingCostArray[nid] (%i) to maxVertexArray[nid] (%i).\n", updatingCostArrayHost[nid], maxVertexArrayHost[nid]);
                        }
                    }
                }
            }
        }
    }
}

const char* costToString(int cost) {
    static char str[8];
    if (cost == INT_MAX) {
        sprintf(str, "inf");
    }
    else {
        sprintf(str, "%i", cost);
    }
    return str;
}

int compare(const void * a, const void * b)
{
    return ( *(int*)a - *(int*)b );
}

int median(int array[], int nArray)
{
    qsort (array, nArray, sizeof(int), compare);
    for (int i = 0; i < nArray; i++) {
    }
    return array[nArray/2];
}

// This function is destructuve, overwriting the first sample of costs with median values.
void getMedianGraph(GraphData *graph) {
    int *costSampleArray = (int*) malloc(sizeof(int) * graph->graphCount);
    for (int iVertex = 0; iVertex < graph->vertexCount; iVertex++) {
        for (int iGraph = 0; iGraph < graph->graphCount; iGraph++) {
            costSampleArray[iGraph] = graph->costArray[iGraph*graph->vertexCount + iVertex];
        }
        graph->costArray[iVertex] = median(costSampleArray, graph->graphCount);
    }
}

void printMathematicaString(GraphData *graph, int iGraph, bool printSum) {
    char str[128*graph->edgeCount];
    int *hasEdge = (int*) malloc(sizeof(int) * graph->vertexCount);
    
    
    sprintf(str, "Graph[{");
    for (int localSource = 0; localSource < graph->vertexCount; localSource++) {
        hasEdge[localSource]=0;
        int globalSource = iGraph*graph->vertexCount + localSource;
        int edgeStart = graph->vertexArray[localSource];
        int edgeEnd = getEdgeEnd(localSource, graph->vertexCount, graph->vertexArray, graph->edgeCount);
        for(int edge = edgeStart; edge < edgeEnd; edge++) {
            int localTarget = graph->edgeArray[edge];
            int globalTarget = iGraph*graph->vertexCount + localTarget;
            sprintf(str + strlen(str), "%i \\[DirectedEdge] %i, ", globalSource, globalTarget);
            hasEdge[localSource]=1;
            hasEdge[localTarget]=1;
        }
    }
    
    sprintf(str + strlen(str)-2, "}, VertexLabels -> {");
    for (int vertex = 0; vertex < graph->vertexCount; vertex++) {
        if (hasEdge[vertex]) {
            int globalVertex = iGraph*graph->vertexCount + vertex;
            const char* maxSourceString = costToString(graph->costArray[globalVertex]);
            sprintf(str + strlen(str), "%i -> \"%i [%s", globalVertex, globalVertex, maxSourceString);
            if (printSum) {
            const char* sumSourceString = costToString(graph->sumCostArray[globalVertex]);
            sprintf(str + strlen(str), "-%s]\", ", sumSourceString);
            }
            else {
                sprintf(str + strlen(str), "]\", ");
            }
        }
    }
    sprintf(str + strlen(str)-2, "}, EdgeLabels -> {");
    for (int localSource = 0; localSource < graph->vertexCount; localSource++) {
        int globalSource = iGraph*graph->vertexCount + localSource;
        int edgeStart = graph->vertexArray[localSource];
        int edgeEnd = getEdgeEnd(localSource, graph->vertexCount, graph->vertexArray, graph->edgeCount);
        for(int localEdge = edgeStart; localEdge < edgeEnd; localEdge++) {
            int localTarget = graph->edgeArray[localEdge];
            int globalTarget = iGraph*graph->vertexCount + localTarget;
            int globalEdge = iGraph*graph->edgeCount + localEdge;
            const char* edgeString = costToString(graph->weightArray[globalEdge]);
            sprintf(str + strlen(str), "%i \\[DirectedEdge] %i -> %s, ", globalSource, globalTarget, edgeString);
        }
    }
    sprintf(str + strlen(str)-2, "}, EdgeStyle -> {");
    for (int localSource = 0; localSource < graph->vertexCount; localSource++) {
        int globalSource = iGraph*graph->vertexCount + localSource;
        int edgeStart = graph->vertexArray[localSource];
        int edgeEnd = getEdgeEnd(localSource, graph->vertexCount, graph->vertexArray, graph->edgeCount);
        for(int localEdge = edgeStart; localEdge < edgeEnd; localEdge++) {
            int localTarget = graph->edgeArray[localEdge];
            int globalTarget = iGraph*graph->vertexCount + localTarget;
            int globalEdge = iGraph*graph->edgeCount + localEdge;
            //printf("Checking if edge %i from %i to %i is shortest.\n", globalEdge, globalSource, globalTarget);
            if (graph->shortestParentsArray[globalEdge]==1) {
                //printf("It is. \n");
                sprintf(str + strlen(str), "%i \\[DirectedEdge] %i -> Red, ", globalSource, globalTarget);
            }
        }
    }
    sprintf(str + strlen(str)-2, "}, VertexShapeFunction -> {");
    for (int localVertex = 0; localVertex < graph->vertexCount; localVertex++) {
        if (hasEdge[localVertex]) {
            int globalVertex = iGraph*graph->vertexCount + localVertex;
            if (graph->sourceArray[globalVertex]) {
                sprintf(str + strlen(str), "%i -> \"Star\", ", globalVertex);
            }
            else {
                if (graph->maxVertexArray[localVertex]>=0) {
                    sprintf(str + strlen(str), "%i -> \"Square\", ", globalVertex);
                }
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
    int *maxVertexArrayHost = (int*) malloc(sizeof(int) * graph->graphCount*graph->vertexCount);
    cl_event readDone;
    
    int errNum = clEnqueueReadBuffer(*commandQueue, *maxVertexArrayDevice, CL_FALSE, 0, sizeof(int) * graph->graphCount*graph->vertexCount, maxVertexArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    clWaitForEvents(1, &readDone);
    
    for (int iGraph=0; iGraph < graph->graphCount; iGraph++) {
        for (int iVertex=0; iVertex<graph->vertexCount; iVertex++) {
            int iGlobalVertex =iGraph * graph->vertexCount + iVertex;
            printf("Max of vertex %i is %i.\n",iGlobalVertex, maxVertexArrayHost[iGlobalVertex]);
        }
    }
}

// A utility function to print the constructed distance array
void printSolution(int *dist, int n)
{
    printf("Vertex   Distance from Source\n");
    for (int i = 0; i < n; i++)
        printf("%d \t\t %i\n", i, dist[i]);
}

void compareToCPUComputation(GraphData *graph, bool verbose, int nGraphsToCheck) {
    if (nGraphsToCheck>graph->graphCount) {
        nGraphsToCheck=graph->graphCount;
    }
    printf("Checking correctness against sequential implementation in %i graphs.\n", nGraphsToCheck);
    int nInfinite = 0;
    int iErrors = 0;
    for (int iCheck = 0; iCheck<nGraphsToCheck; iCheck++) {
        //int iGraph = rand() % graph->graphCount;
        int iGraph = iCheck;
        //printf("Checking graph %i.\n", iGraph);
        int *dist = dijkstra(graph, iGraph, verbose);
        for (int iVertex = 0; iVertex < graph->vertexCount; iVertex++) {
            if (verbose) {
                printf("%i: CPU=%i, GPU=%i\n", iVertex, dist[iVertex], graph->costArray[iGraph*graph->vertexCount + iVertex]);
            }
            if (dist[iVertex] != graph->costArray[iGraph*graph->vertexCount + iVertex] || graph->costArray[iGraph*graph->vertexCount + iVertex] != dist[iVertex]) {
                printf("CPU computed %i for vertex %i while GPU computed %i\n", dist[iVertex], iVertex, graph->costArray[iGraph*graph->vertexCount + iVertex]);
                iErrors++;
                // exit(1);
            }
            if (graph->costArray[iGraph*graph->vertexCount + iVertex] == INT_MAX) {
                nInfinite++;
            }
        }
    }
    printf("%i errors.\n", iErrors);
    printf("On average %.0f%% infinite-time attack steps.\n", 100*(float)nInfinite/(nGraphsToCheck*graph->vertexCount));
}

#define PRECISION 1000


int getEdgeEnd(int iVertex, int vertexCount, int *vertexArray, int edgeCount) {
    if (iVertex + 1 < (vertexCount))
        return vertexArray[iVertex + 1];
    else
        return edgeCount;
}

void maxSumDifference(GraphData *graph) {
    long diff = 0;
    long sum = 0;
    int nDiff = 0;
    float maxDiff = 0;
    int iMaxDiff = -1;
    int totalVerticeCount = graph->graphCount*graph->vertexCount;
    for (int i=0; i<totalVerticeCount; i++) {
        sum = sum + graph->costArray[i];
        if (graph->sumCostArray[i] != graph->costArray[i]) {
            diff = diff + graph->sumCostArray[i] - graph->costArray[i];
            nDiff ++;
            float currDiff = 100*(float)(graph->sumCostArray[i] - graph->costArray[i])/graph->costArray[i];
            if (currDiff > maxDiff) {
                maxDiff = currDiff;
                iMaxDiff = i;
            }
        }
    }
    printf("The average difference between max and sum is %f%% per node. %.2f%% of nodes were approximated.\n", 100*(float)diff/(float)sum, 100*(float)nDiff/totalVerticeCount);
    printf("The biggeste single difference was %.2f%% in node %i (%i).\n", maxDiff, iMaxDiff % graph->graphCount, iMaxDiff);
}

void writeGraphToFile(GraphData *graph, char filePath[512]) {
    ofstream myfile;
    myfile.open (filePath);
    myfile << graph->graphCount << "," << graph->vertexCount << "," << graph->edgeCount << "," << graph->graphCount << ",\n";
    for (int iVertex = 0; iVertex < graph->vertexCount - 1; iVertex++) {
        myfile << graph->vertexArray[iVertex] << ",";
    }
    myfile << graph->vertexArray[graph->vertexCount - 1] << ",\n";
    
    for (int iVertex = 0; iVertex < graph->vertexCount - 1; iVertex++) {
        myfile << graph->maxVertexArray[iVertex] << ",";
    }
    myfile << graph->maxVertexArray[graph->vertexCount - 1] << ",\n";
    
    for (int iSource = 0; iSource < graph->graphCount * graph->vertexCount - 1; iSource++) {
        myfile << graph->sourceArray[iSource] << ",";
    }
    myfile << graph->sourceArray[graph->graphCount * graph->vertexCount - 1] << ",\n";
    
    for (int iEdge = 0; iEdge < graph->edgeCount - 1; iEdge++) {
        myfile << graph->edgeArray[iEdge] << ",";
    }
    myfile << graph->edgeArray[graph->edgeCount - 1] << ",\n";
    
    for (int iWeight = 0; iWeight < graph->graphCount * graph->edgeCount- 1; iWeight++) {
        myfile << graph->weightArray[iWeight] << ",";
    }
    myfile << graph->weightArray[graph->graphCount * graph->edgeCount - 1] << ",\n";

    for (int iVertex = 0; iVertex < graph->graphCount * graph->vertexCount - 1; iVertex++) {
        myfile << graph->costArray[iVertex] << ",";
    }
    myfile << graph->costArray[graph->graphCount * graph->vertexCount - 1] << ",\n";

    for (int iShortestParent = 0; iShortestParent < graph->graphCount * graph->edgeCount- 1; iShortestParent++) {
        myfile << graph->shortestParentsArray[iShortestParent] << ",";
    }
    myfile << graph->shortestParentsArray[graph->graphCount * graph->edgeCount - 1] << ",\n";

    myfile.close();
}

void readGraphFromFile(GraphData *graph, char filePath[512]) {
    char line[512];
    ifstream myfile;
    myfile.open (filePath);
    if (myfile.is_open())
    {
        myfile.getline (line, 64, ',');
        graph->graphCount = (int)std::strtol(line, NULL, 10);
        myfile.getline (line, 64, ',');
        graph->vertexCount = (int)std::strtol(line, NULL, 10);
        myfile.getline (line, 64, ',');
        graph->edgeCount = (int)std::strtol(line, NULL, 10);
        myfile.getline (line, 64, ',');
        graph->sourceCount = (int)std::strtol(line, NULL, 10);
        graph->vertexArray = (int*) malloc(graph->vertexCount * sizeof(int));
        for (int iVertex = 0; iVertex<graph->vertexCount; iVertex++) {
            myfile.getline (line, 64, ',');
            graph->vertexArray[iVertex] = (int)std::strtol(line, NULL, 10);
        }
        graph->maxVertexArray = (int*) malloc(graph->vertexCount * sizeof(int));
        for (int iVertex = 0; iVertex<graph->vertexCount; iVertex++) {
            myfile.getline (line, 64, ',');
            graph->maxVertexArray[iVertex] = (int)std::strtol(line, NULL, 10);
        }
        graph->sourceArray = (int*) malloc(graph->graphCount * graph->vertexCount * sizeof(int));
        for (int iSource = 0; iSource < graph->graphCount * graph->vertexCount; iSource++) {
            myfile.getline (line, 64, ',');
            graph->sourceArray[iSource] = (int)std::strtol(line, NULL, 10);

        }
        graph->edgeArray = (int*) malloc(graph->edgeCount * sizeof(int));
        for (int iEdge = 0; iEdge<graph->edgeCount; iEdge++) {
            myfile.getline (line, 64, ',');
            graph->edgeArray[iEdge] = (int)std::strtol(line, NULL, 10);
        }
        graph->weightArray = (int*) malloc(graph->graphCount * graph->edgeCount * sizeof(int));
        for (int iWeight = 0; iWeight<(graph->graphCount * graph->edgeCount); iWeight++) {
            myfile.getline (line, 64, ',');
            graph->weightArray[iWeight] = (int)std::strtol(line, NULL, 10);
        }
        myfile.close();
    }
    else cout << "Unable to open file";
}

void readVerticeNames(char filePath[512], char **verticeNameArray) {
    ifstream myfile;
    myfile.open (filePath);
    if (myfile.is_open()) {
        int iVertex = 0;
        string s;
        while( getline(myfile, s) ) {
            strcpy(verticeNameArray[iVertex], s.c_str());
            iVertex++;
        }
        myfile.close();
    }
    else cout << "Unable to open file";
}

