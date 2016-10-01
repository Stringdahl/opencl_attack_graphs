#define MAXTRACE 20000


int getEdgeEnd(int iVertex, int vertexCount, __global int *vertexArray, int edgeCount) {
    if (iVertex + 1 < (vertexCount))
        return vertexArray[iVertex + 1];
    else
        return edgeCount;
}

//int getInfluentialParents(int *influentialEdges, int nInfluentialEdges, int globalChild,  int vertexCount, int edgeCount, __global int *inverseVertexArray,  __global int *inverseEdgeArray, __global int *maxVertexArray,  __global int *maxCostArray,  __global int *inverseWeightArray) {
//    influentialEdges[0] = 3;
//    int minEdgeVal = INT_MAX;
//    int minParent = -1;
//    int iGraph = globalChild / vertexCount;
//    int localChild = globalChild % vertexCount;
//    int inverseEdgeStart = inverseVertexArray[localChild];
//    int inverseEdgeEnd = getEdgeEnd(localChild, vertexCount, inverseVertexArray, edgeCount);
//    if (inverseEdgeEnd-inverseEdgeStart <= 0) {
//        return -1;
//    }
//    if (maxVertexArray[localChild]<0) {
//        for(int localInverseEdge = inverseEdgeStart; localInverseEdge < inverseEdgeEnd; localInverseEdge++) {
//            int localParent = inverseEdgeArray[localInverseEdge];
//            int globalParent = iGraph*vertexCount + localParent;
//            int globalInverseEdge = iGraph*edgeCount + localInverseEdge;
//            int currEdgeVal = maxCostArray[globalParent] + inverseWeightArray[globalInverseEdge];
//            if (currEdgeVal<minEdgeVal) {
//                minEdgeVal = currEdgeVal;
//                minParent = globalParent;
//            }
//        }
//    }
//    return minParent;
//}

//int leastCommonAncestor(int localParent1, int localParent2, int vertexCount, int edgeCount, __global int *maxVertexArray, __global int *inverseVertexArray,  __global int *inverseEdgeArray,  __global int *maxCostArray,  __global int *inverseWeightArray) {
//    //printf("Parent 1 = %i, parent 2 = %i.\n", localParent1, localParent2);
//    int influentialEdges[MAXTRACE];
//    int nInfluentialEdges = 0;
//    //int ip1 = getInfluentialParents(influentialEdges, nInfluentialEdges, localParent1, vertexCount, edgeCount, inverseVertexArray, inverseEdgeArray, maxVertexArray, maxCostArray, inverseWeightArray);
//    printf("influentialEdges[0] = %i\n", influentialEdges[0]);
//    //printf("Node %i is the influential parent of node %i.\n", ip1, localParent1);
//    return 0;
//}
//
//
//int trueValueOfAndVertice(int globalTarget,  int vertexCount, int edgeCount,  __global int *maxCostArray, __global int *maxVertexArray,  __global int *inverseVertexArray,  __global int *inverseEdgeArray,  __global int *inverseWeightArray) {
//    int localTarget = globalTarget % vertexCount;
//    int inverseEdgeStart = inverseVertexArray[localTarget];
//    int inverseEdgeEnd = getEdgeEnd(localTarget, vertexCount, inverseVertexArray, edgeCount);
//    for(int localInverseEdge1 = inverseEdgeStart; localInverseEdge1 < inverseEdgeEnd; localInverseEdge1++) {
//        for(int localInverseEdge2 = localInverseEdge1 + 1; localInverseEdge2 < inverseEdgeEnd; localInverseEdge2++) {
//            int localInverseParent1 = inverseEdgeArray[localInverseEdge1];
//            int localInverseParent2 = inverseEdgeArray[localInverseEdge2];
//            //printf("In node %i, looking for lca.\n", globalTarget);
//            //            int lca = leastCommonAncestor(localInverseParent1, localInverseParent2, vertexCount, edgeCount, maxVertexArray, inverseVertexArray, inverseEdgeArray, maxCostArray, inverseWeightArray);
//        }
//    }
//    return 0;
//}

__kernel void OCL_SSSP_KERNEL1(__global int *vertexArray, __global int *inverseVertexArray, __global int *edgeArray, __global int *inverseEdgeArray, __global int *weightArray, __global int *inverseWeightArray, __global int *maskArray, __global int *maxCostArray, __global int *maxUpdatingCostArray, __global int *sumCostArray, __global int *sumUpdatingCostArray, int vertexCount, int edgeCount, __global int *traversedEdgeCountArray, __global int *parentCountArray, __global int *maxVertexArray, __global int *influentialParentArray)
{
    // access thread id
    int globalSource = get_global_id(0);
    
    int iGraph = globalSource / vertexCount;
    int localSource = globalSource % vertexCount;
    
    // Only consider vertices that are marked for update
    if ( maskArray[globalSource] != 0 ) {
        // After attempting to update, don't do it again unless (i) a parent updated this, or (ii) recalculation is required due to kernel 2.
        maskArray[globalSource] = 0;
        // Only update if (i) this is a min node, or (ii) this is a max node and all parents have been visited.
        if (maxVertexArray[globalSource]<0 || parentCountArray[globalSource]==0) {
            {
                // Get the edges
                int edgeStart = vertexArray[localSource];
                int edgeEnd = getEdgeEnd(localSource, vertexCount, vertexArray, edgeCount);
                
                // Iterate over the edges
                for(int localEdge = edgeStart; localEdge < edgeEnd; localEdge++)
                {
                    int localTarget = edgeArray[localEdge];
                    int globalTarget = iGraph*vertexCount + edgeArray[localEdge];
                    int globalEdge = iGraph*edgeCount + localEdge;
                    
                    // If this edge has never been traversed, reduce the remaining parents of the target by one, so that they reach zero when all incoming edges have been visited.
                    if (traversedEdgeCountArray[globalEdge] == 0) {
                        atomic_dec(&parentCountArray[globalTarget]);
                    }
                    // Mark that this edge has been traversed.
                    traversedEdgeCountArray[globalEdge] ++;
                    int inverseEdgeStart = inverseVertexArray[localTarget];
                    int inverseEdgeEnd = getEdgeEnd(localTarget, vertexCount, inverseVertexArray, edgeCount);
                    
                    // If this is a min node ...
                    if (maxVertexArray[globalTarget]<0) {
                        long currentMaxCost = maxCostArray[globalSource];
                        long currentWeight = weightArray[globalEdge];
                        if (currentMaxCost + currentWeight < INT_MAX)
                            currentMaxCost = currentMaxCost + currentWeight;
                        else
                            currentMaxCost = INT_MAX;
                        
                        // ...atomically choose the lesser of the current and candidate updatingCost
                        atomic_min(&maxUpdatingCostArray[globalTarget], currentMaxCost);
                        atomic_min(&sumUpdatingCostArray[globalTarget], currentMaxCost);
                        
                    }
                    
                    // If this is a max node...
                    else {
                        if (parentCountArray[globalTarget]==0) {
                            // If all parents have been visited ...
                            // Iterate over the edges
                            int maxEdgeVal = 0;
                            int sumEdgeVal = 0;
                            int trueValue = 0;
                            
                            for(int localInverseEdge = inverseEdgeStart; localInverseEdge < inverseEdgeEnd; localInverseEdge++) {
                                int localInverseTarget = inverseEdgeArray[localInverseEdge];
                                int globalInverseTarget = iGraph*vertexCount + localInverseTarget;
                                int globalInverseEdge = iGraph*edgeCount + localInverseEdge;
                                int currEdgeVal;
                                long currentMaxCost = maxCostArray[globalInverseTarget];
                                long currentWeight = inverseWeightArray[globalInverseEdge];
                                if (currentMaxCost + currentWeight < INT_MAX)
                                    currEdgeVal = currentMaxCost + currentWeight;
                                else
                                    currEdgeVal = INT_MAX;
                                if (currEdgeVal>maxEdgeVal) {
                                    maxEdgeVal = currEdgeVal;
                                }
                                long longSumEdgeVal = sumEdgeVal;
                                longSumEdgeVal = longSumEdgeVal + currEdgeVal;
                                if (longSumEdgeVal < INT_MAX)
                                    sumEdgeVal = sumEdgeVal + currEdgeVal;
                                else
                                    sumEdgeVal = INT_MAX;
                                if(weightArray[localEdge] != inverseWeightArray[globalInverseEdge]) {
                                    printf("Error: Updating max vertex %i from parent %i. Parent %i has cost %i, inverse edge %i has weight %i. My updated cost is %i. Local edge from parent to child is %i, with a weight of %i.\n", globalTarget, globalSource, globalInverseTarget, currentMaxCost, globalInverseEdge, currentWeight, maxEdgeVal, localEdge, weightArray[localEdge]);
                                }
                            }
                            maxCostArray[globalTarget] = maxEdgeVal;
                            maxUpdatingCostArray[globalTarget] = maxEdgeVal;
                            sumCostArray[globalTarget] = sumEdgeVal;
                            sumUpdatingCostArray[globalTarget] = sumEdgeVal;
                            // Mark the target for update
                            maskArray[globalTarget] = 1;
                        }
                    }
                }
            }
        }
    }
}


__kernel void OCL_SSSP_KERNEL2(__global int *vertexArray, __global int *edgeArray, __global int *weightArray,
                               __global int *maskArray, __global int *maxCostArray, __global int *maxUpdatingCostArray, __global int *sumCostArray, __global int *sumUpdatingCostArray, int vertexCount, __global int *maxVertexArray)
{
    // access thread id
    int tid = get_global_id(0);
    
    if (maxCostArray[tid] > maxUpdatingCostArray[tid])
    {
        maxCostArray[tid] = maxUpdatingCostArray[tid];
        maskArray[tid] = 1;
    }
    if (sumCostArray[tid] > sumUpdatingCostArray[tid])
    {
        sumCostArray[tid] = sumUpdatingCostArray[tid];
        maskArray[tid] = 1;
    }
    
    maxUpdatingCostArray[tid] = maxCostArray[tid];
    sumUpdatingCostArray[tid] = sumCostArray[tid];
}


int getEdgeId(int globalParent, int globalChild, int weight, int vertexCount, int edgeCount, __global int *vertexArray, __global int *edgeArray, __global int *weightArray) {
    int iGraph = globalParent / vertexCount;
    int localParent = globalParent % vertexCount;
    int edgeStart = vertexArray[localParent];
    int edgeEnd = getEdgeEnd(localParent, vertexCount, vertexArray, edgeCount);
    
    // Iterate over the edges
    for(int localEdge = edgeStart; localEdge < edgeEnd; localEdge++)
    {
        int currentLocalChild = edgeArray[localEdge];
        int currentGlobalChild = iGraph*vertexCount + currentLocalChild;
        int globalEdge = iGraph*edgeCount + localEdge;
        printf("Checking getEdgeId() for globalParent = %i, localParent = %i, globalChild = %i, curreltLocalChild = %i, currentGlobalChild = %i, weight = %i, globalEdge = %i, weightArray[globalEdge] = %i\n", globalParent, localParent, globalChild, currentLocalChild, currentGlobalChild, weight, globalEdge, weightArray[globalEdge]);
        
        if (currentGlobalChild == globalChild && weightArray[globalEdge] == weight) {
            return globalEdge;
        }
    }
    return -1;
}


__kernel void SHORTEST_PARENTS(int vertexCount, int edgeCount,
                               __global int *vertexArray,
                               __global int *inverseVertexArray,
                               __global int *edgeArray,
                               __global int *inverseEdgeArray,
                               __global int *weightArray,
                               __global int *inverseWeightArray,
                               __global int *maxCostArray,
                               __global int *maxUpdatingCostArray,
                               __global int *maxVertexArray,
                               __global int *shortestParentEdgeArray)
{
    // access thread id
    int globalChild = get_global_id(0);
    if (maxCostArray[globalChild] != INT_MAX) {
        
        int iGraph = globalChild / vertexCount;
        int localChild = globalChild % vertexCount;
        
        int inverseEdgeStart = inverseVertexArray[localChild];
        int inverseEdgeEnd = getEdgeEnd(localChild, vertexCount, inverseVertexArray, edgeCount);
        int minCost = INT_MAX-1;
        
        for(int localParentEdge = inverseEdgeStart; localParentEdge < inverseEdgeEnd; localParentEdge++) {
            int localParent = inverseEdgeArray[localParentEdge];
            int globalParent = iGraph*vertexCount + localParent;
            int globalParentEdge = iGraph*edgeCount + localParentEdge;
            int edge = getEdgeId(globalParent, globalChild, inverseWeightArray[globalParentEdge], vertexCount, edgeCount, vertexArray, edgeArray, weightArray);
            if (edge == -1) {
                printf("ERROR! getEdgeId(%i, %i, %i, ...) returned -1\n", globalParent, globalChild, inverseWeightArray[globalParentEdge]);
            }
            // If this is a min node...
            if (maxVertexArray[localChild] < 0) {
                int currCost;
                long currentMaxCost = maxCostArray[globalParent];
                long currentWeight = inverseWeightArray[globalParentEdge];
                if (currentMaxCost + currentWeight < INT_MAX)
                    currCost = currentMaxCost + currentWeight;
                else
                    currCost = INT_MAX;
                // shortestParentEdgeArray[i] is 1 if edge i (according to the inverseEdgeArray numbering scheme) is a shortest parent, otherwise 0.)
                if (currCost==maxCostArray[globalChild] && maxCostArray[globalChild] != INT_MAX && currentMaxCost != INT_MAX && currentWeight != INT_MAX && currCost != INT_MAX) {
                    minCost = currCost;
                    shortestParentEdgeArray[edge] = 1;
                }
                else {
                    shortestParentEdgeArray[edge] = 0;
                }
            }
            // If this is a max node...
            else {
                // ...return all parents.
                if (maxCostArray[globalChild] != INT_MAX && maxCostArray[globalParent]!= INT_MAX) {
                    shortestParentEdgeArray[edge] = 1;
                }
                else  {
                    shortestParentEdgeArray[edge] = 0;
                }
            }
        }
        
    }
}

__kernel void INIT_INV_VERTEX_DIFF(__global int *inverseVertexDiffArray, __global int *inverseEdgeIncrTrackerArray)
{
    //Because we only enqueue graph->vertexCount work items, localVertex = global_id
    int localVertex = get_global_id(0);
    inverseVertexDiffArray[localVertex] = 0;
    inverseEdgeIncrTrackerArray[localVertex] = 0;
}


__kernel void INV_VERTEX_DIFF(__global int *edgeArray, __global int *inverseVertexDiffArray)
{
    int localEdge = get_global_id(0);
    
    int childVertex = edgeArray[localEdge];
    atomic_inc(&inverseVertexDiffArray[childVertex]);
}


__kernel void INV_EDGE_ARRAY(int vertexCount,
                             int edgeCount,
                             __global int *vertexArray,
                             __global int *edgeArray,
                             __global int *weightArray,
                             __global int *inverseVertexArray,
                             __global int *inverseEdgeArray,
                             __global int *inverseWeightArray,
                             __global int *inverseEdgeIncrTrackerArray)
{
    int localParent = get_global_id(0);
    //int localParent = globalParent % vertexCount;
    //int iGraph = globalParent / vertexCount;
    
    int edgeStart = vertexArray[localParent];
    int edgeEnd = getEdgeEnd(localParent, vertexCount, vertexArray, edgeCount);
    // Iterate over the edges
    for(int localEdge = edgeStart; localEdge < edgeEnd; localEdge++)
    {
        //int globalEdge = iGraph * edgeCount + localEdge;
        int localChild = edgeArray[localEdge];
        //int globalChild = iGraph * vertexCount + localChild;
        // Increment the tracker variable, which determines the position of the parent id in the inverseEdgeArray
        int offset = atomic_inc(&inverseEdgeIncrTrackerArray[localChild]);
        //Determine parent's own ID's proper location of inverseEdgeArray
        int localLocation = inverseVertexArray[localChild] + offset;
        //int globalLocation = iGraph * edgeCount + localLocation;
        // write to that location
        
        
        // The problem here could be that the higher samples don't know what order the first sample's parents end up in. Thus, localLocation, and thus globalLocation, could be different between samples, so that the first parent in one sample becomes the second in another.
        
        //if (iGraph == 0) {
        inverseEdgeArray[localLocation] = localParent;
        //}
        //inverseWeightArray[globalLocation] = weightArray[globalEdge];
        printf("Setting inverseEdgeArray[%i] = %i from child %i to parent %i.\n", localLocation, inverseEdgeArray[localLocation], localChild, localParent);
    }
}

__kernel void INV_WEIGHT_ARRAY(int vertexCount,
                               int edgeCount,
                               __global int *vertexArray,
                               __global int *edgeArray,
                               __global int *weightArray,
                               __global int *inverseVertexArray,
                               __global int *inverseEdgeArray,
                               __global int *inverseWeightArray)
{
    int globalParent = get_global_id(0);
    int localParent = globalParent % vertexCount;
    int iGraph = globalParent / vertexCount;
    
    int edgeStart = vertexArray[localParent];
    int edgeEnd = getEdgeEnd(localParent, vertexCount, vertexArray, edgeCount);
    for(int localEdge = edgeStart; localEdge < edgeEnd; localEdge++)
    {
        int globalEdge = iGraph * edgeCount + localEdge;
        int localChild = edgeArray[localEdge];
        int globalChild = iGraph * vertexCount + localChild;
        int inverseEdgeStart = inverseVertexArray[localChild];
        int inverseEdgeEnd = getEdgeEnd(localChild, vertexCount, inverseVertexArray, edgeCount);
        for(int localInverseEdge = inverseEdgeStart; localInverseEdge < inverseEdgeEnd; localInverseEdge++) {
            int globalInverseEdge = iGraph * edgeCount + localInverseEdge;
            int localInverseParent = inverseEdgeArray[localInverseEdge];
            int globalnverseParent = iGraph * vertexCount + localInverseParent;
            if (globalParent == globalnverseParent) {
                inverseWeightArray[globalInverseEdge] = weightArray[globalEdge];
            }
            //            printf("Edge %i, from parent %i to child %i has weight %i. Corresponding invEdge, %i, has the weight %i\n", globalEdge, globalParent, globalChild, weightArray[globalEdge], globalInverseEdge, inverseWeightArray[globalInverseEdge]);
        }
    }
}




///
/// Kernel to initialize buffers
///
__kernel void initializeBuffers(__global int *maskArray,
                                __global int *maxCostArray,
                                __global int *maxUpdatingCostArray,
                                __global int *sumCostArray,
                                __global int *sumUpdatingCostArray,
                                int vertexCount,
                                int sourceCount,
                                __global int *sourceArray,
                                __global int *influentialParentArray)
{
    // access thread id
    int tid = get_global_id(0);
    int iGraph = tid / vertexCount;
    int localTid = tid % vertexCount;
    
    influentialParentArray[tid] = -1;
    
    if (sourceArray[tid] == 1) {
        maskArray[tid] = 1;
        maxCostArray[tid] = 0;
        maxUpdatingCostArray[tid] = 0;
        sumCostArray[tid] = 0;
        sumUpdatingCostArray[tid] = 0;
    }
    else {
        maskArray[tid] = 0;
        maxCostArray[tid] = INT_MAX;
        maxUpdatingCostArray[tid] = INT_MAX;
        sumCostArray[tid] = INT_MAX;
        sumUpdatingCostArray[tid] = INT_MAX;
    }
}
