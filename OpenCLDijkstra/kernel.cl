#define COST_MAX 2147482646


int getEdgeEnd(int iVertex, int vertexCount, __global int *vertexArray, int edgeCount) {
    if (iVertex + 1 < (vertexCount))
        return vertexArray[iVertex + 1];
    else
        return edgeCount;
}

///
/// This is part 1 of the Kernel from Algorithm 4 in the paper
///
__kernel void OCL_SSSP_KERNEL1(__global int *vertexArray, __global int *inverseVertexArray, __global int *edgeArray, __global int *inverseEdgeArray, __global int *weightArray, __global int *inverseWeightArray, __global int *maskArray, __global int *maxCostArray, __global int *maxUpdatingCostArray, int vertexCount, int edgeCount, __global int *traversedEdgeCountArray, __global int *parentCountArray, __global int *maxVertexArray)
{
    // access thread id
    int globalSource = get_global_id(0);
    
    int iGraph = globalSource / vertexCount;
    int localSource = globalSource % vertexCount;
    
    //printf("Start of Kernel 1: globalSource = %i, iGraph = %i, vertexCount = %i, localSource = %i, maskArray[%i] = %i, maxUpdatingCostArray[%i] = %i, maxCostArray[%i] = %i.\n", globalSource, iGraph, vertexCount, localSource, globalSource, maskArray[globalSource], globalSource, maxUpdatingCostArray[globalSource], globalSource, maxCostArray[globalSource]);
    // Only consider vertices that are marked for update
    if ( maskArray[globalSource] != 0 ) {
        // After attempting to update, don't do it again unless (i) a parent updated this, or (ii) recalculation is required due to kernel 2.
        maskArray[globalSource] = 0;
        // Only update if (i) this is a min node, or (ii) this is a max node and all parents have been visited.
        //printf("Check max and parents: globalSource = %i, maxVertexArray[%i] = %i, parentCountArray[%i] = %i\n", globalSource, globalSource, maxVertexArray[globalSource], globalSource, parentCountArray[globalSource]);
        if (maxVertexArray[globalSource]<0 || parentCountArray[globalSource]==0) {
            {
                // Get the edges
                int edgeStart = vertexArray[localSource];
                int edgeEnd = getEdgeEnd(localSource, vertexCount, vertexArray, edgeCount);
                
                //printf("globalSource = %i, localSource = %i, edgeStart = %i, edgeEnd = %i.\n", globalSource, localSource, edgeStart, edgeEnd);
                // Iterate over the edges
                for(int localEdge = edgeStart; localEdge < edgeEnd; localEdge++)
                {
                    // nid is the (globally indexed) target node
                    int localTarget = edgeArray[localEdge];
                    int globalTarget = iGraph*vertexCount + edgeArray[localEdge];
                    // eid is the globally indexed edge
                    int globalEdge = iGraph*edgeCount + localEdge;
                    
                    // If this edge has never been traversed, reduce the remaining parents of the target by one, so that they reach zero when all incoming edges have been visited.
                    if (traversedEdgeCountArray[globalEdge] == 0) {
                       atomic_dec(&parentCountArray[globalTarget]);
                       // parentCountArray[globalTarget]--;
                    }
                    // Mark that this edge has been traversed.
                    traversedEdgeCountArray[globalEdge] ++;
                    int inverseEdgeStart = inverseVertexArray[localTarget];
                    int inverseEdgeEnd = getEdgeEnd(localTarget, vertexCount, inverseVertexArray, edgeCount);
                    //printf("Before min/max: globalSource = %i, globalTarget = %i, maxVertexArray[%i] = %i, parentCountArray[%i] = %i.\n", globalSource, globalTarget, globalTarget, maxVertexArray[globalTarget], globalTarget, parentCountArray[globalTarget]);
                    // If this is a min node ...
                    if (maxVertexArray[globalTarget]<0) {
                        int currentCost;
//                        if (maxCostArray[globalSource] + weightArray[globalEdge] >= COST_MAX)
//                            currentCost = COST_MAX;
//                        else
                            currentCost = maxCostArray[globalSource] + weightArray[globalEdge];
                        
                        // ...atomically choose the lesser of the current and candidate updatingCost
                        atomic_min(&maxUpdatingCostArray[globalTarget], currentCost);
                        // Reconvert the integer representation to float and store in maxUpdatingCostArray
                        // Iterate over the edges
                        int minEdgeVal = COST_MAX;
                        
                        //maxUpdatingCostArray[globalTarget] = minEdgeVal;
                        // Mark the target for update
                        //maskArray[nid] = 1;
                        
                    }
                    
                    // If this is a max node...
                    else {
                        if (parentCountArray[globalTarget]==0) {
                            // If all parents have been visited ...
                            // Iterate over the edges
                            int maxEdgeVal = 0;
                            
                            for(int localInverseEdge = inverseEdgeStart; localInverseEdge < inverseEdgeEnd; localInverseEdge++) {
                                int localInverseTarget = inverseEdgeArray[localInverseEdge];
                                int globalInverseTarget = iGraph*vertexCount + localInverseTarget;
                                int globalInverseEdge = iGraph*edgeCount + localInverseEdge;
                                int currEdgeVal = maxCostArray[globalInverseTarget] + inverseWeightArray[globalInverseEdge];
                               if (currEdgeVal>maxEdgeVal) {
                                    maxEdgeVal = currEdgeVal;
                                }
                               // printf("In max: globalSource = %i, globalTarget = %i, currEdgeVal = %i, minEdgeVal = %i.\n", globalSource, globalTarget, currEdgeVal, maxEdgeVal);
                            }
//                            int currentCost;
//                            if (maxCostArray[globalSource] + weightArray[globalEdge] >= COST_MAX)
//                                currentCost = COST_MAX;
//                            else
//                                currentCost = maxCostArray[globalSource] + weightArray[globalEdge];
//                            
//                            atomic_max(&maxUpdatingCostArray[globalTarget], currentCost);
//                            atomic_max(&maxCostArray[globalTarget], currentCost);

                            maxCostArray[globalTarget] = maxEdgeVal;
                            maxUpdatingCostArray[globalTarget] = maxEdgeVal;
                            // Mark the target for update
                            maskArray[globalTarget] = 1;
                        }
                    }
                }
            }
        }
    }
}


///
/// This is part 2 of the Kernel from Algorithm 5 in the paper.
///
__kernel void OCL_SSSP_KERNEL2(__global int *vertexArray, __global int *edgeArray, __global int *weightArray,
                               __global int *maskArray, __global int *maxCostArray, __global int *maxUpdatingCostArray,
                               int vertexCount, __global int *maxVertexArray)
{
    // access thread id
    int tid = get_global_id(0);
    
    if (maxCostArray[tid] > maxUpdatingCostArray[tid])
    {
        maxCostArray[tid] = maxUpdatingCostArray[tid];
        maskArray[tid] = 1;
    }
    
    maxUpdatingCostArray[tid] = maxCostArray[tid];
}



///
/// Kernel to initialize buffers
///
__kernel void initializeBuffers( __global int *maskArray,
                                __global int *maxCostArray,
                                __global int *maxUpdatingCostArray,
                                int vertexCount,
                                __global int *sourceArray)
{
    // access thread id
    int tid = get_global_id(0);
    int iGraph = tid / vertexCount;
    int localTid = tid % vertexCount;


    if (localTid == sourceArray[iGraph])
    {
        maskArray[tid] = 1;
        maxCostArray[tid] = 0;
        maxUpdatingCostArray[tid] = 0;
    }
    else
    {
        maskArray[tid] = 0;
        maxCostArray[tid] = COST_MAX;
        maxUpdatingCostArray[tid] = COST_MAX;
    }
}



