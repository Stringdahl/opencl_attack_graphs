///
/// This is part 1 of the Kernel from Algorithm 4 in the paper
///
__kernel void OCL_SSSP_KERNEL1(__global int *vertexArray, __global int *inverseVertexArray, __global int *edgeArray, __global int *inverseEdgeArray, __global float *weightArray, __global float *inverseWeightArray, __global float *aggregatedWeightArray, __global int *maskArray, __global float *costArray, __global float *updatingCostArray, int vertexCount, int edgeCount, __global int *traversedEdgeCountArray, __global int *parentCountArray, __global float *maxVertexArray)
{
    // access thread id
    int tid = get_global_id(0);
    
    int iGraph = tid / vertexCount;
    int localTid = tid % vertexCount;
    
    // Only consider vertices that are marked for update
    if ( maskArray[tid] != 0 ) {
        // After attempting to update, don't do it again unless (i) a parent updated this, or (ii) recalculation is required due to kernel 2.
        maskArray[tid] = 0;
        // Only update if (i) this is a min node, or (ii) this is a max node and all parents have been visited.
        if (maxVertexArray[tid]<0 || parentCountArray[tid]==0) {
            {
                // Get the edges
                int edgeStart = vertexArray[localTid];
                int edgeEnd;
                if (localTid + 1 < (vertexCount))
                {
                    edgeEnd = vertexArray[localTid + 1];
                }
                else
                {
                    edgeEnd = edgeCount;
                }
                // Iterate over the edges
                for(int edge = edgeStart; edge < edgeEnd; edge++)
                {
                    // nid is the (globally indexed) target node
                    int nid = iGraph*vertexCount + edgeArray[edge];
                    // eid is the globbally indexed edge
                    int eid = iGraph*edgeCount + edge;
                    
                    // If this edge has never been traversed, reduce the remaining parents of the target by one, so that they reach zero when all incoming edges have been visited.
                    if (traversedEdgeCountArray[eid] == 0) {
                        parentCountArray[nid]--;
                    }
                    // Mark that this edge has been traversed.
                    traversedEdgeCountArray[eid] ++;
                    
                    // If this is a min node and the incoming coming cost is lower than the previously seen...
                    if (maxVertexArray[nid]<0 && (updatingCostArray[nid] > (costArray[tid] + weightArray[eid])))
                    {
                        // ... then update the cost of the temporary variable updatingCost.
                        updatingCostArray[nid] = (costArray[tid] + weightArray[eid]);
                    }
                    
                    // If this is a max node...
                    if (maxVertexArray[nid]>=0)
                    {
                        // ... and the incoming cost is higher than the previously seen...
                        if (maxVertexArray[nid] < (costArray[tid] + weightArray[eid])) {
                            // ... then update the maximum cost to the incoming.
                            maxVertexArray[nid] = (costArray[tid] + weightArray[eid]);
                        }
                        // If all parents have been visited ...
                        if (parentCountArray[nid]==0) {
                            // ... update the cost to the highest of all encountered.
                            updatingCostArray[nid] = maxVertexArray[nid];
                            // Get the edges
                            int inverseEdgeStart = inverseVertexArray[nid];
                            int inverseEdgeEnd;
                            if (nid + 1 < (vertexCount))
                            {
                                inverseEdgeEnd = inverseVertexArray[nid + 1];
                            }
                            else
                            {
                                inverseEdgeEnd = edgeCount;
                            }
                            // Iterate over the edges
                            float maxEdgeVal = 0;
                            for(int inverseEdge = inverseEdgeStart; inverseEdge < inverseEdgeEnd; inverseEdge++) {
                                float currEdgeVal = costArray[inverseEdgeArray[inverseEdge]] + inverseWeightArray[inverseEdge];
                                if (currEdgeVal>maxEdgeVal) {
                                    maxEdgeVal = currEdgeVal;
                                }
                            }
                            
                            costArray[nid] = maxEdgeVal;
                            // Mark the target for update
                            maskArray[nid] = 1;
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
__kernel void OCL_SSSP_KERNEL2(__global int *vertexArray, __global int *edgeArray, __global float *weightArray,
                               __global int *maskArray, __global float *costArray, __global float *updatingCostArray,
                               int vertexCount, __global float *maxVertexArray)
{
    // access thread id
    int tid = get_global_id(0);
    
    if (costArray[tid] > updatingCostArray[tid])
    {
        costArray[tid] = updatingCostArray[tid];
        maskArray[tid] = 1;
    }
    
    updatingCostArray[tid] = costArray[tid];
}



///
/// Kernel to initialize buffers
///
__kernel void initializeBuffers( __global int *maskArray,
                                __global float *costArray,
                                __global float *updatingCostArray,
                                int vertexCount,
                                int nSourceVertices,
                                __global int *sourceArray)
{
    // access thread id
    int tid = get_global_id(0);
    bool isSource = false;
    
    for (int i = 0; i < nSourceVertices; i++) {
        if (sourceArray[i] == tid) {
            isSource = true;
        }
    }
    if (isSource)
    {
        maskArray[tid] = 1;
        costArray[tid] = 0.0;
        updatingCostArray[tid] = 0.0;
    }
    else
    {
        maskArray[tid] = 0;
        costArray[tid] = FLT_MAX;
        updatingCostArray[tid] = FLT_MAX;
    }
}
