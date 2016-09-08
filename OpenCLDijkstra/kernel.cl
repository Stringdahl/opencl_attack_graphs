///
/// This is part 1 of the Kernel from Algorithm 4 in the paper
///
__kernel void OCL_SSSP_KERNEL1(__global int *vertexArray, __global int *edgeArray, __global float *weightArray, __global int *maskArray, __global float *costArray, __global float *updatingCostArray, int vertexCount, int edgeCount, __global int *traversedEdgeCountArray)
{
    // access thread id
    int tid = get_global_id(0);
    
    int iGraph = tid / vertexCount;
    int localTid = tid % vertexCount;
    
    if ( maskArray[tid] != 0 )
    {
        maskArray[tid] = 0;

        
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
        
        for(int edge = edgeStart; edge < edgeEnd; edge++)
        {
            
            int nid = iGraph*vertexCount + edgeArray[edge];
            int eid = iGraph*edgeCount + edge;
            
            traversedEdgeCountArray[eid] ++;

            if (updatingCostArray[nid] > (costArray[tid] + weightArray[eid]))
            {
                updatingCostArray[nid] = (costArray[tid] + weightArray[eid]);
            }
        }
        
    }
}

///
/// This is part 2 of the Kernel from Algorithm 5 in the paper.
///
__kernel void OCL_SSSP_KERNEL2(__global int *vertexArray, __global int *edgeArray, __global float *weightArray,
                                __global int *maskArray, __global float *costArray, __global float *updatingCostArray,
                                int vertexCount)
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
