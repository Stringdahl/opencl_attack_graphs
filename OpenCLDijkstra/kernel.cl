///
/// This is part 1 of the Kernel from Algorithm 4 in the paper
///
__kernel void OCL_SSSP_KERNEL1(__global int *vertexArray, __global int *edgeArray, __global float *weightArray,
                                __global int *maskArray, __global float *costArray, __global float *updatingCostArray,
                                int vertexCount, int edgeCount)
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
            // traversedEdge[edge] = 1;
            
            int nid = iGraph*vertexCount + edgeArray[edge];
            int eid = iGraph*edgeCount + edge;
            
            // One note here: whereas the paper specified weightArray[nid], I
            //  found that the correct thing to do was weightArray[edge].  I think
            //  this was a typo in the paper.  Either that, or I misunderstood
            //  the data structure.
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
                                int sourceVertex,
                                int vertexCount )
{
    // access thread id
    int tid = get_global_id(0);
    
    
    if (sourceVertex == tid)
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
