__kernel void square(
                     __global int *vertexArray,
                     __global int *edgeArray,
                     __global float *weightArray,
                     __global int *maskArray,
                     __global float *costArray,
                     __global float *updatingCostArray,
                     const int vertexCount,
                     const int edgeCount)
{
    // access thread id
    int tid = get_global_id(0);
    
    if ( maskArray[tid] != 0 )
    {
        maskArray[tid] = 0;
        
        int edgeStart = vertexArray[tid];
        int edgeEnd;
        if (tid + 1 < (vertexCount))
        {
            edgeEnd = vertexArray[tid + 1];
        }
        else
        {
            edgeEnd = edgeCount;
        }
        
        for(int edge = edgeStart; edge < edgeEnd; edge++)
        {
            int nid = edgeArray[edge];
            
            // One note here: whereas the paper specified weightArray[nid], I
            //  found that the correct thing to do was weightArray[edge].  I think
            //  this was a typo in the paper.  Either that, or I misunderstood
            //  the data structure.
            if (updatingCostArray[nid] > (costArray[tid] + weightArray[edge]))
            {
                updatingCostArray[nid] = (costArray[tid] + weightArray[edge]);
            }
        }
    }
}

///
/// Kernel to initialize buffers
///
__kernel void initializeBuffers( __global int *maskArray, __global float *costArray, __global float *updatingCostArray,
                                int sourceVertex, int vertexCount )
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

