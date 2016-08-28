__kernel void square(
                     __global int *vertexArray,
                     __global int *edgeArray,
                     __global float *weightArray,
                     __global int *maskArray,
                     __global float *costArray,
                     __global float *updatingCostArray,
                     const int vertexCount,
                     const int edgeCount,
                     __global float* input,
                     __global float* output,
                     const unsigned int count)
{
    int i = get_global_id(0);
    if(i < count)
        output[i] = input[i] * input[i];
}
