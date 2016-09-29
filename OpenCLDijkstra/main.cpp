#include <fcntl.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <pthread.h>
#include<time.h>
#include "graph.hpp"
#include "utility.hpp"

#define __CL_ENABLE_EXCEPTIONS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.hpp>
#endif


#define kernelPath "/Users/pontus/Documents/Pontus Program Files/XCode/OpenCLDijkstra/OpenCLDijkstra/kernel.cl"
#define checkError(a, b) checkErrorFileLine(a, b, __FILE__ , __LINE__)
#define NUM_ASYNCHRONOUS_ITERATIONS 20  // Number of async loop iterations before attempting to read results back

///
//  Utility functions adapted from NVIDIA GPU Computing SDK
//
void checkErrorFileLine(int errNum, int expected, const char* file, const int lineNumber);
cl_device_id getFirstDev(cl_context cxGPUContext);

///
//  Namespaces
//
using namespace std;

///
//  Globals
//
pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;



///
///  Allocate memory for input CUDA buffers and copy the data into device memory
///
void allocateOCLBuffers(cl_context gpuContext, cl_command_queue commandQueue, GraphData *graph, cl_mem *vertexArrayDevice, cl_mem *inverseVertexArrayDevice, cl_mem *edgeArrayDevice, cl_mem *inverseEdgeArrayDevice, cl_mem *weightArrayDevice, cl_mem *inverseWeightArrayDevice, cl_mem *maskArrayDevice, cl_mem *maxCostArrayDevice, cl_mem *maxUpdatingCostArrayDevice, cl_mem *sumCostArrayDevice, cl_mem *sumUpdatingCostArrayDevice, cl_mem *traversedEdgeArrayDevice, cl_mem *sourceArrayDevice, cl_mem *parentCountArrayDevice, cl_mem *maxVertexArrayDevice, cl_mem *shortestParentsArrayDevice)
{
    cl_int errNum;
    cl_mem hostVertexArrayBuffer;
    cl_mem hostInverseVertexArrayBuffer;
    cl_mem hostEdgeArrayBuffer;
    cl_mem hostInverseEdgeArrayBuffer;
    cl_mem hostWeightArrayBuffer;
    cl_mem hostInverseWeightArrayBuffer;
    cl_mem hostAggregatedWeightArrayBuffer;
    cl_mem hostTraversedEdgeCountArrayBuffer;
    cl_mem hostSourceArrayBuffer;
    cl_mem hostParentCountArrayBuffer;
    cl_mem hostMaxVertexArrayBuffer;
    int totalVertexCount = graph->graphCount * graph->vertexCount;
    int totalEdgeCount = graph->graphCount * graph->edgeCount;
    
    
    // Initially, no edges have been travelled
    int *traversedEdgeCountArray = (int*)malloc(totalEdgeCount * sizeof(int));
    for (int iEdge=0; iEdge<totalEdgeCount; iEdge++) {
        traversedEdgeCountArray[iEdge]=0;
    }
    
    int *aggregatedWeightArray = (int*)malloc(totalEdgeCount * sizeof(int));
    for (int iEdge=0; iEdge<totalEdgeCount; iEdge++) {
        aggregatedWeightArray[iEdge]=0;
    }
    
    int *parentCountArray = (int*)malloc(totalVertexCount * sizeof(int));
    for (int iVertex=0; iVertex<totalVertexCount; iVertex++) {
        parentCountArray[iVertex]=graph->parentCountArray[iVertex % graph->vertexCount];
    }
    
    int *maxVertexArray = (int*)malloc(totalVertexCount * sizeof(int));
    for (int iGraph=0; iGraph<graph->graphCount; iGraph++) {
        for (int iVertex=0; iVertex<graph->vertexCount; iVertex++) {
            maxVertexArray[iGraph*graph->vertexCount + iVertex]=graph->maxVertexArray[iVertex];
        }
    }
    
    
    // First, need to create OpenCL Host buffers that can be copied to device buffers
    hostVertexArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                           sizeof(int) * graph->vertexCount, graph->vertexArray, &errNum);
    checkError(errNum, CL_SUCCESS);
    hostInverseVertexArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                                  sizeof(int) * graph->vertexCount, graph->inverseVertexArray, &errNum);
    checkError(errNum, CL_SUCCESS);
    hostEdgeArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                         sizeof(int) * graph->edgeCount, graph->edgeArray, &errNum);
    checkError(errNum, CL_SUCCESS);
    hostInverseEdgeArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                                sizeof(int) * graph->edgeCount, graph->inverseEdgeArray, &errNum);
    checkError(errNum, CL_SUCCESS);
    hostWeightArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                           sizeof(int) * totalEdgeCount, graph->weightArray, &errNum);
    
    checkError(errNum, CL_SUCCESS);
    hostInverseWeightArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                                  sizeof(int) * totalEdgeCount, graph->inverseWeightArray, &errNum);
    checkError(errNum, CL_SUCCESS);
    hostAggregatedWeightArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                                     sizeof(int) * totalEdgeCount, aggregatedWeightArray, &errNum);
    checkError(errNum, CL_SUCCESS);
    hostTraversedEdgeCountArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                                       sizeof(int) * totalEdgeCount, traversedEdgeCountArray, &errNum);
    checkError(errNum, CL_SUCCESS);
    hostSourceArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                           sizeof(int) * totalVertexCount, graph->sourceArray, &errNum);
    
    checkError(errNum, CL_SUCCESS);
    hostParentCountArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                                sizeof(int) * totalVertexCount, parentCountArray, &errNum);
    checkError(errNum, CL_SUCCESS);
    hostMaxVertexArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                              sizeof(int) * totalVertexCount, maxVertexArray, &errNum);
    checkError(errNum, CL_SUCCESS);
    
    // Now create all of the GPU buffers
    *vertexArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(int) * graph->vertexCount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *inverseVertexArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(int) * graph->vertexCount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *edgeArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(int) * graph->edgeCount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *inverseEdgeArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(int) * graph->edgeCount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *weightArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(int) * totalEdgeCount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *inverseWeightArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(int) * totalEdgeCount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *maskArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(int) * totalVertexCount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *maxCostArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(int) * totalVertexCount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *maxUpdatingCostArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(int) * totalVertexCount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *sumCostArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(int) * totalVertexCount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *sumUpdatingCostArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(int) * totalVertexCount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *parentCountArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(int) * totalVertexCount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *maxVertexArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(int) * totalVertexCount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *traversedEdgeArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(int) * totalEdgeCount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *sourceArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(int) * totalVertexCount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *shortestParentsArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(int) * totalEdgeCount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    
    
    //
    
    // Now queue up the data to be copied to the device
    errNum = clEnqueueCopyBuffer(commandQueue, hostVertexArrayBuffer, *vertexArrayDevice, 0, 0,
                                 sizeof(int) * graph->vertexCount, 0, NULL, NULL);
    checkError(errNum, CL_SUCCESS);
    
    errNum = clEnqueueCopyBuffer(commandQueue, hostInverseVertexArrayBuffer, *inverseVertexArrayDevice, 0, 0,
                                 sizeof(int) * graph->vertexCount, 0, NULL, NULL);
    checkError(errNum, CL_SUCCESS);
    
    errNum = clEnqueueCopyBuffer(commandQueue, hostEdgeArrayBuffer, *edgeArrayDevice, 0, 0,
                                 sizeof(int) * graph->edgeCount, 0, NULL, NULL);
    checkError(errNum, CL_SUCCESS);
    
    errNum = clEnqueueCopyBuffer(commandQueue, hostInverseEdgeArrayBuffer, *inverseEdgeArrayDevice, 0, 0,
                                 sizeof(int) * graph->edgeCount, 0, NULL, NULL);
    checkError(errNum, CL_SUCCESS);
    
    errNum = clEnqueueCopyBuffer(commandQueue, hostWeightArrayBuffer, *weightArrayDevice, 0, 0,
                                 sizeof(int) * totalEdgeCount, 0, NULL, NULL);
    checkError(errNum, CL_SUCCESS);
    
    errNum = clEnqueueCopyBuffer(commandQueue, hostInverseWeightArrayBuffer, *inverseWeightArrayDevice, 0, 0,
                                 sizeof(int) * totalEdgeCount, 0, NULL, NULL);
    checkError(errNum, CL_SUCCESS);
    
    errNum = clEnqueueCopyBuffer(commandQueue, hostParentCountArrayBuffer, *parentCountArrayDevice, 0, 0,
                                 sizeof(int) * totalVertexCount, 0, NULL, NULL);
    checkError(errNum, CL_SUCCESS);
    
    errNum = clEnqueueCopyBuffer(commandQueue, hostMaxVertexArrayBuffer, *maxVertexArrayDevice, 0, 0,
                                 sizeof(int) * totalVertexCount, 0, NULL, NULL);
    checkError(errNum, CL_SUCCESS);
    
    errNum = clEnqueueCopyBuffer(commandQueue, hostTraversedEdgeCountArrayBuffer, *traversedEdgeArrayDevice, 0, 0,
                                 sizeof(int) * graph->edgeCount, 0, NULL, NULL);
    checkError(errNum, CL_SUCCESS);
    
    errNum = clEnqueueCopyBuffer(commandQueue, hostSourceArrayBuffer, *sourceArrayDevice, 0, 0,
                                 sizeof(int) * totalVertexCount, 0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        printf("Error: Failed to enqueue buffer!\n");
        exit(1);
    }
    checkError(errNum, CL_SUCCESS);
    
    free(traversedEdgeCountArray);
    free(aggregatedWeightArray);
    free(parentCountArray);
    free(maxVertexArray);
    clReleaseMemObject(hostVertexArrayBuffer);
    clReleaseMemObject(hostInverseVertexArrayBuffer);
    clReleaseMemObject(hostEdgeArrayBuffer);
    clReleaseMemObject(hostInverseEdgeArrayBuffer);
    clReleaseMemObject(hostWeightArrayBuffer);
    clReleaseMemObject(hostInverseWeightArrayBuffer);
    clReleaseMemObject(hostAggregatedWeightArrayBuffer);
    clReleaseMemObject(hostTraversedEdgeCountArrayBuffer);
    clReleaseMemObject(hostParentCountArrayBuffer);
    clReleaseMemObject(hostMaxVertexArrayBuffer);
    clReleaseMemObject(hostSourceArrayBuffer);
}


///
/// Load and build an OpenCL program from source file
/// \param gpuContext GPU context on which to load and build the program
/// \param fileName File name of source file that holds the kernels
/// \return Handle to the program
///
cl_program loadAndBuildProgram( cl_context gpuContext, const char *fileName )
{
    pthread_mutex_lock(&mutex1);
    
    cl_int errNum;
    cl_program program;
    
    // Load the OpenCL source code from the .cl file
    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }
    
    std::ostringstream oss;
    oss << kernelFile.rdbuf();
    
    std::string srcStdStr = oss.str();
    const char *source = srcStdStr.c_str();
    
    checkError(source != NULL, true);
    
    // Create the program for all GPUs in the context
    program = clCreateProgramWithSource(gpuContext, 1, (const char **)&source, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    // build the program for all devices on the context
    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        char cBuildLog[20240];
        clGetProgramBuildInfo(program, getFirstDev(gpuContext), CL_PROGRAM_BUILD_LOG,
                              sizeof(cBuildLog), cBuildLog, NULL );
        
        cerr << cBuildLog << endl;
        printf("%i\n\n", errNum);
        checkError(errNum, CL_SUCCESS);
    }
    
    pthread_mutex_unlock(&mutex1);
    return program;
}

///
/// Gets the id of the first device from the context (from the NVIDIA SDK)
///
cl_device_id getFirstDev(cl_context cxGPUContext)
{
    size_t szParmDataBytes;
    cl_device_id* cdDevices;
    
    // get the list of GPU devices associated with context
    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
    cdDevices = (cl_device_id*) malloc(szParmDataBytes);
    
    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);
    
    cl_device_id first = cdDevices[0];
    free(cdDevices);
    
    return first;
}


///
/// Check whether the mask array is empty.  This tells the algorithm whether
/// it needs to continue running or not.
///
bool maskArrayEmpty(int *maskArray, int count)
{
    for(int i = 0; i < count; i++ )
    {
        if (maskArray[i] == 1)
        {
            return false;
        }
    }
    
    return true;
}


int  initializeComputing(cl_device_id *device_id, cl_context *context, cl_command_queue *commands, cl_program *program) {
    // Connect to a compute device
    //
    int gpu = 1;
    int err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
    
    // Create a compute context
    //
    *context = clCreateContext(0, 1, device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }
    
    // Create a command commands
    //
    *commands = clCreateCommandQueue(*context, *device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }
    
    // Create the compute program from the source file
    *program = loadAndBuildProgram(*context, kernelPath);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }
    
    // Build the program executable
    //
    err = clBuildProgram(*program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(*program, *device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }
    
    return err;
}

int createKernels(cl_kernel *initializeKernel, cl_kernel *ssspKernel1, cl_kernel *ssspKernel2, cl_kernel *shortestParentsKernel, cl_program *program) {
    
    int errNum;
    
    // Create the compute kernel in the program we wish to run
    *initializeKernel = clCreateKernel(*program, "initializeBuffers", &errNum);
    if (!initializeKernel || errNum != CL_SUCCESS)
    {
        printf("Error: Failed to create initializeKernel initializeBuffers!\n");
        exit(1);
    }
    
    // Kernel 1
    *ssspKernel1 = clCreateKernel(*program, "OCL_SSSP_KERNEL1", &errNum);
    if (!ssspKernel1 || errNum != CL_SUCCESS)
    {
        printf("Error: Failed to create ssspKernel1 initializeBuffers!\n");
        exit(1);
    }
    
    // Kernel 2
    *ssspKernel2 = clCreateKernel(*program, "OCL_SSSP_KERNEL2", &errNum);
    if (!ssspKernel2 || errNum != CL_SUCCESS)
    {
        printf("Error: Failed to create ssspKernel2 initializeBuffers!\n");
        exit(1);
    }

    // Shortest parent kernel
    *shortestParentsKernel = clCreateKernel(*program, "SHORTEST_PARENTS", &errNum);
    if (!shortestParentsKernel || errNum != CL_SUCCESS)
    {
        printf("Error: Failed to create ssspKernel2 initializeBuffers!\n");
        exit(1);
    }
return errNum;
}


int setKernelArguments(cl_kernel *initializeKernel, cl_kernel *ssspKernel1, cl_kernel *ssspKernel2, cl_kernel *shortestParentsKernel, int graphCount, int vertexCount, int edgeCount, int sourceCount,  cl_mem *maskArrayDevice, cl_mem *vertexArrayDevice, cl_mem *inverseVertexArrayDevice, cl_mem *edgeArrayDevice, cl_mem *inverseEdgeArrayDevice, cl_mem *maxCostArrayDevice, cl_mem *maxUpdatingCostArrayDevice, cl_mem *sumCostArrayDevice, cl_mem *sumUpdatingCostArrayDevice, cl_mem *sourceArrayDevice, cl_mem *weightArrayDevice, cl_mem *inverseWeightArrayDevice, cl_mem *traversedEdgeCountArrayDevice, cl_mem *parentCountArrayDevice, cl_mem *maxVerticeArrayDevice, cl_mem *shortestParentsArrayDevice) {
    
    int totalVertexCount = graphCount*vertexCount;
    
    // Set the arguments to initializeKernel
    //
    int errNum = 0;
    errNum |= clSetKernelArg(*initializeKernel, 0, sizeof(cl_mem), maskArrayDevice);
    errNum |= clSetKernelArg(*initializeKernel, 1, sizeof(cl_mem), maxCostArrayDevice);
    errNum |= clSetKernelArg(*initializeKernel, 2, sizeof(cl_mem), maxUpdatingCostArrayDevice);
    errNum |= clSetKernelArg(*initializeKernel, 3, sizeof(cl_mem), sumCostArrayDevice);
    errNum |= clSetKernelArg(*initializeKernel, 4, sizeof(cl_mem), sumUpdatingCostArrayDevice);
    errNum |= clSetKernelArg(*initializeKernel, 5, sizeof(int), &vertexCount);
    errNum |= clSetKernelArg(*initializeKernel, 6, sizeof(int), &sourceCount);
    errNum |= clSetKernelArg(*initializeKernel, 7, sizeof(cl_mem), sourceArrayDevice);
    errNum |= clSetKernelArg(*initializeKernel, 8, sizeof(cl_mem), shortestParentsArrayDevice);
    
    // Set the arguments to ssspKernel1
    errNum |= clSetKernelArg(*ssspKernel1, 0, sizeof(cl_mem), vertexArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel1, 1, sizeof(cl_mem), inverseVertexArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel1, 2, sizeof(cl_mem), edgeArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel1, 3, sizeof(cl_mem), inverseEdgeArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel1, 4, sizeof(cl_mem), weightArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel1, 5, sizeof(cl_mem), inverseWeightArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel1, 6, sizeof(cl_mem), maskArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel1, 7, sizeof(cl_mem), maxCostArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel1, 8, sizeof(cl_mem), maxUpdatingCostArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel1, 9, sizeof(cl_mem), sumCostArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel1, 10, sizeof(cl_mem), sumUpdatingCostArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel1, 11, sizeof(int), &vertexCount);
    errNum |= clSetKernelArg(*ssspKernel1, 12, sizeof(int), &edgeCount);
    errNum |= clSetKernelArg(*ssspKernel1, 13, sizeof(cl_mem), traversedEdgeCountArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel1, 14, sizeof(cl_mem), parentCountArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel1, 15, sizeof(cl_mem), maxVerticeArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel1, 16, sizeof(cl_mem), shortestParentsArrayDevice);
    
    // Set the arguments to ssspKernel2
    errNum |= clSetKernelArg(*ssspKernel2, 0, sizeof(cl_mem), vertexArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel2, 1, sizeof(cl_mem), edgeArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel2, 2, sizeof(cl_mem), weightArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel2, 3, sizeof(cl_mem), maskArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel2, 4, sizeof(cl_mem), maxCostArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel2, 5, sizeof(cl_mem), maxUpdatingCostArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel2, 6, sizeof(cl_mem), sumCostArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel2, 7, sizeof(cl_mem), sumUpdatingCostArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel2, 8, sizeof(int), &totalVertexCount);
    errNum |= clSetKernelArg(*ssspKernel2, 9, sizeof(cl_mem), maxVerticeArrayDevice);

    // Set the arguments to shortestParentsKernel
    errNum |= clSetKernelArg(*shortestParentsKernel, 0, sizeof(int), &vertexCount);
    errNum |= clSetKernelArg(*shortestParentsKernel, 1, sizeof(int), &edgeCount);
    errNum |= clSetKernelArg(*shortestParentsKernel, 2, sizeof(cl_mem), vertexArrayDevice);
    errNum |= clSetKernelArg(*shortestParentsKernel, 3, sizeof(cl_mem), inverseVertexArrayDevice);
    errNum |= clSetKernelArg(*shortestParentsKernel, 4, sizeof(cl_mem), edgeArrayDevice);
    errNum |= clSetKernelArg(*shortestParentsKernel, 5, sizeof(cl_mem), inverseEdgeArrayDevice);
    errNum |= clSetKernelArg(*shortestParentsKernel, 6, sizeof(cl_mem), weightArrayDevice);
    errNum |= clSetKernelArg(*shortestParentsKernel, 7, sizeof(cl_mem), inverseWeightArrayDevice);
    errNum |= clSetKernelArg(*shortestParentsKernel, 8, sizeof(cl_mem), maxCostArrayDevice);
    errNum |= clSetKernelArg(*shortestParentsKernel, 9, sizeof(cl_mem), maxUpdatingCostArrayDevice);
    errNum |= clSetKernelArg(*shortestParentsKernel, 10, sizeof(cl_mem), maxVerticeArrayDevice);
    errNum |= clSetKernelArg(*shortestParentsKernel, 11, sizeof(cl_mem), shortestParentsArrayDevice);

    if (errNum != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", errNum);
    }
    return errNum;
}

void calculateGraphs(GraphData *graph, bool debug) {
    
    
    int errNum;                            // error code returned from api calls
    size_t global;                      // global domain size for our calculation
    
    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commandQueue;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel initializeKernel;                   // compute kernel
    cl_kernel ssspKernel1;
    cl_kernel ssspKernel2;
    cl_kernel shortestParentsKernel;
    cl_event readDone;
    cl_event kernel1event, kernel2event;
    
    cl_mem vertexArrayDevice;                       // device memory used for the input array
    cl_mem inverseVertexArrayDevice;                       // device memory used for the input array
    cl_mem edgeArrayDevice;                       // device memory used for the input array
    cl_mem inverseEdgeArrayDevice;                       // device memory used for the input array
    cl_mem weightArrayDevice;                       // device memory used for the input array
    cl_mem inverseWeightArrayDevice;                       // device memory used for the input array
    cl_mem maskArrayDevice;                       // device memory used for the input array
    cl_mem maxCostArrayDevice;                       // device memory used for the input array
    cl_mem maxUpdatingCostArrayDevice;                       // device memory used for the input array
    cl_mem sumCostArrayDevice;                       // device memory used for the input array
    cl_mem sumUpdatingCostArrayDevice;                       // device memory used for the input array
    cl_mem traversedEdgeCountArrayDevice;
    cl_mem sourceArrayDevice;
    cl_mem parentCountArrayDevice;
    cl_mem maxVerticeArrayDevice;
    cl_mem shortestParentsArrayDevice;
    
    
    int totalVertexCount = graph->graphCount * graph->vertexCount;
    int totalEdgeCount = graph->graphCount * graph->edgeCount;
    int *maskArrayHost = (int*) malloc(sizeof(int) * totalVertexCount);
    
    
    // Set up OpenCL computing environment, getting GPU device ID, command queue, context, and program
    initializeComputing(&device_id, &context, &commandQueue, &program);
    
    // Create kernels from the program (kernel.cl)
    createKernels(&initializeKernel, &ssspKernel1, &ssspKernel2, &shortestParentsKernel, &program);
    
    // Allocate buffers in Device memory
    allocateOCLBuffers(context, commandQueue, graph, &vertexArrayDevice, &inverseVertexArrayDevice, &edgeArrayDevice, &inverseEdgeArrayDevice, &weightArrayDevice, &inverseWeightArrayDevice, &maskArrayDevice, &maxCostArrayDevice, &maxUpdatingCostArrayDevice, &sumCostArrayDevice, &sumUpdatingCostArrayDevice, &traversedEdgeCountArrayDevice, &sourceArrayDevice, &parentCountArrayDevice, &maxVerticeArrayDevice, &shortestParentsArrayDevice);
    
    // Setting the kernel arguments
    errNum = setKernelArguments(&initializeKernel, &ssspKernel1, &ssspKernel2, &shortestParentsKernel, graph->graphCount, graph->vertexCount, graph->edgeCount, graph->sourceCount, &maskArrayDevice, &vertexArrayDevice, &inverseVertexArrayDevice, &edgeArrayDevice, &inverseEdgeArrayDevice, &maxCostArrayDevice, &maxUpdatingCostArrayDevice, &sumCostArrayDevice, &sumUpdatingCostArrayDevice, &sourceArrayDevice, &weightArrayDevice, &inverseWeightArrayDevice, &traversedEdgeCountArrayDevice, &parentCountArrayDevice, &maxVerticeArrayDevice, &shortestParentsArrayDevice);
    
    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global = totalVertexCount;
    
    errNum = clEnqueueNDRangeKernel(commandQueue, initializeKernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    checkError(errNum, CL_SUCCESS);
    
    
    
    errNum = clEnqueueReadBuffer( commandQueue, maskArrayDevice, CL_FALSE, 0, sizeof(int) * totalVertexCount, maskArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    
    clWaitForEvents(1, &readDone);
    
    int count = 0;
    while(!maskArrayEmpty(maskArrayHost, totalVertexCount))
    {
        
        // In order to improve performance, we run some number of iterations
        // without reading the results.  This might result in running more iterations
        // than necessary at times, but it will in most cases be faster because
        // we are doing less stalling of the GPU waiting for results.
        
        for(int asyncIter = 0; asyncIter < NUM_ASYNCHRONOUS_ITERATIONS; asyncIter++)
        {
            count ++;
            
            errNum = clEnqueueNDRangeKernel(commandQueue, ssspKernel1, 1, 0, &global, NULL, 0, NULL, &kernel1event);
            checkError(errNum, CL_SUCCESS);
            
            errNum = clEnqueueNDRangeKernel(commandQueue, ssspKernel2, 1, 0, &global, NULL, 0, NULL, &kernel2event);
            checkError(errNum, CL_SUCCESS);
            
        }
        
        errNum = clEnqueueReadBuffer(commandQueue, maskArrayDevice, CL_FALSE, 0, sizeof(int) * totalVertexCount, maskArrayHost, 0, NULL, &readDone);
        checkError(errNum, CL_SUCCESS);
        clWaitForEvents(1, &readDone);
        
    }
    // Wait for the command commands to get serviced before reading back results
    clFinish(commandQueue);
    
    // Read back the results from the device to verify the output
    
    errNum = clEnqueueReadBuffer( commandQueue, maxCostArrayDevice, CL_TRUE, 0, sizeof(int) * totalVertexCount, graph->costArray, 0, NULL, &readDone );
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer( commandQueue, sumCostArrayDevice, CL_TRUE, 0, sizeof(int) * totalVertexCount, graph->sumCostArray, 0, NULL, &readDone );
    checkError(errNum, CL_SUCCESS);
    clFinish(commandQueue);
    
    errNum = clEnqueueNDRangeKernel(commandQueue, shortestParentsKernel, 1, 0, &global, NULL, 0, NULL, NULL);
    checkError(errNum, CL_SUCCESS);
    clFinish(commandQueue);

    errNum = clEnqueueReadBuffer(commandQueue, shortestParentsArrayDevice, CL_FALSE, 0, sizeof(int) * totalEdgeCount, graph->shortestParentsArray, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    clFinish(commandQueue);


    
    // Shutdown and cleanup
    //
    free(maskArrayHost);
    
    clReleaseMemObject(vertexArrayDevice);
    clReleaseMemObject(inverseVertexArrayDevice);
    clReleaseMemObject(edgeArrayDevice);
    clReleaseMemObject(inverseEdgeArrayDevice);
    clReleaseMemObject(weightArrayDevice);
    clReleaseMemObject(inverseWeightArrayDevice);
    clReleaseMemObject(maskArrayDevice);
    clReleaseMemObject(maxCostArrayDevice);
    clReleaseMemObject(maxUpdatingCostArrayDevice);
    clReleaseMemObject(sumCostArrayDevice);
    clReleaseMemObject(sumUpdatingCostArrayDevice);
    clReleaseMemObject(traversedEdgeCountArrayDevice);
    clReleaseMemObject(sourceArrayDevice);
    clReleaseMemObject(parentCountArrayDevice);
    clReleaseMemObject(maxVerticeArrayDevice);
    
    clReleaseProgram(program);
    clReleaseKernel(initializeKernel);
    clReleaseKernel(ssspKernel1);
    clReleaseKernel(ssspKernel2);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);
    clReleaseEvent(readDone);
    clReleaseDevice(device_id);
    
}

void testRandomGraphs(int graphSetCount, int graphCount, int sourceCount, int verticeCount, int edgePerVerticeCount, float probOfMax) {
    
    GraphData graph;
    
    printf("Performing tests on randomly generated graphs.\n");
    
    srand(0);
    clock_t start_time = clock();
    generateRandomGraph(&graph, verticeCount, edgePerVerticeCount, graphCount, sourceCount, probOfMax);
    printf("Time to generate graph, including overhead: %.2f seconds.\n", (float)(clock()-start_time)/1000000);
    printf("%i vertices. %i attack steps per sample. %i samples divided into %i sets.\n", graph.vertexCount*graph.graphCount*graphSetCount, graph.vertexCount, graph.graphCount*graphSetCount, graphSetCount);
    
    start_time = clock();
    int *maxCostArray = (int*) malloc(graphSetCount* graph.graphCount * graph.vertexCount * sizeof(int));
    int *sumCostArray = (int*) malloc(graphSetCount* graph.graphCount * graph.vertexCount * sizeof(int));
    
    for (int iGraphSet = 0; iGraphSet < graphSetCount; iGraphSet++) {
        updateGraphWithNewRandomWeights(&graph);
        calculateGraphs(&graph, false);
        for (int iGlobalVertex=0; iGlobalVertex < graph.graphCount * graph.vertexCount; iGlobalVertex++) {
            maxCostArray[iGraphSet * graph.graphCount * graph.vertexCount + iGlobalVertex] = graph.costArray[iGlobalVertex];
            sumCostArray[iGraphSet * graph.graphCount * graph.vertexCount + iGlobalVertex] = graph.sumCostArray[iGlobalVertex];
        }
    }
    printf("\nTime to calculate graph, including overhead: %.2f seconds.\n", (float)(clock()-start_time)/1000000);
    
    maxSumDifference(&graph);
    compareToCPUComputation(&graph, false, 10);
    //printMathematicaString(&graph, 0, false);
    
    
    
    
    
}

void computeGraphsFromFile(char filePathToInData[], char filePathToOutData[], char filePathToNames[]) {
    GraphData graph;
    srand(0);
    
    
    printf("\nReading graph from file.\n");
    readGraphFromFile(&graph, filePathToInData);
    completeReadGraph(&graph);
    printf("Computing...\n");
    clock_t start_time = clock();
    calculateGraphs(&graph, false);
    printf("Time to calculate graph, including overhead: %.2f seconds.\n", (float)(clock()-start_time)/1000000);
    
    writeGraphToFile(&graph, filePathToOutData);
    
    char **verticeNameArray = (char**) malloc(graph.vertexCount * sizeof(char*));
    for (int i = 0; i < graph.vertexCount; i++)
        verticeNameArray[i] = (char*) malloc((512) * sizeof(char));
    //readVerticeNames(filePathToNames, verticeNameArray);
    
    compareToCPUComputation(&graph, false, 10);

    //printMathematicaString(&graph, 0, false);
    
    
}


int main(int argc, char** argv)
{
    
    testRandomGraphs(10, 10, 20, 200, 2, 0.2);
    
//    char filePathToInData[512] = "/Users/pontus/Documents/graph-2.out";
//    char filePathToOutData[512] = "/Users/pontus/Documents/graph-2.rst";
//    char filePathToNames[512] = "/Users/pontus/Documents/nodeNames.cvs";
//    computeGraphsFromFile(filePathToInData, filePathToOutData, filePathToNames);
    

    
    return 0;
}

