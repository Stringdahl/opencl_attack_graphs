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
#include <OpenCL/opencl.h>
#include <pthread.h>
#include<time.h>
#include "graph.hpp"
#include "utility.hpp"



///
//  Macros
//
#define checkError(a, b) checkErrorFileLine(a, b, __FILE__ , __LINE__)
#define NUM_ASYNCHRONOUS_ITERATIONS 10  // Number of async loop iterations before attempting to read results back

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
void allocateOCLBuffers(cl_context gpuContext, cl_command_queue commandQueue, GraphData *graph, cl_mem *vertexArrayDevice, cl_mem *edgeArrayDevice, cl_mem *weightArrayDevice, cl_mem *maskArrayDevice, cl_mem *costArrayDevice, cl_mem *updatingCostArrayDevice, cl_mem *traversedEdgeArrayDevice, cl_mem *sourceArrayDevice, size_t globalWorkSize)
{
    cl_int errNum;
    cl_mem hostVertexArrayBuffer;
    cl_mem hostEdgeArrayBuffer;
    cl_mem hostWeightArrayBuffer;
    cl_mem hostTraversedEdgeArrayBuffer;
    cl_mem hostSourceArrayBuffer;
    
    
    // Initially, no edges have been travelled
    int *traversedEdgeArray = (int*)malloc(graph->edgeCount * sizeof(int));
    for (int iEdge=0; iEdge<graph->edgeCount; iEdge++) {
        traversedEdgeArray[iEdge]=0;
    }
    
    
    // First, need to create OpenCL Host buffers that can be copied to device buffers
    hostVertexArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                           sizeof(int) * graph->vertexCount, graph->vertexArray, &errNum);
    checkError(errNum, CL_SUCCESS);
    
    hostEdgeArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                         sizeof(int) * graph->edgeCount, graph->edgeArray, &errNum);
    checkError(errNum, CL_SUCCESS);
    
    hostWeightArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                           sizeof(float) * graph->graphCount * graph->edgeCount, graph->weightArray, &errNum);
    checkError(errNum, CL_SUCCESS);
    
    hostTraversedEdgeArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                                  sizeof(int) * graph->edgeCount, traversedEdgeArray, &errNum);
    checkError(errNum, CL_SUCCESS);
    
    hostSourceArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                           sizeof(int) * graph->graphCount, graph->sourceArray, &errNum);
    checkError(errNum, CL_SUCCESS);
    
    // Now create all of the GPU buffers
    *vertexArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(int) * globalWorkSize, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *edgeArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(int) * graph->edgeCount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *weightArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(float) * graph->graphCount* graph->edgeCount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *maskArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(int) * globalWorkSize, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *costArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(float) * globalWorkSize, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *updatingCostArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(float) * globalWorkSize, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *traversedEdgeArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(int) * graph->edgeCount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *sourceArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(int) * graph->graphCount, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        printf("Error: Failed to create source array buffer!\n");
        exit(1);
    }
    checkError(errNum, CL_SUCCESS);
    
    //
    
    // Now queue up the data to be copied to the device
    errNum = clEnqueueCopyBuffer(commandQueue, hostVertexArrayBuffer, *vertexArrayDevice, 0, 0,
                                 sizeof(int) * graph->vertexCount, 0, NULL, NULL);
    checkError(errNum, CL_SUCCESS);
    
    errNum = clEnqueueCopyBuffer(commandQueue, hostEdgeArrayBuffer, *edgeArrayDevice, 0, 0,
                                 sizeof(int) * graph->edgeCount, 0, NULL, NULL);
    checkError(errNum, CL_SUCCESS);
    
    errNum = clEnqueueCopyBuffer(commandQueue, hostWeightArrayBuffer, *weightArrayDevice, 0, 0,
                                 sizeof(float) * graph->graphCount* graph->edgeCount, 0, NULL, NULL);
    checkError(errNum, CL_SUCCESS);
    
    errNum = clEnqueueCopyBuffer(commandQueue, hostTraversedEdgeArrayBuffer, *traversedEdgeArrayDevice, 0, 0,
                                 sizeof(int) * graph->edgeCount, 0, NULL, NULL);
    checkError(errNum, CL_SUCCESS);
    
    errNum = clEnqueueCopyBuffer(commandQueue, hostSourceArrayBuffer, *sourceArrayDevice, 0, 0,
                                 sizeof(int) * graph->graphCount, 0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        printf("Error: Failed to enqueue source array buffer!\n");
        exit(1);
    }
    checkError(errNum, CL_SUCCESS);
    
    clReleaseMemObject(hostVertexArrayBuffer);
    clReleaseMemObject(hostEdgeArrayBuffer);
    clReleaseMemObject(hostWeightArrayBuffer);
    clReleaseMemObject(hostTraversedEdgeArrayBuffer);
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
        char cBuildLog[10240];
        clGetProgramBuildInfo(program, getFirstDev(gpuContext), CL_PROGRAM_BUILD_LOG,
                              sizeof(cBuildLog), cBuildLog, NULL );
        
        cerr << cBuildLog << endl;
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
    *commands = clCreateCommandQueue(*context, *device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }
    
    // Create the compute program from the source file
    *program = loadAndBuildProgram(*context, "/Users/pontus/Dropbox/Pontus/Pontus Program Files/XCode/OpenCLDijkstra/OpenCLDijkstra/kernel.cl");
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

int createKernels(cl_kernel *initializeKernel, cl_kernel *ssspKernel1, cl_kernel *ssspKernel2, cl_program *program) {
    
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
    return errNum;
}


int setKernelArguments(cl_kernel *initializeKernel, cl_kernel *ssspKernel1, cl_kernel *ssspKernel2, int graphCount, int vertexCount, int edgeCount, cl_mem *maskArrayDevice, cl_mem *vertexArrayDevice, cl_mem *edgeArrayDevice, cl_mem *costArrayDevice, cl_mem *updatingCostArrayDevice, cl_mem *sourceArrayDevice, cl_mem *weightArrayDevice) {
    
    
    // Set the arguments to initializeKernel
    //
    int errNum = 0;
    errNum |= clSetKernelArg(*initializeKernel, 0, sizeof(cl_mem), maskArrayDevice);
    errNum |= clSetKernelArg(*initializeKernel, 1, sizeof(cl_mem), costArrayDevice);
    errNum |= clSetKernelArg(*initializeKernel, 2, sizeof(cl_mem), updatingCostArrayDevice);
    errNum |= clSetKernelArg(*initializeKernel, 3, sizeof(int), &vertexCount);
    errNum |= clSetKernelArg(*initializeKernel, 4, sizeof(int), &graphCount);
    errNum |= clSetKernelArg(*initializeKernel, 5, sizeof(cl_mem), sourceArrayDevice);
    
    // Set the arguments to ssspKernel1
    errNum |= clSetKernelArg(*ssspKernel1, 0, sizeof(cl_mem), vertexArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel1, 1, sizeof(cl_mem), edgeArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel1, 2, sizeof(cl_mem), weightArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel1, 3, sizeof(cl_mem), maskArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel1, 4, sizeof(cl_mem), costArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel1, 5, sizeof(cl_mem), updatingCostArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel1, 6, sizeof(int), &vertexCount);
    errNum |= clSetKernelArg(*ssspKernel1, 7, sizeof(int), &edgeCount);
    
    // Set the arguments to ssspKernel2
    errNum |= clSetKernelArg(*ssspKernel2, 0, sizeof(cl_mem), vertexArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel2, 1, sizeof(cl_mem), edgeArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel2, 2, sizeof(cl_mem), weightArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel2, 3, sizeof(cl_mem), maskArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel2, 4, sizeof(cl_mem), costArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel2, 5, sizeof(cl_mem), updatingCostArrayDevice);
    errNum |= clSetKernelArg(*ssspKernel2, 6, sizeof(int), &vertexCount);
    if (errNum != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", errNum);
    }
    return errNum;
}



int main(int argc, char** argv)
{
    int errNum;                            // error code returned from api calls
    GraphData graph;
    
    size_t global;                      // global domain size for our calculation
    
    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commandQueue;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel initializeKernel;                   // compute kernel
    cl_kernel ssspKernel1;
    cl_kernel ssspKernel2;
    cl_event readDone;
    
    cl_mem vertexArrayDevice;                       // device memory used for the input array
    cl_mem edgeArrayDevice;                       // device memory used for the input array
    cl_mem weightArrayDevice;                       // device memory used for the input array
    cl_mem maskArrayDevice;                       // device memory used for the input array
    cl_mem costArrayDevice;                       // device memory used for the input array
    cl_mem updatingCostArrayDevice;                       // device memory used for the input array
    cl_mem traversedEdgeArrayDevice;            // was this edge already traversed?
    cl_mem sourceArrayDevice;            // which are teh sources?
    
    int nVertices = 2500;
    int nEdgePerVertice = 2;
    int nGraphs = 25;
    
    generateRandomGraph(&graph, nVertices, nEdgePerVertice, nGraphs);

    
    int totalVertexCount = graph.graphCount * graph.vertexCount;
    int *maskArrayHost = (int*) malloc(sizeof(int) * totalVertexCount);
    
    // printSources(&graph);
    // printGraph(graph);
    // printWeights(&graph);
    
    // Set up OpenCL computing environment, getting GPU device ID, command queue, context, and program
    initializeComputing(&device_id, &context, &commandQueue, &program);
    
    // Create kernels from the program (kernel.cl)
    createKernels(&initializeKernel, &ssspKernel1, &ssspKernel2, &program);
    
    // Allocate buffers in Device memory
    allocateOCLBuffers(context, commandQueue, &graph, &vertexArrayDevice, &edgeArrayDevice, &weightArrayDevice, &maskArrayDevice, &costArrayDevice, &updatingCostArrayDevice, &traversedEdgeArrayDevice, &sourceArrayDevice, totalVertexCount);
    
    // Setting the kernel arguments
    errNum = setKernelArguments(&initializeKernel, &ssspKernel1, &ssspKernel2, graph.graphCount, graph.vertexCount, graph.edgeCount, &maskArrayDevice, &vertexArrayDevice, &edgeArrayDevice, &costArrayDevice, &updatingCostArrayDevice, &sourceArrayDevice, &weightArrayDevice);
    
    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global = totalVertexCount;
    
    errNum = clEnqueueNDRangeKernel(commandQueue, initializeKernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    checkError(errNum, CL_SUCCESS);
    
    errNum = clEnqueueReadBuffer( commandQueue, maskArrayDevice, CL_FALSE, 0, sizeof(int) * totalVertexCount, maskArrayHost, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS);
    
    clWaitForEvents(1, &readDone);
    
    clock_t startTime = clock();

    while(!maskArrayEmpty(maskArrayHost, totalVertexCount))
    {
        // printMaskArray(maskArrayHost, totalVertexCount);
        // printCostUpdating(&graph, &commandQueue, &maskArrayDevice, &costArrayDevice, &updatingCostArrayDevice, &weightArrayDevice);
        
        // In order to improve performance, we run some number of iterations
        // without reading the results.  This might result in running more iterations
        // than necessary at times, but it will in most cases be faster because
        // we are doing less stalling of the GPU waiting for results.
        
        //for(int asyncIter = 0; asyncIter < NUM_ASYNCHRONOUS_ITERATIONS; asyncIter++)
        //{
        errNum = clEnqueueNDRangeKernel(commandQueue, ssspKernel1, 1, 0, &global, NULL, 0, NULL, NULL);
        checkError(errNum, CL_SUCCESS);
        errNum = clEnqueueNDRangeKernel(commandQueue, ssspKernel2, 1, 0, &global, NULL, 0, NULL, NULL);
        checkError(errNum, CL_SUCCESS);
        //}
        
        errNum = clEnqueueReadBuffer(commandQueue, maskArrayDevice, CL_FALSE, 0, sizeof(int) * totalVertexCount, maskArrayHost, 0, NULL, &readDone);
        checkError(errNum, CL_SUCCESS);
        clWaitForEvents(1, &readDone);
    }
    // Wait for the command commands to get serviced before reading back results
    clFinish(commandQueue);
    float diff = ((float)(clock() - startTime) / 1000000.0F ) * 1000;

    // Read back the results from the device to verify the output
    
    errNum = clEnqueueReadBuffer( commandQueue, costArrayDevice, CL_TRUE, 0, sizeof(float) * totalVertexCount, graph.costArray, 0, NULL, NULL );
    checkError(errNum, CL_SUCCESS);
    
    printCostOfRandomVertices(graph.costArray, 30, totalVertexCount);
    
    printf("Completed calculations in %f milliseconds.\n", diff);
    
    
    printMathematicaString(&graph, 1);
    
    
    // Produce graph string for Mathematica.
    
    
    
    
    
    
    
    
    // Shutdown and cleanup
    //
    clReleaseProgram(program);
    clReleaseKernel(initializeKernel);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);
    
    return 0;
}

