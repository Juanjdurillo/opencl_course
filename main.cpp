#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "pthread.h"

#define FILE_MAX_LENGTH 1000000

typedef struct
{
	int *vertexArray;	// vertexes
	int vertexCount;	// #vertexes
	int *edgeArray;		// edges
	int edgeCount;		// #edges
	float *weightArray;	// weights
} GraphData;


typedef struct {
	int begin;
	int end;
	cl_device_id *device;
	cl_context context;
	GraphData graph;
} Arguments;



int roundWorkSizeUp(int groupSize, int globalSize)
{
	int remainder = globalSize % groupSize;
	if (remainder == 0)
	{
		return globalSize;
	}
	else
	{
		return globalSize + groupSize - remainder;
	}
}

void generateRandomGraph(GraphData *graph, int numVertexes, int neighborsPerVertex)
{
	graph->vertexCount = numVertexes;
	graph->vertexArray = (int *)malloc(sizeof(int)*numVertexes);
	graph->edgeCount = numVertexes * neighborsPerVertex;
	graph->edgeArray = (int *)malloc(sizeof(int)*graph->edgeCount);
	graph->weightArray = (float *)malloc(sizeof(float)* graph->edgeCount);


	for (int i = 0; i < graph->vertexCount; i++)
	{
		graph->vertexArray[i] = i * neighborsPerVertex;
	}

	for (int i = 0; i < graph->edgeCount; i++)
	{
		graph->edgeArray[i] = (rand() % graph->vertexCount);
		graph->weightArray[i] = (float)(rand() % 1000) / 1000.0f;
	}

}

bool newShortestPathFound(int *maskArray, int count) {
	for (int i = 0; i < count; i++) {
		if (maskArray[i] == 1)
			return false;
	}
	return true;
}

char *createProgramFromFile(const char*fileName, size_t *sourceSize) {
	//const char fileName[] = "matrixmul2.cl";
	char *source_str;
	FILE* fp = fopen(fileName, "rb");
	if (!fp) {
		printf("Error while loading the source code %s\n", fp);
		exit(-1);
	}
	source_str = (char *)malloc(sizeof(char)*FILE_MAX_LENGTH);
	*sourceSize = fread(source_str, 1, FILE_MAX_LENGTH, fp);
	fclose(fp);
	return source_str;
}

void dijkstra_thread(Arguments *args) {

	int begin = args->begin; 
	int end  =  args->end;
	cl_context context = args->context;
	GraphData graph = args->graph;


	cl_int status;
	size_t sourceSize;
	char * programStr = createProgramFromFile("Kernels.cl", &sourceSize);
	size_t maxWorkGroupSize;
	clGetDeviceInfo(*args->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);
	size_t localWorkSize = maxWorkGroupSize;
	printf("%d vertex count\n",graph.vertexCount);
	printf("%d begin\n", begin);
	printf("%d end \n", end);
	
	size_t globalWorkSize = roundWorkSizeUp(localWorkSize, graph.vertexCount);


	cl_command_queue cmdQueue;

	cmdQueue = clCreateCommandQueueWithProperties(context, *args->device, NULL, &status);
	if (status != 0) {
		printf("Error %d when creating the queue for the device\n", status);
		exit(status);
	}


	cl_program clProgram;
	clProgram = clCreateProgramWithSource(context, 1, (const char **)&programStr, (const size_t*)&sourceSize, &status);
	if (status != 0) {
		printf("Error when creating the program from the sources %d\n", status);
		exit(status);
	}
	status = clBuildProgram(clProgram, 0, NULL, NULL, NULL, NULL);
	if (status != 0) {
		printf("Error when compiling the opencl code: %d\n", status);
		// Allocate memory for the log
		int log_size = 100000;
		char *log = (char *)malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(clProgram,*args->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		printf("%s\n", log);
		exit(status);
	}


	cl_kernel initializeBuffersKernel, phase1, phase2;
	initializeBuffersKernel = clCreateKernel(clProgram, "initializeBuffers", &status);
	if (status != 0) {
		printf("Error: %d while creating kernel initializeBuffers\n", status);
		exit(status);
	}
	phase1 = clCreateKernel(clProgram, "phase1", &status);
	if (status != 0) {
		printf("Error: %d while creating kernel phase1\n", status);
		exit(status);
	}
	phase2 = clCreateKernel(clProgram, "phase2", &status);
	if (status != 0) {
		printf("Error: %d while creating kernel phase2\n", status);
		exit(status);
	}

	cl_mem vertexArrayDevice;
	cl_mem edgeArrayDevice;
	cl_mem weightArrayDevice;
	cl_mem maskArrayDevice;
	cl_mem costArrayDevice;
	cl_mem updatingCostArrayDevice;
	cl_mem hostVertexArrayBuffer;
	cl_mem hostEdgeArrayBuffer;
	cl_mem hostWeightArrayBuffer;


	hostVertexArrayBuffer =
		clCreateBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, sizeof(int)*graph.vertexCount, graph.vertexArray, &status);
	if (status != 0) {
		printf("Error: %d\n", status);
		exit(status);
	}

	hostEdgeArrayBuffer =
		clCreateBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, sizeof(int)*graph.edgeCount, graph.edgeArray, &status);
	if (status != 0) {
		printf("Error: %d\n", status);
		exit(status);
	}

	hostWeightArrayBuffer =
		clCreateBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, sizeof(float)*graph.edgeCount, graph.weightArray, &status);
	if (status != 0) {
		printf("Error: %d\n", status);
		exit(status);
	}

	vertexArrayDevice =
		clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)* globalWorkSize, NULL, &status);
	if (status != 0) {
		printf("Error: %d\n", status);
		exit(status);
	}
	edgeArrayDevice =
		clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)* graph.edgeCount, NULL, &status);
	if (status != 0) {
		printf("Error: %d\n", status);
		exit(status);
	}
	weightArrayDevice =
		clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)* graph.edgeCount, NULL, &status);
	if (status != 0) {
		printf("Error: %d\n", status);
		exit(status);
	}
	maskArrayDevice =
		clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)* globalWorkSize, NULL, &status);
	if (status != 0) {
		printf("Error: %d\n", status);
		exit(status);
	}
	costArrayDevice =
		clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)* globalWorkSize, NULL, &status);
	if (status != 0) {
		printf("Error: %d\n", status);
		exit(status);
	}
	updatingCostArrayDevice =
		clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)* globalWorkSize, NULL, &status);

	if (status != 0) {
		printf("Error: %d\n", status);
		exit(status);
	}
	status = clEnqueueCopyBuffer(cmdQueue, hostVertexArrayBuffer, vertexArrayDevice, 0, 0,
		sizeof(int)* graph.vertexCount, 0, NULL, NULL);

	if (status != 0) {
		printf("Error: %d\n", status);
		exit(status);
	}
	status = clEnqueueCopyBuffer(cmdQueue, hostEdgeArrayBuffer, edgeArrayDevice, 0, 0,
		sizeof(int)* graph.edgeCount, 0, NULL, NULL);
	if (status != 0) {
		printf("Error: %d\n", status);
		exit(status);
	}
	status = clEnqueueCopyBuffer(cmdQueue, hostWeightArrayBuffer, weightArrayDevice, 0, 0,
		sizeof(float)* graph.edgeCount, 0, NULL, NULL);
	if (status != 0) {
		printf("Error: %d\n", status);
		exit(status);
	}



	status = clSetKernelArg(initializeBuffersKernel, 0, sizeof(cl_mem), &maskArrayDevice);
	status |= clSetKernelArg(initializeBuffersKernel, 1, sizeof(cl_mem), &costArrayDevice);
	status |= clSetKernelArg(initializeBuffersKernel, 2, sizeof(cl_mem), &updatingCostArrayDevice);
	// argment 3rd of the kernel we set later
	status |= clSetKernelArg(initializeBuffersKernel, 4, sizeof(int), &graph.vertexCount);
	if (status != 0) {
		printf("Error while setting the parameters of the initializingBufferKernel: %d\n", status);
		exit(status);
	}

	status = clSetKernelArg(phase1, 0, sizeof(cl_mem), &vertexArrayDevice);
	status |= clSetKernelArg(phase1, 1, sizeof(cl_mem), &edgeArrayDevice);
	status |= clSetKernelArg(phase1, 2, sizeof(cl_mem), &weightArrayDevice);
	status |= clSetKernelArg(phase1, 3, sizeof(cl_mem), &maskArrayDevice);
	status |= clSetKernelArg(phase1, 4, sizeof(cl_mem), &costArrayDevice);
	status |= clSetKernelArg(phase1, 5, sizeof(cl_mem), &updatingCostArrayDevice);
	status |= clSetKernelArg(phase1, 6, sizeof(int), &graph.vertexCount);
	status |= clSetKernelArg(phase1, 7, sizeof(int), &graph.edgeCount);
	if (status != 0) {
		printf("Error while setting the parameters of the first phase kernel: %d\n", status);
		exit(status);
	}

	status = clSetKernelArg(phase2, 0, sizeof(cl_mem), &vertexArrayDevice);
	status |= clSetKernelArg(phase2, 1, sizeof(cl_mem), &edgeArrayDevice);
	status |= clSetKernelArg(phase2, 2, sizeof(cl_mem), &weightArrayDevice);
	status |= clSetKernelArg(phase2, 3, sizeof(cl_mem), &maskArrayDevice);
	status |= clSetKernelArg(phase2, 4, sizeof(cl_mem), &costArrayDevice);
	status |= clSetKernelArg(phase2, 5, sizeof(cl_mem), &updatingCostArrayDevice);
	status |= clSetKernelArg(phase2, 6, sizeof(int), &graph.vertexCount);
	if (status != 0) {
		printf("Error while setting the parameters of the second phase kernel: %d\n", status);
		exit(status);
	}


	

	for (int s = begin; s < end; s++) {
		int *maskArrayHost = (int *)malloc(sizeof(int)*graph.vertexCount);
		status = clSetKernelArg(initializeBuffersKernel, 3, sizeof(int), &s);
		if (status != 0) {
			printf("Error while setting the 3rd parameters of the initialization buffer kernel: %d\n", status);
			exit(status);
		}

		status = clEnqueueNDRangeKernel(cmdQueue, initializeBuffersKernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
		if (status != 0) {
			printf("Error while launching the kernel: %d\n", status);
			exit(status);
		}

		cl_event readDone;
		status = clEnqueueReadBuffer(cmdQueue, maskArrayDevice, CL_FALSE, 0, sizeof(int)*graph.vertexCount, maskArrayHost, 0, NULL, &readDone);
		if (status != 0) {
			printf("Error reading the buffer: %d\n", status);
			exit(status);
		}
		status = clWaitForEvents(1, &readDone);
		if (status != 0) {
			printf("Error: %d\n", status);
			exit(status);
		}

		while (!newShortestPathFound(maskArrayHost, graph.vertexCount)) {
			status = clEnqueueNDRangeKernel(cmdQueue, phase1, 1, 0, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
			if (status != 0) {
				printf("Error executing the first phase: %d\n", status);
				exit(status);
			}
			status = clEnqueueNDRangeKernel(cmdQueue, phase2, 1, 0, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
			if (status != 0) {
				printf("Error executing the second phase: %d\n", status);
				exit(status);
			}
			status = clEnqueueReadBuffer(cmdQueue, maskArrayDevice, CL_FALSE, 0, sizeof(int)*graph.vertexCount, maskArrayHost, 0, NULL, &readDone);
			if (status != 0) {
				printf("Error: %d\n", status);
				exit(status);
			}
			status = clWaitForEvents(1, &readDone);
			if (status != 0) {
				printf("Error: %d\n", status);
				exit(status);
			}
		}

		float *results = (float*)malloc(sizeof(float)* graph.vertexCount);
		status = clEnqueueReadBuffer(cmdQueue, costArrayDevice, CL_FALSE, 0, sizeof(float)*graph.vertexCount, results, 0, NULL, &readDone);
		if (status != 0) {
			printf("Error: %d\n", status);
			exit(status);
		}
		clWaitForEvents(1, &readDone);
		for (int i = 0; i < graph.vertexCount; i++) {
			printf("%f\t", results[i]);
		}
		printf("\n");
		free(maskArrayHost);
	}
	free(programStr);
	clReleaseCommandQueue(cmdQueue);
	clReleaseProgram(clProgram);
	clReleaseKernel(initializeBuffersKernel);
	clReleaseKernel(phase1);
	clReleaseKernel(phase2);

	clReleaseMemObject(vertexArrayDevice);
	clReleaseMemObject(edgeArrayDevice);
	clReleaseMemObject(weightArrayDevice);
	clReleaseMemObject(maskArrayDevice);
	clReleaseMemObject(costArrayDevice);
	clReleaseMemObject(updatingCostArrayDevice);
	clReleaseMemObject(hostVertexArrayBuffer);
	clReleaseMemObject(hostEdgeArrayBuffer);
	clReleaseMemObject(hostWeightArrayBuffer);

}




int main() {

	cl_int status;
	cl_uint numPlatforms = 0;
	cl_uint numDevices = 0;
	cl_platform_id *platforms = NULL;
	cl_device_id   *devices = NULL;


	// the number of platforms is retrieved by using a first call
	// to clGetPlatformsIDs() with NULL argument as second argument
	pthread_t *threadIDs = (pthread_t*)malloc(sizeof(pthread_t)* 2);
	Arguments *args = (Arguments *)malloc(sizeof(Arguments)* 2);
	cl_device_id *devicesToUse = (cl_device_id *)malloc(sizeof(cl_device_id)*2);
	
	
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != 0) {
		printf("Status value %d\n",status);
		exit(status);
	}

	// allocating memory for the platforms
	platforms = (cl_platform_id *)malloc(numPlatforms * sizeof(cl_platform_id));
	if (platforms == NULL) {
		printf("Not enough memory\n");
		exit(-1);
	}
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	if (status != 0) {
		printf("Status value %d\n", status);
		exit(status);
	}


	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
	if (status != 0) {
		printf("Error %d while retrieving number of devices for platform %d\n", status, numPlatforms-1);
		exit(-1);
	}

	// allocate space for devices
	devices = (cl_device_id *)malloc(sizeof(cl_device_id)*numDevices);
	if (devices == NULL) {
		printf("Error allocating memory to store devices id for platform %d\n", numPlatforms-1);
		exit(-1);
	}
	
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, numDevices, devices, 0);
	if (status != 0) {
		printf("Error %d while retrieving number of devices for platform %d\n", status, numPlatforms - 1);
		exit(-1);
	}
	
	cl_context context1 = NULL;
	context1 = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
	if (status != 0) {
		printf("Error %d when creating the context in platform %d\n", status, numPlatforms - 1);
		exit(status);
	}

	
	

	cl_device_id *devices2;
	status = clGetDeviceIDs(platforms[2], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
	if (status != 0) {
		printf("Error %d while retrieving number of devices for platform %d\n", status, numPlatforms - 1);
		exit(-1);
	}
	devices2 = (cl_device_id *)malloc(sizeof(cl_device_id)*numDevices);
	if (devices == NULL) {
		printf("Error allocating memory to store devices id for platform %d\n", numPlatforms - 1);
		exit(-1);
	}
	status = clGetDeviceIDs(platforms[2], CL_DEVICE_TYPE_ALL, numDevices, devices2, 0);
	if (status != 0) {
		printf("Error %d while retrieving number of devices for platform %d\n", status, numPlatforms - 1);
		exit(-1);
	}
	
	devicesToUse[0] = devices[1];
	devicesToUse[1] = devices2[0];

	GraphData graph;
	generateRandomGraph(&graph, 10, 3);


	

	cl_context context2 = NULL;
	context2 = clCreateContext(NULL, numDevices, devices2, NULL, NULL, &status);
	if (status != 0) {
		printf("Error %d when creating the context in platform %d\n", status, numPlatforms - 1);
		exit(status);
	}


	args[0].begin = 0;
	args[0].end = 4;
	args[0].context = context1;
	args[0].graph = graph;


	args[1].begin = 5;
	args[1].end = 9;
	args[1].context = context2;
	args[1].graph = graph;


	// creating a command queue
	
	numDevices = 2;
	for (unsigned int selectedDevice = 0; selectedDevice < numDevices; selectedDevice++) {
		printf("%d selected device\n ", selectedDevice);
		args[selectedDevice].device = (devicesToUse + selectedDevice);
		pthread_create(&threadIDs[selectedDevice], NULL, (void* (*)(void*))dijkstra_thread, (void*)(args + selectedDevice));
		//dijkstra_thread((Arguments *)(args + selectedDevice));
	}



	// Wait for the results from all threads
	for (unsigned int selectedDevice = 0; selectedDevice < numDevices; selectedDevice++)
	{
		pthread_join(threadIDs[selectedDevice], NULL);
	}
	


	while (true) {
		;
	}
	//free host resources
	clReleaseContext(context1);
	clReleaseContext(context2);
	free(devices);
	free(platforms);
	
	return 0;
}
