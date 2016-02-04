#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <atomic>
#define FILE_MAX_LENGTH 1000000

typedef struct
{
	int *vertexArray;	// vertexes
	int vertexCount;	// #vertexes
	int *edgeArray;		// edges
	int edgeCount;		// #edges
	float *weightArray;	// weights
} GraphData;


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

int main() {

	cl_int status;
	cl_uint numPlatforms = 0;
	cl_uint numDevices = 0;
	cl_platform_id *platforms = NULL;
	cl_device_id   *devices = NULL;


	// the number of platforms is retrieved by using a first call
	// to clGetPlatformsIDs() with NULL argument as second argument
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != 0) {
		printf("Status value %d\n",status);
		exit(status);
	}

	printf("Number of platforms: %d\n", numPlatforms);


	// allocating memory for the platforms
	platforms = (cl_platform_id *)malloc(numPlatforms * sizeof(cl_platform_id));

	if (platforms == NULL) {
		printf("Not enough memory\n");
		exit(-1);
	}

	// the second call, get the platforms
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	if (status != 0) {
		printf("Status value %d\n", status);
		exit(status);
	}

	status = clGetDeviceIDs(platforms[2], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
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

	//we obtain them althoug so far we do nothing with them
	status = clGetDeviceIDs(platforms[2], CL_DEVICE_TYPE_ALL, numDevices, devices, 0);
	if (status != 0) {
		printf("Error %d when obtaining devices on platform %d\n", status, numPlatforms - 1);
		exit(status);
	}
		
	//creating a context 
	//-------------creating a context-------------------------//
	cl_context context = NULL;
	context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
	if (status != 0) {
		printf("Error %d when creating the context in platform %d\n", status, numPlatforms - 1);
		exit(status);
	}

	// creating a command queue
	cl_int selectedDevice = 0;
	cl_command_queue cmdQueue;

	cl_queue_properties qprop[] = { CL_QUEUE_PROPERTIES,(cl_command_queue_properties)
									CL_QUEUE_PROFILING_ENABLE, 0 };
	

	cmdQueue = clCreateCommandQueueWithProperties(context, devices[selectedDevice],qprop, &status);
	if (status != 0) {
		printf("Error %d when creating the queue for the device %d on platform %d\n", status, selectedDevice, numPlatforms - 1);
		exit(status);
	}

	
	
	cl_queue_properties qprop2[] = {CL_QUEUE_PROPERTIES,
		(cl_command_queue_properties)CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT, 0 };

	//cl_command_queue my_device_q = clCreateCommandQueueWithProperties(CLU_CONTEXT, cluGetDevice(CL_DEVICE_TYPE_GPU), qprop, &status);
	cl_command_queue my_device;
	my_device = clCreateCommandQueueWithProperties(context, devices[0], qprop2, &status);
	if (status != 0) {
		printf("Error when creating the program from the sources %d\n", status);
		exit(status);
	}
	
	
	
	// creating the program
	size_t sourceSize;
	char * programStr = createProgramFromFile("Kernels.cl",&sourceSize);
	const char * options = "-cl-std=CL2.0 -D CL_VERSION_2_0";

	cl_program clProgram;
	clProgram = clCreateProgramWithSource(context, 1, (const char **)&programStr, (const size_t*)&sourceSize, &status);
	if (status != 0) {
		printf("Error when creating the program from the sources %d\n", status);
		exit(status);
	}
	status = clBuildProgram(clProgram, 0, NULL, options, NULL, NULL);
	if (status != 0) {
		printf("Error when compiling the opencl code: %d\n", status);
		// Allocate memory for the log
		int log_size = 100000;
		char *log = (char *)malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(clProgram, devices[selectedDevice], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		printf("%s\n", log);
		exit(status);
	}
	

	cl_kernel initializeBuffersKernel;
	initializeBuffersKernel = clCreateKernel(clProgram, "initializeBuffers", &status);
	if (status != 0) {
		printf("Error: %d while creating kernel initializeBuffers\n", status);
		exit(status);
	}

	cl_kernel phase3;
	phase3 = clCreateKernel(clProgram, "phase3", &status);
	if (status != 0) {
		printf("Error: %d while creating kernel phase2\n", status);
		exit(status);
	}

	GraphData graph;
	generateRandomGraph(&graph, 100, 30);
	size_t maxWorkGroupSize;
	//clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);
	size_t localWorkSize = graph.vertexCount;// maxWorkGroupSize;
	size_t globalWorkSize = graph.vertexCount;// roundWorkSizeUp(localWorkSize, graph.vertexCount);


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






	status =  clSetKernelArg(initializeBuffersKernel, 0, sizeof(cl_mem), &maskArrayDevice);
	status |= clSetKernelArg(initializeBuffersKernel, 1, sizeof(cl_mem), &costArrayDevice);
	status |= clSetKernelArg(initializeBuffersKernel, 2, sizeof(cl_mem), &updatingCostArrayDevice);
	// argment 3rd of the kernel we set later
	//status |= clSetKernelArg(initializeBuffersKernel, 4, sizeof(int), &graph.vertexCount);
	if (status != 0) {
		printf("Error while setting the parameters of the initializingBufferKernel: %d\n", status);
		exit(status);
	}


	std::atomic_uint value;
	std::atomic_store(&value, 1);

	cl_mem buffer =
		clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(std::atomic_uint), NULL, &status);
	if (status != 0) {
		printf("Error: %d\n", status);
		exit(status);
	}

	clEnqueueWriteBuffer(cmdQueue, buffer, CL_TRUE, 0,sizeof(std::atomic_uint), &value,0,NULL,NULL);

	status = clSetKernelArg(phase3, 0, sizeof(cl_mem), &vertexArrayDevice);
	status |= clSetKernelArg(phase3, 1, sizeof(cl_mem), &edgeArrayDevice);
	status |= clSetKernelArg(phase3, 2, sizeof(cl_mem), &weightArrayDevice);
	status |= clSetKernelArg(phase3, 5, sizeof(cl_mem), &maskArrayDevice);
	status |= clSetKernelArg(phase3, 3, sizeof(cl_mem), &costArrayDevice);
	status |= clSetKernelArg(phase3, 4, sizeof(cl_mem), &updatingCostArrayDevice);
	status |= clSetKernelArg(phase3, 6, sizeof(cl_mem), &buffer);
	if (status != 0) {
		printf("Error while setting the parameters of the first phase kerne3l: %d\n", status);
		exit(status);
	}



	int *maskArrayHost = (int *)malloc(sizeof(int)*graph.vertexCount);
	cl_int source = 0;
	status = clSetKernelArg(initializeBuffersKernel, 3, sizeof(int), &source);
	if (status != 0) {
		printf("Error while setting the 3rd parameters of the initialization buffer kernel: %d\n", status);
		exit(status);
	}
	cl_event readDone;
	status = clEnqueueNDRangeKernel(cmdQueue, initializeBuffersKernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, &readDone);
	if (status != 0) {
		printf("Error while launching the kernel: %d\n", status);
		exit(status);
	}
	
	status = clWaitForEvents(1, &readDone);
	if (status != 0) {
		printf("Error: %d\n", status);
		exit(status);
	}

	printf("calling phase 3 \n");
	status = clEnqueueNDRangeKernel(cmdQueue, phase3, 1, 0, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
	if (status != 0) {
		printf("Error executing the first phase: %d\n", status);
		exit(status);
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


	//free host resources
	clReleaseContext(context);
	clReleaseCommandQueue(cmdQueue);
	clReleaseProgram(clProgram);
	clReleaseKernel(initializeBuffersKernel);

	clReleaseMemObject(vertexArrayDevice);
	clReleaseMemObject(edgeArrayDevice);
	clReleaseMemObject(weightArrayDevice);
	clReleaseMemObject(maskArrayDevice);
	clReleaseMemObject(costArrayDevice);
	clReleaseMemObject(updatingCostArrayDevice);
	clReleaseMemObject(hostVertexArrayBuffer);
	clReleaseMemObject(hostEdgeArrayBuffer);
	clReleaseMemObject(hostWeightArrayBuffer);

	free(devices);
	free(platforms);
	free(programStr);
	return 0;
}
