#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define FILE_MAX_LENGTH 1000000
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
	
	//Elements in each array
	const int elements = 16;

	//size of the data
	size_t datasize = sizeof(float)*elements*elements;


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

	status = clGetDeviceIDs(platforms[numPlatforms-1], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
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
	status = clGetDeviceIDs(platforms[numPlatforms-1], CL_DEVICE_TYPE_ALL, numDevices, devices, 0);
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

	// creating the program
	size_t sourceSize;
	char * programStr = createProgramFromFile("Kernels.cl",&sourceSize);

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
		clGetProgramBuildInfo(clProgram, devices[selectedDevice], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		printf("%s\n", log);
		exit(status);
	}

	cl_kernel clKernel;
	clKernel = clCreateKernel(clProgram, "matrixMul", &status);
	if (status != 0) {
		printf("Error when selecting the kernel to execute: %d\n", status);
		exit(status);
	}

	float* A = (float *)clSVMAlloc(context, CL_MEM_READ_ONLY,  datasize,0 );
	float *B = (float *)clSVMAlloc(context, CL_MEM_READ_ONLY, datasize, 0);;
	float *C = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, datasize, 0);;

	// Initialize the input matrices
	for (int i = 0; i < elements*elements; i++)  {
		A[i] = 1.0*i;
		B[i] = 0.0;
	}
	B[0] = 1;
	for (int i = elements + 1; i < elements*elements; i = i + elements + 1)  {
		B[i] = 1.0;
	}


	clSetKernelArgSVMPointer(clKernel, 1, A);
	clSetKernelArgSVMPointer(clKernel, 2, B);
	clSetKernelArgSVMPointer(clKernel, 0, C);
	
	status |= clSetKernelArg(clKernel, 3, sizeof(int), (void *)&elements);
	status |= clSetKernelArg(clKernel, 4, sizeof(int), (void *)&elements);
	if (status != 0) {
		printf("Error when setting the parameters of the kernel\n");
		exit(-1);
	}

	size_t globalWorkSize[2];
	size_t localWorkSize[2];

	globalWorkSize[0] = elements;
	globalWorkSize[1] = elements;

	localWorkSize[0] = 16;
	localWorkSize[1] = 16;
	cl_event event;
	status = clEnqueueNDRangeKernel(cmdQueue, clKernel, 2, 0, globalWorkSize, localWorkSize, 0, NULL, &event);
	if (status != 0) {
		printf("Errror %d when enqueuing the kernel for execution",status);
		exit(status);
	}

	clWaitForEvents(1, &event);
	cl_ulong start_time;
	cl_ulong finish_time;
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(finish_time), &finish_time, NULL);


	cl_ulong total_time = finish_time - start_time;
	printf("\start time in nanoseconds = %lu\n", start_time);
	printf("\nend time in nanoseconds = %lu\n", finish_time);
	printf("\nAverage time in nanoseconds = %lu\n", total_time);

	
	for (int i = 0; i < elements; i++) {
		printf("%f\n", C[i]);
	}



	//free host resources
	clReleaseContext(context);
	clReleaseCommandQueue(cmdQueue);
	clReleaseProgram(clProgram);
	clReleaseKernel(clKernel);
	free(devices);
	free(platforms);
	free(programStr);
	clSVMFree(context,A);
	clSVMFree(context, B);
	clSVMFree(context, C);
	
	return 0;
}
