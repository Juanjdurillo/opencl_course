#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <atomic>

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
	cl_mem buffer;

	//Elements in each array
	const int elements = 8;

	//size of the data
	size_t datasize = sizeof(int)*elements;

	// Allocate space for input/output data
	std::atomic_uint value;
	std::atomic_store(&value, 0);

	// the number of platforms is retrieved by using a first call
	// to clGetPlatformsIDs() with NULL argument as second argument
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != 0) {
		printf("Status value %d\n", status);
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

	status = clGetDeviceIDs(platforms[numPlatforms - 1], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
	if (status != 0) {
		printf("Error %d while retrieving number of devices for platform %d\n", status, numPlatforms - 1);
		exit(-1);
	}

	// allocate space for devices
	devices = (cl_device_id *)malloc(sizeof(cl_device_id)*numDevices);
	if (devices == NULL) {
		printf("Error allocating memory to store devices id for platform %d\n", numPlatforms - 1);
		exit(-1);
	}

	//we obtain them althoug so far we do nothing with them
	status = clGetDeviceIDs(platforms[numPlatforms - 1], CL_DEVICE_TYPE_ALL, numDevices, devices, 0);
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

	cl_queue_properties qprop[] = { CL_QUEUE_PROPERTIES, (cl_command_queue_properties)
		CL_QUEUE_PROFILING_ENABLE, 0 };


	cmdQueue = clCreateCommandQueueWithProperties(context, devices[selectedDevice], qprop, &status);
	if (status != 0) {
		printf("Error %d when creating the queue for the device %d on platform %d\n", status, selectedDevice, numPlatforms - 1);
		exit(status);
	}

	// creating the program
	size_t sourceSize;
	char * programStr = createProgramFromFile("Kernels.cl", &sourceSize);

	cl_program clProgram;
	clProgram = clCreateProgramWithSource(context, 1, (const char **)&programStr, (const size_t*)&sourceSize, &status);
	if (status != 0) {
		printf("Error when creating the program from the sources %d\n", status);
		exit(status);
	}

	const char * options = "-cl-std=CL2.0 -D CL_VERSION_2_0";
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

	cl_kernel clKernel;
	clKernel = clCreateKernel(clProgram, "foo", &status);
	if (status != 0) {
		printf("Error when selecting the kernel to execute: %d\n", status);
		exit(status);
	}

	buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(std::atomic_uint), NULL, &status);
	if (status != 0) {
		printf("Error %d when creating a memory buffer\n", status);
		exit(status);
	}
	

	// write data from the host to the device 
	status = clEnqueueWriteBuffer(cmdQueue, buffer, CL_FALSE, 0, sizeof(std::atomic_uint), &value, 0, NULL, NULL);
	if (status != 0) {
		printf("Error %d when writing information into the device\n", status);
		exit(status);
	}

	status = clSetKernelArg(clKernel, 0, sizeof(cl_mem), &buffer);
	if (status != 0) {
		printf("Error when setting the parameters of the kernel\n");
		exit(-1);
	}

	size_t globalWorkSize[1];
	size_t localWorkSize[1];
	globalWorkSize[0] = 8;
	localWorkSize[0] = 8;

	status = clEnqueueNDRangeKernel(cmdQueue, clKernel, 1, 0, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if (status != 0) {
		printf("Errror %d when enqueuing the kernel for execution", status);
		exit(status);
	}

	std::atomic_uint result;
	status = clEnqueueReadBuffer(cmdQueue, buffer, CL_TRUE, 0, sizeof(std::atomic_uint), &result, 0, NULL, NULL);

	unsigned int obgtained = std::atomic_load(&result);
	printf("%d\n", result);
	

	//free host resources
	clReleaseContext(context);
	clReleaseCommandQueue(cmdQueue);
	clReleaseProgram(clProgram);
	clReleaseKernel(clKernel);
	clReleaseMemObject(buffer);
	free(devices);
	free(platforms);
	free(programStr);
	
	return 0;
}