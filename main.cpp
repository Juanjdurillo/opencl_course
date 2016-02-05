#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>

#define MAX_SOURCE_SIZE 1000000

int main() {
	int *A = NULL; // input array
	int *B = NULL; // input array
	int *C = NULL; // input array

	//Elements in each array
	const int elements = 12;

	//size of the data
	size_t datasize = sizeof(int)*elements;

	// Allocate space for input/output data
	A = (int *)malloc(datasize);
	B = (int *)malloc(datasize);
	C = (int *)malloc(datasize);

	// Initialize the input data
	for (int i = 0; i < elements; i++)  {
		A[i] = i;
		B[i] = i;
	}


	//it is a good practice to check the status of each API call
	cl_int status;

	cl_uint numPlatforms = 0;
	cl_platform_id *platforms = NULL;


	//---------------------------------------------------------//
	//--------------retrieving platforms-----------------------//	
	// the number of platforms is retrieved by using a first call
	// to clGetPlatformsIDs() with NULL argument as second argument
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	platforms = (cl_platform_id *)malloc(
		numPlatforms * sizeof(cl_platform_id));

	printf("Number of platforms: %d\n", numPlatforms);
	// the second call, get the platforms
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);

	//--------------------------------------------------------//
	//-------------retrieving devices ------------------------//
	cl_uint numDevices = 0;
	cl_device_id *devices = NULL;

	// as before, the devices are retrieved in two steps
	status = clGetDeviceIDs(platforms[2], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
	printf("Number of devices: %d\n", numPlatforms);
	// allocating space for the devices
	devices = (cl_device_id *)malloc(numDevices * sizeof(cl_device_id));

	// the second call get the devices
	status = clGetDeviceIDs(platforms[2], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);

	//--------------------------------------------------------//
	//-------------creating a context-------------------------//
	cl_context context = NULL;
	context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);

	//--------------------------------------------------------//
	//-------------creating a comand queue -------------------//
	cl_command_queue cmdQueue;
	cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);
	cl_command_queue_properties properties = CL_QUEUE_ON_DEVICE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE_DEFAULT;


	cl_queue_properties qprop[] = { CL_QUEUE_SIZE, 10000000, CL_QUEUE_PROPERTIES,
		(cl_command_queue_properties)CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT, 0 };

	//cl_command_queue my_device_q = clCreateCommandQueueWithProperties(CLU_CONTEXT, cluGetDevice(CL_DEVICE_TYPE_GPU), qprop, &status);
	cl_command_queue my_device;
	my_device = clCreateCommandQueueWithProperties(context, devices[0], qprop, &status);
	if (status != 0) {
		printf("Error al crear la cola: %d\n", status);
		while (true) {
			;
		}
		exit(status);
	}


	//--------------------------------------------------------//
	//-------------creating device buffers -------------------//
	cl_mem bufferA; // Input Array on the device
	cl_mem bufferB; // Input Array on the device
	cl_mem bufferC; // Output Array on the device

	bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
	bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
	bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);

	// ------------------------------------------------------//
	//-------------writing the data on the device------------//	
	// write host data to device buffer (similar to CUDA_memcpy)
	status = clEnqueueWriteBuffer(cmdQueue, bufferA, CL_FALSE, 0, datasize, A, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(cmdQueue, bufferB, CL_FALSE, 0, datasize, B, 0, NULL, NULL);

	//-------------------------------------------------------//
	//-------------CREATE AND COMPILE THE PROGRAM------------//
	// 6. Load and build OpenCL kernel
	const char fileName[] = "kernels.cl";
	size_t sourceSize;
	char *source_str;
	FILE* fp = fopen(fileName, "rb");
	if (!fp) {
		printf("Error while loading the source code %s\n", fp);
		exit(1);
	}
	source_str = (char *)malloc(sizeof(char)*MAX_SOURCE_SIZE);

	sourceSize = fread(source_str, 1, MAX_SOURCE_SIZE, fp);

	fclose(fp);


	cl_program clProgram;
	cl_int errcode;
	const char * options = "-cl-std=CL2.0 -D CL_VERSION_2_0";
	clProgram = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t*)&sourceSize, &errcode);
	if (errcode != 0) {
		printf("Error: %d\n", errcode);
		exit(errcode);
	}
	errcode = clBuildProgram(clProgram, 0, NULL, options, NULL, NULL);
	if (errcode != 0) {
		printf("Error 2: %d\n", errcode);

		// Allocate memory for the log
		int log_size = 100000;
		char *log = (char *)malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(clProgram, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		printf("Error: %d\n", errcode);
		// Print the log
		//printf("%s\n", log);
		while (true) {
			;
		}

		exit(errcode);
	}
	cl_kernel kernel;
	kernel = clCreateKernel(clProgram, "parent_vecadd", &errcode);
	if (errcode != 0) {
		printf("Error: %d\n", errcode);
		exit(errcode);
	}


	//-------------------------------------------------------//
	//-------------Set the arguments of the kernel ----------//
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
	status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);

	//-------------------------------------------------------//
	//-------------configure the work-item structure---------//
	size_t globalWorkSize[1];
	globalWorkSize[0] = elements;

	//-------------------------------------------------------//
	//---------- ENQUEUE THE KERNEL EXECUTION ---------------//
	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (status != 0) {
		printf("Error about the kernel: %d\n", status);
		while (true) {
			;
		}
		exit(status);
	}

	//-------------------------------------------------------//
	//----------- Read the output buffer back to the host ---//
	cl_event event;
	clEnqueueReadBuffer(cmdQueue, bufferC, CL_FALSE, 0, datasize, C, 0, NULL, &event);



	status = clWaitForEvents(1, &event);
	//status = clWaitForEvents(1, &readDone);
	if (status != 0) {
		printf("Error: %d\n", status);
		exit(status);
	}
	//print the result
	for (int i = 0; i < elements; i++) {
		printf("%d + %d = %d\n", A[i], B[i], C[i]);
	}

	clReleaseKernel(kernel);
	clReleaseProgram(clProgram);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(bufferA);
	clReleaseMemObject(bufferB);
	clReleaseMemObject(bufferC);
	clReleaseContext(context);
	clReleaseEvent(event);
	free(A);
	free(B);
	free(C);
	free(platforms);
	free(devices);

	return 0;
}
