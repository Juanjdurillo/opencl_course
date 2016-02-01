#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>



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

	cl_queue_properties qprop[] = { CL_QUEUE_PROPERTIES,(cl_command_queue_properties)CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
									CL_QUEUE_PROFILING_ENABLE, 0 };
	

	cmdQueue = clCreateCommandQueueWithProperties(context, devices[selectedDevice],qprop, &status);
	if (status != 0) {
		printf("Error %d when creating the queue for the device %d on platform %d\n", status, selectedDevice, numPlatforms - 1);
		exit(status);
	}
	
	//free host resources
	clReleaseContext(context);
	clReleaseCommandQueue(cmdQueue);
	free(devices);
	free(platforms);
	return 0;
}
