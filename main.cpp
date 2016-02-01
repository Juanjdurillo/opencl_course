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

	//--------------------------------------------------------//
	//---------printing_platforms_information-----------------//
	//--------------------------------------------------------//

	for (unsigned int i = 0; i < numPlatforms; i++) {

		status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
		if (status != 0) {
			printf("Error %d while retrieving number of devices for platform %d\n", status, i);
			exit(-1);
		}

		// allocate space for devices
		devices = (cl_device_id *)malloc(sizeof(cl_device_id)*numDevices);
		if (devices == NULL) {
			printf("Error allocating memory to store devices id for platform %d\n", i);
			exit(-1);
		}

		//we obtain them althoug so far we do nothing with them
		status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices, 0);

		
		char buffer[10240];
		printf("  Information about platform -- %d --\n", i);
		printf("The platform has %d devices \n",numDevices);
		clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, 10240, buffer, NULL);
		printf("  PROFILE = %s\n", buffer);
		clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 10240, buffer, NULL);
		printf("  VERSION = %s\n", buffer);
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 10240, buffer, NULL);
		printf("  NAME = %s\n", buffer);
		clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 10240, buffer, NULL);
		printf("  VENDOR = %s\n", buffer);
		clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, 10240, buffer, NULL);
		printf("  EXTENSIONS = %s\n", buffer);
		free(devices);
	}


	//free host resources	
	free(platforms);
	return 0;
}
