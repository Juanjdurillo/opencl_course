#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>



int main() {

	cl_int status;
	cl_uint numPlatforms = 0;
	cl_platform_id *platforms = NULL;


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
		char buffer[10240];
		printf("  More information about platform -- %d --\n", i);
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
	}

	//free host resources	
	free(platforms);
	return 0;
}
