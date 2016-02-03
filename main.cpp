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


		cl_device_svm_capabilities caps;

		cl_int err = clGetDeviceInfo(
			devices[0],
			CL_DEVICE_SVM_CAPABILITIES,
			sizeof(cl_device_svm_capabilities),
			&caps,
			0
			);

		
		if (err == CL_SUCCESS && (caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) && (caps & CL_DEVICE_SVM_ATOMICS))
			printf("Fine-grained system with atomics\n");
		else if (err == CL_SUCCESS && (caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM))
			printf("Fine-grained system\n");
		else if (err == CL_SUCCESS && (caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) && (caps & CL_DEVICE_SVM_ATOMICS))
			printf("Fine-grained buffer with atomics\n");
		else if (err == CL_SUCCESS && (caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER))
			printf("Fine-grained buffer\n");
		//else if (err == CL_SUCCESS && (caps & CL_DEVICE_SVM_COARSE_GRAIN))
		//	printf("Coarse-grained buffer\n");
		else
			printf("No support \n");
	
	}

	while (true) {
		;
	}

	//free host resources	
	free(platforms);
	return 0;
}
