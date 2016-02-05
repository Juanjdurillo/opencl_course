__kernel void vecadd(__global int *A, __global int *B, __global int *C) {
	printf("child\n");
	int idx = get_global_id(0);
	C[idx] = A[idx] + B[idx];
}


__kernel void parent_vecadd(__global int *A, __global int* B, __global int* C) {
	ndrange_t child_ndrange = ndrange_1D(get_global_size(0));
	printf("parent\n");
	//only enqueue one child kernel
	if (get_global_id(0) == 0) {
		enqueue_kernel(get_default_queue(),
			CLK_ENQUEUE_FLAGS_NO_WAIT,
			child_ndrange,
			^{ vecadd(A, B, C); });
	}
}
