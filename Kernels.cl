#define VERTEX_COUNT 10
#define EDGE_COUNT 30

__kernel void phase3(__global int   *vertexArray, __global int *edgeArray, __global float *weightArray,
	__global float *costArray, __global float *updatingCostArray, __global int *maskArray, __global atomic_uint *modified);
__kernel  void phase1(__global int *vertexArray, __global int *edgeArray, __global float *weightArray,
	 __global float *costArray, __global float *updatingCostArray,__global int *maskArray, __global atomic_uint *modified)
{

	printf("%d phase 1\n", get_global_id(0));
	// access thread id
	int tid = get_global_id(0);


	if (maskArray[tid] != 0) {
		maskArray[tid] = 0;

		int edgeStart = vertexArray[tid];
		int edgeEnd;
		if (tid + 1 < VERTEX_COUNT)
		{
			edgeEnd = vertexArray[tid + 1];
		}
		else
		{
			edgeEnd = EDGE_COUNT;
		}

		for (int edge = edgeStart; edge < edgeEnd; edge++)
		{
			int nid = edgeArray[edge];
			if (updatingCostArray[nid] >(costArray[tid] + weightArray[edge]))
			{
				atomic_store(modified, 1);
				updatingCostArray[nid] = (costArray[tid] + weightArray[edge]);
			}
		}
	}
	if (tid == 0) {
		ndrange_t child_ndrange = ndrange_1D(get_global_size(0));
		enqueue_kernel(get_default_queue(),
			CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
			child_ndrange,
			^{ phase3(vertexArray, edgeArray, weightArray, costArray, updatingCostArray, maskArray, modified); });
	}
}


/*
__kernel  void phase2(__global int *vertexArray, __global int *edgeArray, __global float *weightArray,
	__global int *maskArray, __global float *costArray, __global float *updatingCostArray)
{
	// access thread id
	int tid = get_global_id(0);

	if (costArray[tid] > updatingCostArray[tid])
	{
		costArray[tid] = updatingCostArray[tid];
		maskArray[tid] = 1;
	}

	updatingCostArray[tid] = costArray[tid];
}


*/
__kernel void phase3(__global int   *vertexArray, __global int *edgeArray, __global float *weightArray,
	                 __global float *costArray, __global float *updatingCostArray, __global int *maskArray, __global atomic_uint *modified) {
	
	printf("%d phase 3\n", get_global_id(0));
	int tid = get_global_id(0);
	if (costArray[tid] > updatingCostArray[tid])
	{
		costArray[tid] = updatingCostArray[tid];
		maskArray[tid] = 1;

	}

	if (tid == 0) {
		int value = atomic_load(modified);
		if (value > 0) {  
			ndrange_t child_ndrange = ndrange_1D(get_global_size(0));
			//clk_event_t event1, event2, event3;
			atomic_store(modified, 0);
			enqueue_kernel(get_default_queue(),
				CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
				child_ndrange,
				^{ phase1(vertexArray, edgeArray, weightArray, costArray, updatingCostArray, maskArray,modified); });
				
		}

	}

}
///
/// Kernel to initialize buffers
///
__kernel void initializeBuffers(__global int *maskArray, 
								__global float *costArray, 
								__global float *updatingCostArray,
								int sourceVertex)
{
	// access thread id
	int tid = get_global_id(0);


	if (sourceVertex == tid)
	{
		maskArray[tid] = 1;
		costArray[tid] = 0.0;
		updatingCostArray[tid] = 0.0;
	}
	else
	{
		maskArray[tid] = 0;
		costArray[tid] = 3000.0f;
		updatingCostArray[tid] = 3000.0f;
	}
}

