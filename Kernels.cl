__kernel void foo() {
 int idx = get_global_id(0);
 printf("Hola %d\n",idx);
 }


__kernel void vecadd(__global int *A, __global int *B, __global int *C) {
 int idx = get_global_id(0); 
 C[idx] = A[idx] + B[idx]; 
 }


__kernel void matrixMul(__global float* C, __global float* A, __global float* B, int wA, int wB)
{

	int tx = get_global_id(0);
	int ty = get_global_id(1);

	float value = 0;
	for (int k = 0; k < wA; ++k)
	{
		float elementA = A[ty * wA + k];
		float elementB = B[k * wB + tx];
		value += elementA * elementB;
	}
	printf("value %f\n", value);

	C[ty * wA + tx] = value;
}
