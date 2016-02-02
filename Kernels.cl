#define BLOCK_SIZE 8

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

	C[ty * wA + tx] = value;
}



__kernel void matrixMul2(__global float* C, __global float* A, __global float* B, int wA, int wB) {

	int g_x = get_global_id(0);
	int g_y = get_global_id(1);
	int l_x = get_local_id(0);
	int l_y = get_local_id(1);
	int h_s = get_global_size(0);
	int v_s = get_global_size(1);
	int group_x = get_group_id(0);
	int steps = h_s / BLOCK_SIZE;

	__local float LA[BLOCK_SIZE][BLOCK_SIZE];
	__local float LB[BLOCK_SIZE][BLOCK_SIZE];

	float value = 0;

	for (int i = 0; i < steps; i++) {
		LA[l_x][l_y] = A[h_s*g_y + i*BLOCK_SIZE + l_x];
		LB[l_x][l_y] = B[(i*BLOCK_SIZE + l_y)*h_s + group_x * BLOCK_SIZE + l_x];

		//printf("%d,%d :: %f, %f\n", g_x, g_y, LA[l_x][l_y], LB[l_x][l_y]);


		barrier(CLK_LOCAL_MEM_FENCE);
		for (int k = 0; k < BLOCK_SIZE; k++)
			value += LA[k][l_y] * LB[l_x][k];


		barrier(CLK_LOCAL_MEM_FENCE);
	}
	C[h_s*g_y + g_x] = value;


}