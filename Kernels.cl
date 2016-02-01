__kernel void foo() {
 int idx = get_global_id(0);
 printf("Hola %d\n",idx);
 }


__kernel void vecadd(__global int *A, __global int *B, __global int *C) {
 int idx = get_global_id(0); 
 C[idx] = A[idx] + B[idx]; 
 }
