__kernel void foo(global float *p) {
 int idx = get_global_id(0);
 printf("Hola %f\n",*(p+idx));
 }
