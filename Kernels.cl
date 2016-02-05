__kernel void foo() {
	int idx = get_global_id(0);
	printf("Hola %d\n", idx);
}