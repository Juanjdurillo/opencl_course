__kernel void foo(__global atomic_uint *value) {
	atomic_fetch_add(value, 1);
}