#include <cuda_runtime.h>
#include <iostream>

#define N 10000000

int main() {
   float* h_a, *h_b, *h_c_cpu, *h_c_gpu;
   float* d_a, *d_b, *d_c;

   size_t size = N * sizeof(float);

   h_a = malloc(size);
   h_b = malloc(size);
   h_c = malloc(size);

   initRand(h_a, N);
   initRand(h_b, N);
  
   cudaMalloc(&d_a, size);
   cudaMalloc(&d_b, size);
   cudaMalloc(&d_c, size);

   free(h_a);
   free(h_b);
   free(h_c);
   free(d_a);
   free(d_b);
   free(d_c);
}
