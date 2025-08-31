#include <cuda_runtime.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#define N 10000000
#define BLOCK_SIZE 256

/*
GPU Spec:
NVIDIA GeForce RTX 3090
Compute Capability 8.6
10496 CUDA Cores

Results:

CPU execution time: 47.795247 milliseconds
GPU execution time: 0.153099 milliseconds
GPU speed up over CPU: 312.185230x
*/

void initRand(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

void vector_add_cpu(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_gpu(float* a, float* b, float* c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) 
        c[i] = a[i] + b[i];
}

double getTime() {
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime()");
        return -1;
    }

    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    float* h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float* d_a, *d_b, *d_c;

    size_t size = N * sizeof(float);

    h_a = (float*) malloc(size);
    h_b = (float*) malloc(size);
    h_c_cpu = (float*) malloc(size);
    h_c_gpu = (float*) malloc(size);

    srand(time(0));

    initRand(h_a, N);
    initRand(h_b, N);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;


    printf("Performing warm up runs\n");
    for (int i = 0; i < 3; i++) {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }

    printf("Performing CPU benchmarking...\n");
    double cpu_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start = getTime();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end = getTime();
        cpu_time += end - start;
    }
    cpu_time /= 20;

    printf("Performing GPU benchmarking...\n");
    double gpu_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start = getTime();
        vector_add_gpu<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double end = getTime();
        gpu_time += end - start;
    }
    gpu_time /= 20;

    printf("CPU execution time: %f milliseconds\n", cpu_time*1000);
    printf("GPU execution time: %f milliseconds\n", gpu_time*1000);
    printf("GPU speed up over CPU: %fx\n", cpu_time / gpu_time);

    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_gpu[i] - h_c_cpu[i]) > 1e-5) {
            correct = false;
            break;
        }
    }

    printf("Results are %s\n", correct ? "correct" : "incorrect");

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
