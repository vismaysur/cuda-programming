#include <cuda_runtime.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#define M 256
#define K 512
#define N 256

#define BLOCK_SIZE 32

void initRand(float *mat, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mat[i * n + j] = (float)rand() / RAND_MAX;
        }
    }
}

double getTime() {
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime()");
        return -1;
    }

    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void matmul_cpu(float *A, float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j]; 
            }
            C[i * n + j] = sum;
        }
    }
}

__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < m && j < n) {
        float sum = 0.0f;

        for (int l = 0; l < k; l++) {
            sum += A[i * k + l] * B[l * n + j]; 
        }

        C[i * n + j] = sum;
    }
}

int main() {
    float* h_A, *h_B, *h_C_cpu, *h_C_gpu;
    float* d_A, *d_B, *d_C;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    h_A = (float*) malloc(size_A);
    h_B = (float*) malloc(size_B);
    h_C_cpu = (float*)malloc(size_C);
    h_C_gpu = (float*)malloc(size_C);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    initRand(h_A, M, K);
    initRand(h_B, K, N);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 gridDim((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    printf("Performing warm up runs...\n");
    for (int i = 0; i < 3; i++) {
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
    }

    printf("Performing CPU benchmark.\n");
    double cpu_time = 0;
    for (int i = 0; i < 20; i++) {
        double start = getTime();
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        double end = getTime();
        cpu_time += end - start;
    }
    cpu_time /= 20;

    printf("Performing GPU benchmark.\n");
    double gpu_time = 0;
    for (int i = 0; i < 20; i++) {
        double start = getTime();
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
        double end = getTime();
        gpu_time += end - start;
    }
    gpu_time /= 20;

    printf("CPU execution time: %f milliseconds\n", cpu_time);
    printf("GPU execution time: %f milliseconds\n", gpu_time);
    printf("GPU speed up over CPU: %fx\n", cpu_time / gpu_time);

    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(h_C_cpu[i*N + j] - h_C_gpu[i*N + j]) > 1e-5) {
                correct = false;
                break;
            }
        }
    }

    printf("Results are %s\n", correct ? "correct" : "incorrect");

    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
