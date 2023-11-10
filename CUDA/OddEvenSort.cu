/* 
CUDA Implementation Pseudocode

Variables:
    n : number of elements to sorted
    t : number of threads
    A : array to be sorted

Idea: avoid coordination between threads by strategically launching kernels
      and subsequently synchronzing.

Main Procedure:
    1. Copy A to GPU memory
    2. for each phase in {0, 1, ..., n-1}:
    3.   if phase is even:
    4.     launch EvenPhase kernel
    5.   else:
    6.     launch OddPhase kernel
    7.   synchronize device
    8. Copy A to device memory

EvenPhase Kernel:
    1. Id = threadId + blockDim * blockId
    2. index1 = 2 * Id
    3. index2 = index1 + 1
    4. compare and swap array elements at index1, index2

OddPhase Kernel:
    1. Id = threadId + blockDim * blockId
    2. index1 = 2 * Id + 1
    3. index2 = index1 + 1
    4. if thread is not the last thread
    5.   compare and swap array elements at index1, index2
*/


#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

/*
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp> 
*/

int THREADS_PER_BLOCK;
int BLOCKS;
int NUM_VALS;

float random_float() {
    return (float)rand() / (float)RAND_MAX;
}

void fill_array(float* A, int n) {
    for (int i = 0; i < n; i++) {
        A[i] = random_float();
    }
}

void print_array(float* A, int n) {
    for (int i = 0; i < n; i++) {
        printf("%f ", A[i]);
    }
    printf("\n");
}

__device__ void compare_and_swap(float* A, int i, int j) {
    if (A[i] > A[j]) {
        float temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }
}

__global__ void even_phase(float* A) {
    int Id = threadIdx.x + blockDim.x * blockIdx.x;
    int index1 = 2 * Id;
    int index2 = index1 + 1;
    compare_and_swap(A, index1, index2);
}

__global__ void odd_phase(float* A, int numVals) {
    int Id = threadIdx.x + blockDim.x * blockIdx.x;
    int index1 = 2 * Id + 1;
    int index2 = index1 + 1;
    if (index2 < numVals) {
        compare_and_swap(A, index1, index2);
    }
}

void odd_even_sort(float* A, int numVals) {
    /* allocate space on GPU */
    float* device_A;
    size_t size_bytes = NUM_VALS * sizeof(float);
    cudaMalloc( (void**)&device_A, size_bytes );

    /* copy CPU array to GPU */
    cudaMemcpy( device_A, A, size_bytes, cudaMemcpyHostToDevice );

    /* iterate through each phase, launching the appropriate kernel */
    for (int phase = 0; phase < NUM_VALS; phase++) {
        if (phase % 2 == 0) {
            even_phase<<<BLOCKS, THREADS_PER_BLOCK>>>(device_A);
        } else {
            odd_phase<<<BLOCKS, THREADS_PER_BLOCK>>>(device_A, numVals);
        }
        cudaDeviceSynchronize();
    }

    /* copy GPU array back to Host */
    cudaMemcpy( A, device_A, size_bytes, cudaMemcpyDeviceToHost );

    /* free GPU memory */
    cudaFree(device_A);
}

int main(int argc, char** argv) {
    /* initialize random seed */
    srand(time(NULL));

    /* get command line arguments */
    if (argc != 3) {
        printf("Usage: ./odd_even_sort <num_vals> <threads_per_block>\n");
        exit(1);
    }

    NUM_VALS = atoi(argv[1]);
    THREADS_PER_BLOCK = atoi(argv[2]);
    BLOCKS = NUM_VALS / THREADS_PER_BLOCK;

    /* allocate space for array */
    float* A = (float*)malloc(NUM_VALS * sizeof(float));

    /* fill array with random values */
    fill_array(A, NUM_VALS);

    /* sort array */
    odd_even_sort(A, NUM_VALS);

    /* print sorted array */
    print_array(A, NUM_VALS);

    /* free memory */
    free(A);

    return 0;
}
