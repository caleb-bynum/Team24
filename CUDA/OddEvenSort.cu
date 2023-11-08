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

void compare_and_swap(float* A, int i, int j) {
    if (A[i] > A[j]) {
        float temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }
}

void odd_even_sort(float* A) {
    /* allocate space on GPU */
    float* device_A;
    size_t size_bytes = NUM_VALS * sizeof(float);
    cudaMalloc( (void**)&device_A, size_bytes );

    /* copy A to GPU */
    cudaMemcpy( device_A, A, size_bytes, cudaMemcpyHostToDevice );
    
}
