#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

const int THREADS_PER_BLOCK = 512;

__global__ void gpu_merge_sort(int *data, int *sorted, int n, int width) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lStart = width * tid * 2;
    int rStart = lStart + width;
    int lEnd = rStart;
    int rEnd = lEnd + width;

    if (rStart > n) rStart = n;
    if (lEnd > n) lEnd = n;
    if (rEnd > n) rEnd = n;

    int l = lStart;
    int r = rStart;
    int idx = lStart;

    while (l < lEnd && r < rEnd) {
        if (data[l] <= data[r]) {
            sorted[idx++] = data[l++];
        } else {
            sorted[idx++] = data[r++];
        }
    }

    while (l < lEnd) {
        sorted[idx++] = data[l++];
    }

    while (r < rEnd) {
        sorted[idx++] = data[r++];
    }
}

void merge_sort(int *data, int n) {
    int *d_data, *d_sorted;

    cudaMalloc(&d_data, n * sizeof(int));
    cudaMalloc(&d_sorted, n * sizeof(int));
    
    cudaMemcpy(d_data, data, n * sizeof(int), cudaMemcpyHostToDevice);
    
    int numBlocks = n / (2 * THREADS_PER_BLOCK);
    if (n % (2 * THREADS_PER_BLOCK)) numBlocks++;

    for (int width = 1; width < n; width *= 2) {
        gpu_merge_sort<<<numBlocks, THREADS_PER_BLOCK>>>(d_data, d_sorted, n, width);
        std::swap(d_data, d_sorted);
    }

    cudaMemcpy(data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_sorted);
}

int main() {
    const int N = 50;
    int *data = new int[N];

    for (int i = 0; i < N; i++) {
        data[i] = rand() % 100;
    }

        // Print the array before sorting.
    printf("Array before sorting:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", data[i]);
    }
    printf("\n\n");

    // Call the merge_sort function.
    merge_sort(data, N);

    // Print the array after sorting.
    printf("Array after sorting:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");

    // Check if data is sorted.
    for (int i = 1; i < N; i++) {
        if (data[i-1] > data[i]) {
            fprintf(stderr, "Error: data is not sorted at index %d\n", i);
        }
    }

    delete[] data;

    return 0;
}
