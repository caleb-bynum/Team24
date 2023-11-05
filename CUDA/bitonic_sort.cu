#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>

// const int THREADS_PER_BLOCK = 1024;
int sort_check = 0;

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

void merge_sort(int *data, int n, int THREADS_PER_BLOCK) {
    int *d_data, *d_sorted;

    cudaMalloc(&d_data, n * sizeof(int));
    cudaMalloc(&d_sorted, n * sizeof(int));
    
        CALI_MARK_BEGIN("comm_send_large");
    cudaMemcpy(d_data, data, n * sizeof(int), cudaMemcpyHostToDevice);
    CALI_MARK_END("comm_send_large");

        CALI_MARK_BEGIN("comp_large");
    int numBlocks = n / (2 * THREADS_PER_BLOCK);
    if (n % (2 * THREADS_PER_BLOCK)) numBlocks++;

    for (int width = 1; width < n; width *= 2) {
        gpu_merge_sort<<<numBlocks, THREADS_PER_BLOCK>>>(d_data, d_sorted, n, width);
        std::swap(d_data, d_sorted);
    }
        CALI_MARK_END("comp_large");

        CALI_MARK_BEGIN("comm_recv_large");
    cudaMemcpy(data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
        CALI_MARK_END("comm_recv_large");	

    cudaFree(d_data);
    cudaFree(d_sorted);
}

int main(int argc, char **argv) {
      // Create caliper ConfigManager object
      cali::ConfigManager mgr;
      mgr.start();

        CALI_MARK_BEGIN("main");
    const int N = 50;
    int *data = new int[N];
        // Ensure the user provides the required argument for threads per block.
        if (argc != 2) {
            fprintf(stderr, "Usage: %s THREADS_PER_BLOCK\n", argv[0]);
            return 1;
        }

        // Parse THREADS_PER_BLOCK from the command line arguments
        int THREADS_PER_BLOCK = atoi(argv[1]);
        if (THREADS_PER_BLOCK <= 0) {
            fprintf(stderr, "THREADS_PER_BLOCK must be a positive integer\n");
            return 1;
        }

        CALI_MARK_BEGIN("date_init");
    for (int i = 0; i < N; i++) {
        data[i] = rand() % 100;
    }
        CALI_MARK_END("date_init");
	
        printf("Threads Per Block: ");
        printf("%d ", THREADS_PER_BLOCK);
        printf("\n\n");

    // Print the array before sorting.
    printf("Array before sorting:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", data[i]);
    }
    printf("\n\n");

    // Call the merge_sort function.
    merge_sort(data, N, THREADS_PER_BLOCK);

    // Print the array after sorting.
    printf("Array after sorting:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", data[i]);
    }
    printf("\n\n");
   
         CALI_MARK_BEGIN("correctness_check");
    // Check if data is sorted.
    for (int i = 1; i < N; i++) {
        if (data[i-1] > data[i]) {
            fprintf(stderr, "Error: data is not sorted at index %d\n", i);
	        sort_check++;
        }
    }

        if (sort_check == 0){
               printf("Array is Sorted!");
          }
         CALI_MARK_END("correctness_check");
    delete[] data;

        CALI_MARK_END("main");

        //printf("Main: %.6f seconds\n", palceholder);
// Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();

    return 0;
}
