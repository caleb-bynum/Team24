#include <stdio.h>
#include <cuda.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

__global__ void gpu_mergesort(int *data, int *temp, int size, int width, int slices, int threads_per_slice) {
    int slice = blockIdx.x * blockDim.x + threadIdx.x;
    if (slice >= slices) return;

    int start = slice * width * 2;
    int middle = min(start + width, size);
    int end = min(start + 2 * width, size);
    int i = start;
    int j = middle;
    for (int k = start; k < end; k++) {
        if (i < middle && (j >= end || data[i] <= data[j])) {
            temp[k] = data[i];
            i++;
        } else {
            temp[k] = data[j];
            j++;
        }
    }
}

void mergesort(int *data, int size, int threads) {
    int *d_data, *d_temp;
    cudaMalloc((void **)&d_data, size * sizeof(int));
    cudaMalloc((void **)&d_temp, size * sizeof(int));

       CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    cudaMemcpy(d_data, data, size * sizeof(int), cudaMemcpyHostToDevice);
       CALI_MARK_END("comm_large");
     CALI_MARK_END("comm");

        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_large");
    int threads_per_block = threads; // Max threads per block
    int width, slices;
    for (width = 1; width < size; width *= 2) {
        slices = (size + (width * 2) - 1) / (width * 2);
        int blocks = (slices + threads_per_block - 1) / threads_per_block;
        gpu_mergesort<<<blocks, threads_per_block>>>(d_data, d_temp, size, width, slices, threads_per_block);
        cudaMemcpy(d_data, d_temp, size * sizeof(int), cudaMemcpyDeviceToDevice);
    }
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

        CALI_MARK_BEGIN("comm");
        CALI_MARK_BEGIN("comm_large");
    cudaMemcpy(data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    CALI_MARK_END("comm_large");
        CALI_MARK_END("comm");

    cudaFree(d_data);
    cudaFree(d_temp);
}

int main(int argc, char **argv) {

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

        CALI_MARK_BEGIN("main");
    if (argc != 3) {
        printf("Usage: %s <data_size> <threads>\n", argv[0]);
        return 1;
    }

    int data_size = atoi(argv[1]);
    int threads = atoi(argv[2]);

    printf("Threads Per Block: ");
    printf("%d ", threads);
    printf("\n\n");

    printf("Size of Array: ");
    printf("%d ", data_size);
    printf("\n\n");

         CALI_MARK_BEGIN("date_init");
    // Assume 'data' is already filled with random numbers
    int *data = (int *)malloc(data_size * sizeof(int));
    // Fill data with random integers for testing purposes
    for (int i = 0; i < data_size; ++i) {
        data[i] = rand() % data_size;
    }
        CALI_MARK_END("date_init");

    mergesort(data, data_size, threads);

        CALI_MARK_BEGIN("correctness_check");
    bool sorted = true;
    for (int i = 1; i < data_size; i++) {
        if (data[i] < data[i - 1]) {
            printf("Error: Data not sorted!\n");
            sorted = false;
            break;
        }
    }

    if (sorted) {
        printf("Data is sorted.\n");
                CALI_MARK_END("correctness_check");

        printf("First 10 elements:\n");
        for (int i = 0; i < 10; i++) {
            printf("%d ", data[i]);
        }
        printf("\n");

        printf("Last 10 elements:\n");
        for (int i = data_size - 10; i < data_size; i++) {
            printf("%d ", data[i]);
        }
        printf("\n");
    }

        CALI_MARK_END("main");

        int blocks = data_size / threads;

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", data_size); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", 0); // The number of processors (MPI ranks)
    adiak::value("num_threads", threads); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", blocks); // The number of CUDA blocks 
    adiak::value("group_num", 24); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI / Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

        // Flush Caliper output before finalizing MPI
      mgr.stop();
      mgr.flush();

    free(data);
    return 0;
}