#include <stdio.h>
#include <cuda.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <ctime>
#include <iostream>
#include <random>

__global__ void gpu_mergesort(float *data, float *temp, int size, int width, int slices, int threads_per_slice) {
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

void mergesort(float *data, int size, int threads) {
    float *d_data, *d_temp;
    cudaMalloc((void **)&d_data, size * sizeof(float));
    cudaMalloc((void **)&d_temp, size * sizeof(float));

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("cudaMemcpy");
    cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice);
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    int threads_per_block = threads; // Max threads per block
    int width, slices;
    for (width = 1; width < size; width *= 2) {
        slices = (size + (width * 2) - 1) / (width * 2);
        int blocks = (slices + threads_per_block - 1) / threads_per_block;
	CALI_MARK_BEGIN("comp");
	CALI_MARK_BEGIN("comp_large");
        gpu_mergesort<<<blocks, threads_per_block>>>(d_data, d_temp, size, width, slices, threads_per_block);
	CALI_MARK_END("comp_large");
    	CALI_MARK_END("comp");

        CALI_MARK_BEGIN("comm");
    	CALI_MARK_BEGIN("comm_large");
        CALI_MARK_BEGIN("cudaMemcpy");
        cudaMemcpy(d_data, d_temp, size * sizeof(float), cudaMemcpyDeviceToDevice);
        CALI_MARK_END("cudaMemcpy");
    	CALI_MARK_END("comm_large");
    	CALI_MARK_END("comm");

    }

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("cudaMemcpy");
    cudaMemcpy(data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    cudaFree(d_data);
    cudaFree(d_temp);
}

void switchArrayElements(float* data, int size, float switchPercent) {
    int numSwitches = static_cast<int>(size * switchPercent / 100);
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 eng(rd()); // Seed the generator
    std::uniform_int_distribution<> distr(0, size - 1); // Define the range

    for (int i = 0; i < numSwitches; ++i) {
        int idx1 = distr(eng);
        int idx2 = distr(eng);
        // Swap the elements
        std::swap(data[idx1], data[idx2]);
    }
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
        srand(time(NULL)); // Seed the RNG with the current time
    // Assume 'data' is already filled with random numbers
    float *data = (float *)malloc(data_size * sizeof(float));
    // Fill data with random floats for testing purposes
    for (int i = 0; i < data_size; ++i) {
        //data[i] = static_cast<float>(rand()) / RAND_MAX * data_size; // Generate random float
        data[i] = static_cast<float>(i); // generate sorted 
        //data[i] = static_cast<float>(data_size - i - 1); // Reverse sorted values
    }    

    // Perform the switch if %1 perturbed
    switchArrayElements(data, data_size, 1); // 1% of the elements to be switched 
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
          // Print the values in the data array
       //printf("Sample values in the data array:\n");
       //for (int i = 0; i < data_size; ++i) {
              //printf("%f ", data[i]);
              //if ((i + 1) % 10 == 0) { // New line for every 10 elements for better readability
              //             printf("\n");
              //}
       //}

        printf("Data is sorted.\n");
                CALI_MARK_END("correctness_check");

        printf("First 10 elements:\n");
        for (int i = 0; i < 10; i++) {
            printf("%f ", data[i]);
        }
        printf("\n");

        printf("Last 10 elements:\n");
        for (int i = data_size - 10; i < data_size; i++) {
            printf("%f ", data[i]);
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
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", data_size); // The number of elements in input dataset (1000)
    adiak::value("InputType", "1%Perturbed"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
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