#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <cuda_runtime.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include "../helper.h"

int THREADS;
int BLOCKS;
int NUM_VALS;


/* Define Caliper region names */
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_large = "comp_large";

int bitonic_counter = 0;


__global__ void enumerationSort(float *array, int *rank, int n, int THREADS) {
    long int k = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = k; i < n; i += THREADS){
        
        if (i < n) {
            rank[i] = 0;
            for (int j = 0; j < n; j++) {
                if (array[j] < array[i] || (array[j] == array[i] && j < i)) {
                    rank[i]++;
                }
            }
        }
    }
}

// CUDA kernel for sorting the array based on ranks
__global__ void sortArray(float *array, float *sorted_array, int *rank, int n, int THREADS) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = k; i < n; i += THREADS){
        sorted_array[rank[i]] = array[i];
    }
}


int main(int argc, char *argv[])
{
  
    CALI_MARK_BEGIN("main");
    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    BLOCKS = NUM_VALS / THREADS;
    std::string input_type = argv[3];


    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();
    const int n = NUM_VALS; // Size of the array


    CALI_MARK_BEGIN("data_init");
    // printf("cali 0\n");
    float *h_array = new float[n];
    // printf("cali 1\n");
    float *sorted_array = new float[n];

    // printf("cali 2\n");

    // Initialize the array with random values
    array_fill(h_array, n, input_type);
    // printf("cali 3\n");
    CALI_MARK_END("data_init");


    // Device arrays
    float *d_array, *sorted_array_device;
    int* d_rank;
    cudaMalloc((void**)&d_array, sizeof(float) * n);
    cudaMalloc((void**)&d_rank, sizeof(int) * n);
    cudaMalloc((void**)&sorted_array_device, sizeof(float) * n);

    // printf("Cuda arrays allocated\n");

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("cudaMemcpy");
    // Copy data from host to device
    cudaMemcpy(d_array, h_array, sizeof(float) * n, cudaMemcpyHostToDevice);
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");


    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");

    // Launch the enumeration sort kernel
    enumerationSort<<<BLOCKS, THREADS>>>(d_array, d_rank, n, THREADS);
    cudaDeviceSynchronize();
    CALI_MARK_END("comp_large");

    CALI_MARK_BEGIN("comp_large");
    // Launch the sorting kernel to rearrange the array
    sortArray<<<BLOCKS, THREADS>>>(d_array, sorted_array_device, d_rank, n, THREADS);
    cudaDeviceSynchronize();

    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");


    CALI_MARK_BEGIN("comm");
    

    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("cudaMemcpy");
    cudaMemcpy(sorted_array, sorted_array_device, sizeof(float) * n, cudaMemcpyDeviceToHost);
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END("comm_large");

    CALI_MARK_END("comm");


    CALI_MARK_BEGIN("correctness_check");
    bool correct = correctness_check(sorted_array, NUM_VALS);
    CALI_MARK_END("correctness_check");
    if(correct) printf("Array correctly sorted\n");

 
    // Clean up
    delete[] h_array;
    delete[] sorted_array;
    cudaFree(d_array);
    cudaFree(d_rank);
    cudaFree(sorted_array_device);

    CALI_MARK_END("main");


    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "EnumerationSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", 4); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", input_type); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    // adiak::value("num_procs", ); // The number of processors (MPI ranks)
    adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", 15); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
  

  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();
}